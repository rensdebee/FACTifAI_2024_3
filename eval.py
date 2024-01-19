"""
Reproducibility Study of “Studying How to Efficiently and Effectively Guide Models with Explanations”

Description: This file is part of a project aiming to reproduce the study titled "Studying How to Efficiently and 
Effectively Guide Models with Explanations." The project focuses on verifying the results and methodologies 
proposed in the original study, and potentially extending or refining the study's findings.

Based on the code of orginal paper: https://github.com/sukrutrao/Model-Guidance

eval.py
"""

import torch
import os
from tqdm import tqdm
import datasets
import argparse
import torch.utils.tensorboard
import utils
import losses
import metrics
import bcos.modules
import bcos.data
import numpy as np
from typing import Any, Callable, Optional


def eval_model(model: torch.nn.Module,
               attributor: Any,
               loader: torch.utils.data.DataLoader,
               num_batches: int,
               num_classes: int,
               loss_fn: Callable,
               writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
               epoch: Optional[int] = None,
               mode: str = "bbs",
               vis_flag: bool = False) -> dict:
    """
    Evaluate a model using specified parameters and return a dictionary of metrics.

    This function supports different modes of evaluation:
    - 'bbs': Evaluates only the bounding boxes.
    - 'segment': Should be used with a special segment loader (not included in the arguments).

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        attributor (Any): An object or function used for attributing importance to the model's predictions.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset to be evaluated.
        num_batches (int): The number of batches to process for the evaluation.
        num_classes (int): The total number of classes in the classification task.
        loss_fn (Callable): The loss function used for evaluation.
        writer (Optional[torch.utils.tensorboard.writer.SummaryWriter]): Tensorboard writer for logging, defaults to None.
        epoch (Optional[int]): The current epoch in the training process, for logging purposes, defaults to None.
        mode (str): The mode of evaluation, either 'bbs' or 'segment', defaults to 'bbs'.
        vis_flag (bool): A flag to enable or disable visualization, defaults to False.

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """

    # Set model to eval mode
    model.eval()
    
    # Initialize metrics
    f1_metric = metrics.MultiLabelMetrics(num_classes=num_classes, threshold=0.0)
    bb_metric = metrics.BoundingBoxEnergyMultiple()
    seg_metric = metrics.SegmentEnergyMultiple()
    frac_metric = metrics.SegmentFractionMultiple()
    iou_metric = metrics.BoundingBoxIoUMultiple(vis_flag=vis_flag)

    # Initialize total loss
    total_loss = 0
    
    # Iterate over batches
    for batch_idx, data in enumerate(tqdm(loader)):

        # Check if the mode is bounding boxes or segmentation
        if mode == "bbs":
            test_X, test_y, test_bbs = data
        elif mode == "segment":
            test_X, test_y, test_segment, test_bbs = data

            # Set to GPU to do element wise matrix multiplication
            test_segment = test_segment.cuda() 
        else:
            raise NotImplementedError

        # Set to GPU
        test_X.requires_grad = True
        test_X = test_X.cuda()
        test_y = test_y.cuda()
        logits, features = model(test_X)

        # Calculate loss
        loss = loss_fn(logits, test_y).detach()
        total_loss += loss
        f1_metric.update(logits, test_y)

        # Compute attributions and update metrics
        if attributor:

            # Loop over images in batch
            for img_idx, image in enumerate(test_X):

                # Get target class of the image
                class_target = torch.where(test_y[img_idx] == 1)[0]

                # Loop over target classes
                for pred_idx, pred in enumerate(class_target):

                    # Compute attributions given the target class, logits and features
                    attributions = (
                        attributor(features, logits, pred, img_idx)
                        .detach()
                        .squeeze(0)
                        .squeeze(0)
                    )

                    # Create bounding box list
                    bb_list = utils.filter_bbs(test_bbs[img_idx], pred)

                    # if no bounding boxes are found return
                    if len(bb_list) == 0:
                        print(pred)
                        print(test_bbs[img_idx])
                        print(len(bb_list))
                        raise ValueError
                    
                    # Update Bounding Box Energy metric and IoU metric
                    bb_metric.update(attributions, bb_list)
                    iou_metric.update(attributions, bb_list, image, pred)

                    # Update segmentation metrics if mode is segmentation
                    if mode == "segment":
                        seg_metric.update(
                            attributions=attributions,
                            mask=test_segment[img_idx],
                            label=pred + 1,
                        )

                        frac_metric.update(
                            attributions=attributions,
                            mask=test_segment[img_idx],
                            label=pred + 1,
                            bb_coordinates=bb_list,
                        )

    # Compute F1 metric
    metric_vals = f1_metric.compute()

    # If attributor is not None, compute BB-Loc and BB-IoU metrics
    if attributor:
        bb_metric_vals = bb_metric.compute()
        iou_metric_vals = iou_metric.compute()
        metric_vals["BB-Loc"] = bb_metric_vals
        metric_vals["BB-IoU"] = iou_metric_vals

        # If mode is segmentation, compute BB-Loc-segment and BB-Loc-Fraction metrics
        if mode == "segment":
            seg_bb_metric_vals = seg_metric.compute()
            seg_bb_metric_frac = frac_metric.compute()
            metric_vals["BB-Loc-segment"] = seg_bb_metric_vals
            metric_vals["BB-Loc-Fraction"] = seg_bb_metric_frac

    # Compute average loss
    metric_vals["Average-Loss"] = total_loss.item() / num_batches

    # Print metrics
    print(f"Validation Metrics: {metric_vals}")

    # Set model back to train mode
    model.train()
    
    # Log metrics to tensorboard
    if writer is not None:
        writer.add_scalar("val_loss", total_loss.item() / num_batches, epoch)
        writer.add_scalar("accuracy", metric_vals["Accuracy"], epoch)
        writer.add_scalar("precision", metric_vals["Precision"], epoch)
        writer.add_scalar("recall", metric_vals["Recall"], epoch)
        writer.add_scalar("fscore", metric_vals["F-Score"], epoch)
        if attributor:
            writer.add_scalar("bbloc", metric_vals["BB-Loc"], epoch)
            writer.add_scalar("bbiou", metric_vals["BB-IoU"], epoch)

    return metric_vals


def evaluation_function(model_path: str,
                        fix_layer: Optional[str] = None,
                        pareto: bool = False,
                        eval_batch_size: int = 4,
                        data_path: str = "datasets/",
                        dataset: str = "VOC2007",
                        split: str = "test",
                        annotated_fraction: float = 1,
                        log_path: Optional[str] = None,
                        mode: str = "bbs",
                        npz: bool = False,
                        vis_iou_thr_methods: bool = False,
                        baseline: bool = False) -> dict:
    """
    Evaluate a model's performance on a specific dataset split and return the metrics.

    This function focuses on evaluating the model on different dataset splits and does not log the metrics to TensorBoard. For TensorBoard logging, use a separate command-line script.

    Args:
        model_path (str): Path to the model to be evaluated.
        fix_layer (Optional[str]): Specific layer to focus on during evaluation, defaults to None.
        pareto (bool): Flag to indicate if Pareto optimization is applied, defaults to False.
        eval_batch_size (int): Batch size for evaluation, defaults to 4.
        data_path (str): Path to the dataset directory, defaults to "datasets/".
        dataset (str): Name of the dataset to be used, defaults to "VOC2007".
        split (str): The specific split of the dataset to evaluate on (e.g., "test"), defaults to "test".
        annotated_fraction (float): Fraction of the data that is annotated, defaults to 1.
        log_path (Optional[str]): Path for logging, defaults to None.
        mode (str): Evaluation mode, can be "bbs" or other specific modes, defaults to "bbs".
        npz (bool): Flag to indicate whether to use NPZ for evaluation, defaults to False.
        vis_iou_thr_methods (bool): Flag to enable/disable visualization of IoU threshold methods, defaults to False.
        baseline (bool): Flag to indicate whether a baseline model is used for comparison, defaults to False.

    Returns:
        dict: A dictionary containing the evaluation metrics for the specified model and dataset split.
    """

    # Get model specs
    (
        model_backbone,
        localization_loss_fn,
        layer,
        attribution_method,
    ) = utils.get_model_specs(model_path)

    # If no attribution method set default values
    if not attribution_method:
        if model_backbone == "bcos":
            attribution_method = "BCos"
        elif model_backbone == "vanilla":
            attribution_method = "IxG"

    # Default localistion loss is energy
    if not localization_loss_fn:
        localization_loss_fn = "Energy"

    # If fix_layer is not None, set layer to fix_layer
    if fix_layer:
        layer = fix_layer

    # Print model specs
    print(
        "Found model with specs: ",
        model_backbone,
        localization_loss_fn,
        layer,
        attribution_method,
        dataset,
    )

    # Get model, attributor and transformer
    model_activator, eval_attributor, transformer = utils.get_model(
        model_backbone,
        localization_loss_fn,
        layer,
        attribution_method,
        dataset,
        model_path=model_path,
    )

    # Load dataset base on --split argument
    root = os.path.join(data_path, dataset, "processed")

    # Get dataloader
    if split == "train":
        data = datasets.VOCDetectParsed(
            root=root,
            image_set="train",
            transform=transformer,
            annotated_fraction=annotated_fraction,
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=eval_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=datasets.VOCDetectParsed.collate_fn,
        )
        num_batches = len(data) / eval_batch_size

    # If split is val, use the validation dataset
    elif split == "val":
        data = datasets.VOCDetectParsed(
            root=root, image_set="val", transform=transformer
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=datasets.VOCDetectParsed.collate_fn,
        )
        num_batches = len(data) / eval_batch_size

    # If split is test, use the test dataset
    elif split == "test":
        data = datasets.VOCDetectParsed(
            root=root, image_set="test", transform=transformer
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=datasets.VOCDetectParsed.collate_fn,
        )
        num_batches = len(data) / eval_batch_size

    # If split is seg_test, use the segment dataset
    elif split == "seg_test":
        data = datasets.SegmentDataset(
            root=root + "/all_segment.pt", transform=transformer
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=datasets.SegmentDataset.collate_fn,
        )
        num_batches = len(data) / eval_batch_size
    else:
        raise NotImplementedError(
            f'Data split not valid choose from ["train", "val", "test", "seg_test"] but received "{split}"'
        )

    # Define Binary Cross Entropy loss function to be used for the split
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Get number of classes
    num_classes_dict = {"VOC2007": 20, "COCO2014": 80}
    num_classes = num_classes_dict[dataset]

    # Create TensorBoard writer
    if log_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=os.path.join(log_path, dataset)
        )
    else:
        writer = None

    # Evaluate model
    metric_vals = eval_model(
        model_activator,
        eval_attributor,
        loader,
        num_batches,
        num_classes,
        loss_fn,
        writer,
        1,
        mode=mode,
        vis_flag=vis_iou_thr_methods,
    )

    # Save metrics as .npz file in log_path
    if npz:

        # Save metrics as .npz file in log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        # If pareto is true, add pareto and epoch tothe file name
        if pareto:
            if attribution_method:
                epoch = model_path.split("_")[-1].split(".")[0]
                npz_name = f"{dataset}_{split}_{model_backbone}_{localization_loss_fn}_{layer}_{attribution_method}_Pareto_{epoch}.npz"
            else:
                npz_name = f"{dataset}_{split}_{model_backbone}_Pareto_{epoch}.npz"

        # If baseline is true, add baseline to the file name
        elif baseline:

            if attribution_method:
                epoch = model_path.split("_")[-1].split(".")[0]
                npz_name = f"{dataset}_{split}_{model_backbone}_{localization_loss_fn}_{layer}_{attribution_method}_Baseline.npz"
            else:
                npz_name = f"{dataset}_{split}_{model_backbone}_Baseline.npz"

        else:
            if attribution_method:
                npz_name = f"{dataset}_{split}_{model_backbone}_{localization_loss_fn}_{layer}_{attribution_method}.npz"
            else:
                npz_name = f"{dataset}_{split}_{model_backbone}.npz"

        # Create path to save .npz file
        npz_path = os.path.join(log_path, npz_name)

        # Save .npz file
        np.savez(npz_path, **metric_vals)

    return metric_vals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to checkpoint to eval",
        required=True,
    )
    parser.add_argument(
        "--fix_layer",
        type=str,
        default=None,
        choices=["Input", "Final"],
        help="Layer to get attributions from",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        default=False,
        help="Flag for storing pareto models",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size to use for evaluation.",
    )
    parser.add_argument(
        "--data_path", type=str, default="datasets/", help="Path to datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="VOC2007",
        choices=["VOC2007", "COCO2014"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "seg_test"],
        help="Set to evaluate on",
    )
    parser.add_argument(
        "--annotated_fraction",
        type=float,
        default=1.0,
        help="Fraction of training dataset from which bounding box annotations are to be used.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Path to save TensorBoard logs/npz file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="bbs",
        choices=[
            "bbs",
            "segment",
        ],
        help="mode indicates if also evaluating on sementation mask",
    )
    parser.add_argument(
        "--npz",
        action="store_true",
        default=False,
        help="Flag for storing results in .npz file",
    )
    parser.add_argument(
        "--vis_iou_thr_methods",
        action="store_true",
        default=False,
        help="Flag for displaying the different IoU threshold methods on first image in the batch",
    )

    args = parser.parse_args()
    evaluation_function(**vars(args))