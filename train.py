"""
Reproducibility Study of “Studying How to Efficiently and Effectively Guide Models with Explanations”

Description: This file is part of a project aiming to reproduce the study titled "Studying How to Efficiently and 
Effectively Guide Models with Explanations." The project focuses on verifying the results and methodologies 
proposed in the original study, and potentially extending or refining the study's findings.

Based on the code of orginal paper: https://github.com/sukrutrao/Model-Guidance

train.py
"""

import torch
import os
import argparse
import torchvision
from tqdm import tqdm
import datasets
import argparse
import torch.utils.tensorboard
import utils
import copy
import losses
import metrics
import bcos.models
import model_activators
import attribution_methods
import hubconf
import bcos
import bcos.modules
import bcos.data
import fixup_resnet
from typing import Any, Callable, Optional
from eval import eval_model
from torchvision.transforms import v2


# def eval_model(
#     model: torch.nn.Module,
#     attributor: Any,
#     loader: torch.utils.data.DataLoader,
#     num_batches: int,
#     num_classes: int,
#     loss_fn: Callable,
#     writer: Optional[torch.utils.tensorboard.writer.SummaryWriter] = None,
#     epoch: Optional[int] = None,
# ) -> dict:
#     """
#     Evaluate a model given a dataloader and an attribution method.

#     Args:
#         model (torch.nn.Module): The model to be evaluated.
#         attributor (Any): An object or a function used for attribution, specific to the model's framework.
#         loader (torch.utils.data.DataLoader): DataLoader object for loading the dataset.
#         num_batches (int): Number of batches to evaluate the model on.
#         num_classes (int): Number of classes in the classification task.
#         loss_fn (Callable): Loss function used for evaluation.
#         writer (Optional[torch.utils.tensorboard.writer.SummaryWriter]): Tensorboard writer for logging purposes. Defaults to None.
#         epoch (Optional[int]): Current epoch number, used for logging. Defaults to None.

#     Returns:
#         dict: A dictionary containing evaluation metrics such as accuracy and loss.
#     """

#     # Set model to evaluation mode
#     model.eval()

#     # Initialize metrics
#     f1_metric = metrics.MultiLabelMetrics(num_classes=num_classes, threshold=0.0)
#     bb_metric = metrics.BoundingBoxEnergyMultiple()
#     iou_metric = metrics.BoundingBoxIoUMultiple()

#     # Initialize total loss
#     total_loss = 0

#     # Iterate over batches
#     for batch_idx, (test_X, test_y, test_bbs) in enumerate(tqdm(loader)):
#         test_X.requires_grad = True
#         test_X = test_X.cuda()
#         test_y = test_y.cuda()
#         if len(test_y.shape) == 1:
#                 test_y = test_y.unsqueeze(dim=1)
#         logits, features = model(test_X)
#         loss = loss_fn(logits, test_y).detach()
#         total_loss += loss
#         f1_metric.update(logits, test_y)

#         # Compute attributions and update metrics
#         if attributor:
#             # Loop over images in batch
#             for img_idx in range(len(test_X)):
#                 # Get target class
#                 class_target = torch.where(test_y[img_idx] == 1)[0]

#                 # Loop over target classes
#                 for pred_idx, pred in enumerate(class_target):
#                     # Compute attributions given the target class, logits, and features
#                     attributions = (
#                         attributor(features, logits, pred, img_idx)
#                         .detach()
#                         .squeeze(0)
#                         .squeeze(0)
#                     )

#                     # Create bounding box list
#                     bb_list = utils.filter_bbs(test_bbs[img_idx], pred)

#                     # if no bounding boxes are found return
#                     if len(bb_list) == 0:
#                         print(pred)
#                         print(test_bbs[img_idx])
#                         print(len(bb_list))
#                         raise ValueError

#                     # Update Bounding Box Energy metric and IoU metric
#                     bb_metric.update(attributions, bb_list)
#                     iou_metric.update(attributions, bb_list)

#     # Compute F1 Score
#     metric_vals = f1_metric.compute()

#     # If attributor is not None, compute BB-Loc and BB-IoU metrics
#     if attributor:
#         bb_metric_vals = bb_metric.compute()
#         iou_metric_vals = iou_metric.compute()
#         metric_vals["BB-Loc"] = bb_metric_vals
#         metric_vals["BB-IoU"] = iou_metric_vals

#     # Compute Average Loss
#     metric_vals["Average-Loss"] = total_loss.item() / num_batches

#     # Update Bounding Box Energy metric and IoU metric
#     bb_metric.update(attributions, bb_list)
#     iou_metric.update(attributions, bb_list)

#     # Set model to training mode
#     model.train()

#     # Log metrics
#     if writer is not None:
#         writer.add_scalar("val_loss", total_loss.item() / num_batches, epoch)
#         writer.add_scalar("accuracy", metric_vals["Accuracy"], epoch)
#         writer.add_scalar("precision", metric_vals["Precision"], epoch)
#         writer.add_scalar("recall", metric_vals["Recall"], epoch)
#         writer.add_scalar("fscore", metric_vals["F-Score"], epoch)
#         if attributor:
#             writer.add_scalar("bbloc", metric_vals["BB-Loc"], epoch)
#             writer.add_scalar("bbiou", metric_vals["BB-IoU"], epoch)

#     return metric_vals


def main(args: argparse.Namespace):
    """
    Main function for training a model.

    Args:
        args (argparse.Namespace): Command line arguments.
    """

    # Set random seed
    utils.set_seed(args.seed)

    # Get number of classes
    num_classes_dict = {"VOC2007": 20, "COCO2014": 80, "WATERBIRDS": 2}
    num_classes = num_classes_dict[args.dataset]

    # Get model
    is_bcos = args.model_backbone == "bcos"
    is_xdnn = args.model_backbone == "xdnn"
    is_vanilla = args.model_backbone == "vanilla"

    # If model is BCOS, load BCOS ResNet50
    if is_bcos:
        model = hubconf.resnet50(pretrained=True)
        model[0].fc = bcos.modules.bcosconv2d.BcosConv2d(
            in_channels=model[0].fc.in_channels, out_channels=num_classes
        )
        layer_dict = {"Input": None, "Mid1": 3, "Mid2": 4, "Mid3": 5, "Final": 6}

    # If model is XDNN, load XDNN ResNet50
    elif is_xdnn:
        model = fixup_resnet.xfixup_resnet50()
        imagenet_checkpoint = torch.load(
            os.path.join("weights/xdnn/xfixup_resnet50_model_best.pth.tar")
        )
        imagenet_state_dict = utils.remove_module(imagenet_checkpoint["state_dict"])
        model.load_state_dict(imagenet_state_dict)
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features, out_features=num_classes
        )
        layer_dict = {"Input": None, "Mid1": 3, "Mid2": 4, "Mid3": 5, "Final": 6}

    # If model is vanilla, load vanilla ResNet50
    elif is_vanilla:
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features, out_features=num_classes
        )
        layer_dict = {"Input": None, "Mid1": 4, "Mid2": 5, "Mid3": 6, "Final": 7}

    # If model is not supported, raise NotImplementedError
    else:
        raise NotImplementedError

    # Get layer index
    layer_idx = layer_dict[args.layer]
    # Load model checkpoint if provided
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint["model"])
    # Move model to GPU and set to training mode
    model = model.cuda()
    model.train()

    # Get model name for saving
    orig_name = os.path.basename(args.model_path) if args.model_path else str(None)

    # Get model prefix for saving
    model_prefix = args.model_backbone

    # Get optimization string for saving
    optimize_explanation_str = (
        "finetunedobjloc" if args.optimize_explanations else "standard"
    )
    optimize_explanation_str += "pareto" if args.pareto else ""
    optimize_explanation_str += "_limited" if args.annotated_fraction < 1.0 else ""
    optimize_explanation_str += "_dilated" if args.box_dilation_percentage > 0 else ""

    # Get save path
    out_name = (
        model_prefix
        + "_"
        + optimize_explanation_str
        + "_attr"
        + str(args.attribution_method)
        + "_locloss"
        + str(args.localization_loss_fn)
        + "_orig"
        + orig_name
        + "_resnet50"
        + "_lr"
        + str(args.learning_rate)
        + "_sll"
        + str(args.localization_loss_lambda)
        + "_layer"
        + str(args.layer)
    )

    # Add annotated fraction and box dilation percentage to save path
    if args.annotated_fraction < 1.0:
        out_name += f"limited{args.annotated_fraction}"
    if args.box_dilation_percentage > 0:
        out_name += f"_dilation{args.box_dilation_percentage}"

    # Create save path
    save_path = os.path.join(args.save_path, args.dataset, out_name)
    os.makedirs(save_path, exist_ok=True)

    # Create TensorBoard writer
    if args.log_path is not None:
        writer = torch.utils.tensorboard.SummaryWriter(
            log_dir=os.path.join(args.log_path, args.dataset, out_name)
        )
    else:
        writer = None

    # Create data loaders and if model is BCOS, add inverse normalization
    if is_bcos:
        transformer = bcos.data.transforms.AddInverse(dim=0)
    else:
        transformer = v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    transformer_bbs = None
    if args.dataset == "WATERBIRDS":
        transformer_bbs = v2.Compose(
            [
                v2.RandomResizedCrop(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(p=0.5),
            ]
        )

    # Create data loaders for train, val, and test sets
    root = os.path.join(args.data_path, args.dataset, "processed")
    train_data = datasets.VOCDetectParsed(
        root=root,
        image_set="train",
        transform=transformer,
        annotated_fraction=args.annotated_fraction,
        bbs_transform=transformer_bbs,
    )
    val_data = datasets.VOCDetectParsed(
        root=root, image_set="val", transform=transformer
    )
    test_data = datasets.VOCDetectParsed(
        root=root, image_set="test", transform=transformer
    )

    # Print dataset statistics
    print(f"Train data size: {len(train_data)}")

    # Determine number of annotated images
    annotation_count = 0
    total_count = 0

    # Iterate over train data and count number of annotated images
    for idx in range(len(train_data)):
        if train_data[idx][2] is not None:
            annotation_count += 1
        total_count += 1

    # Print number of annotated images
    print(f"Annotated: {annotation_count}, Total: {total_count}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=datasets.VOCDetectParsed.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=datasets.VOCDetectParsed.collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=datasets.VOCDetectParsed.collate_fn,
    )

    # Get number of batches
    num_train_batches = len(train_data) / args.train_batch_size
    num_val_batches = len(val_data) / args.eval_batch_size
    num_test_batches = len(test_data) / args.eval_batch_size

    # Define Binary Cross Entropy loss function and localization loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_loc = (
        losses.get_localization_loss(args.localization_loss_fn)
        if args.localization_loss_fn
        else None
    )

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Define metric trackers
    if args.dataset == "WATERBIRDS":
        f1_tracker = utils.BestMetricTracker("Accuracy")
    else:
        f1_tracker = utils.BestMetricTracker("F-Score")
    # Define model activator and attributor
    model_activator = model_activators.ResNetModelActivator(
        model=model, layer=layer_idx, is_bcos=is_bcos
    )

    # If attribution method is not None, get attributor
    if args.attribution_method:
        interpolate = True if layer_idx is not None else False
        attributor = attribution_methods.get_attributor(
            model,
            args.attribution_method,
            loss_loc.only_positive,
            loss_loc.binarize,
            interpolate,
            (224, 224),
            batch_mode=True,
        )
        eval_attributor = attribution_methods.get_attributor(
            model,
            args.attribution_method,
            loss_loc.only_positive,
            loss_loc.binarize,
            interpolate,
            (224, 224),
            batch_mode=False,
        )
    else:
        attributor = None
        eval_attributor = None

    # If pareto is True, create Pareto front tracker
    # With the corresponding metric (F1 vs args.pareto_metric) to base the front on
    if args.pareto:
        # Create a pareto front based on both IOU and EPG seperately VS F1
        if args.pareto_metric == "seperate":
            pareto_front_tracker_EPG = utils.ParetoFrontModels(epg=True, iou=False)
            pareto_front_tracker_IOU = utils.ParetoFrontModels(epg=False, iou=True)
        else:
            # Pareto front of IOU and EPG as one combined vs F1
            if args.pareto_metric == "EPG_IOU":
                epg = True
                iou = True
            # Pareto front created with just EPG vs F1
            elif args.pareto_metric == "EPG":
                epg = True
                iou = False
            # Pareto front created with just IOU vs F1
            elif args.pareto_metric == "IOU":
                epg = False
                iou = True
            pareto_front_tracker = utils.ParetoFrontModels(epg=epg, iou=iou)

    # Train model for total_epochs epochs
    for e in tqdm(range(args.total_epochs)):
        # Initialize total loss, total class loss, and total localization loss
        total_loss = 0
        total_class_loss = 0
        total_localization_loss = 0

        # Iterate over batches
        for batch_idx, (train_X, train_y, train_bbs) in enumerate(tqdm(train_loader)):
            # Initialize batch loss and localization loss
            batch_loss = 0
            localization_loss = 0
            optimizer.zero_grad()

            # os.system(f"nvidia-smi")
            # Compute logits and features
            train_X.requires_grad = True
            train_X = train_X.cuda()
            train_y = train_y.cuda()

            # Compute attributions and update localization loss
            logits, features = model_activator(train_X)
            loss = loss_fn(logits, train_y)
            batch_loss += loss
            total_class_loss += loss.detach()

            # Compute attributions and update localization loss
            if args.optimize_explanations:
                # Get ground truth classes
                gt_classes = utils.get_random_optimization_targets(train_y)
                # Compute attributions given the ground truth classes, logits, and features
                attributions = attributor(features, logits, classes=gt_classes).squeeze(
                    1
                )

                # Loop over images in batch
                for img_idx in range(len(train_X)):
                    if train_bbs[img_idx] is None:
                        continue

                    # Create bounding box list
                    bb_list = utils.filter_bbs(train_bbs[img_idx], gt_classes[img_idx])

                    # Dilate bounding boxes if specified
                    if args.box_dilation_percentage > 0:
                        bb_list = utils.enlarge_bb(
                            bb_list, percentage=args.box_dilation_percentage
                        )

                    # Update localization loss
                    localization_loss += loss_loc(attributions[img_idx], bb_list)

                # Update batch loss
                batch_loss += args.localization_loss_lambda * localization_loss

                # Update total localization loss
                if torch.is_tensor(localization_loss):
                    total_localization_loss += localization_loss.detach()
                else:
                    total_localization_loss += localization_loss

            # Backpropagate and update parameters
            batch_loss.backward()
            total_loss += batch_loss.detach()
            optimizer.step()

        # Print average loss
        print(f"Epoch: {e}, Average Loss: {total_loss / num_train_batches}")

        # Log metrics
        if writer:
            writer.add_scalar("train_loss", total_loss, e + 1)
            writer.add_scalar("class_loss", total_class_loss, e + 1)
            writer.add_scalar("localization_loss", total_localization_loss, e + 1)

        # If evaluation frequency is reached, evaluate model
        if (e + 1) % args.evaluation_frequency == 0:
            # Evaluate model
            metric_vals = eval_model(
                model_activator,
                eval_attributor,
                val_loader,
                num_val_batches,
                num_classes,
                loss_fn,
                writer,
                e,
            )

            # If pareto is True, update Pareto front tracker
            if args.pareto:
                if args.pareto_metric == "seperate":
                    pareto_front_tracker_EPG.update(model, metric_vals, e)
                    pareto_front_tracker_IOU.update(model, metric_vals, e)
                else:
                    pareto_front_tracker.update(model, metric_vals, e)

            # Get best F-Score
            best_fscore, _, _, _ = f1_tracker.get_best()

            # If best F-Score is below threshold, stop training
            if (best_fscore is not None) and (best_fscore < args.min_fscore):
                print(
                    f'F-Score below threshold, actual: {metric_vals["F-Score"]}, threshold: {args.min_fscore}'
                )

                # Update metric values
                metric_vals.update({"model": None, "epochs": e + 1} | vars(args))

                # Update metric values
                metric_vals.update({"BelowThresh": True})

                # Save model checkpoint
                torch.save(
                    metric_vals,
                    os.path.join(save_path, f"model_checkpoint_stopped_{e+1}.pt"),
                )

                # If pareto is True, save Pareto front
                if args.pareto:
                    if args.pareto_metric == "seperate":
                        pareto_front_tracker_EPG.save_pareto_front(save_path)
                        pareto_front_tracker_IOU.save_pareto_front(save_path)
                    else:
                        pareto_front_tracker.save_pareto_front(save_path)
                return

            # Update metric values
            f1_tracker.update(metric_vals, model, e)

    # If pareto is True, save Pareto front
    if args.pareto:
        if args.pareto_metric == "seperate":
            pareto_front_tracker_EPG.save_pareto_front(save_path)
            pareto_front_tracker_IOU.save_pareto_front(save_path)
        else:
            pareto_front_tracker.save_pareto_front(save_path)

    # Calculate final metrics
    final_metric_vals = metric_vals
    final_metric_vals = utils.update_val_metrics(final_metric_vals)

    # Evaluate model
    final_metrics = eval_model(
        model_activator,
        eval_attributor,
        test_loader,
        num_test_batches,
        num_classes,
        loss_fn,
    )

    # Create final metrics dictionary
    final_state_dict = copy.deepcopy(model.state_dict())
    final_metrics.update(final_metric_vals)
    final_metrics.update({"model": final_state_dict, "epochs": e + 1} | vars(args))

    # Get best F-Score
    (
        f1_best_score,
        f1_best_model_dict,
        f1_best_epoch,
        f1_best_metric_vals,
    ) = f1_tracker.get_best()
    f1_best_metric_vals = utils.update_val_metrics(f1_best_metric_vals)

    # Load best model and evaluate
    model.load_state_dict(f1_best_model_dict)
    f1_best_metrics = eval_model(
        model_activator,
        eval_attributor,
        test_loader,
        num_test_batches,
        num_classes,
        loss_fn,
    )

    # Update metric values to include best F-Score
    f1_best_metrics.update(f1_best_metric_vals)
    f1_best_metrics.update(
        {"model": f1_best_model_dict, "epochs": f1_best_epoch + 1} | vars(args)
    )

    # Save model checkpoints
    torch.save(
        final_metrics, os.path.join(save_path, f"model_checkpoint_final_{e+1}.pt")
    )
    torch.save(f1_best_metrics, os.path.join(save_path, f"model_checkpoint_f1_best.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_backbone",
        type=str,
        choices=["bcos", "xdnn", "vanilla"],
        required=True,
        help="Model backbone to train.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to checkpoint to fine tune from. When None, a model is trained starting from ImageNet pre-trained weights.",
    )
    parser.add_argument(
        "--data_path", type=str, default="datasets/", help="Path to datasets."
    )
    parser.add_argument(
        "--total_epochs", type=int, default=100, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate to use."
    )
    parser.add_argument(
        "--log_path", type=str, default=None, help="Path to save TensorBoard logs."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/",
        help="Path to save trained models.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed to use.")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["VOC2007", "COCO2014", "WATERBIRDS"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--localization_loss_lambda",
        type=float,
        default=1.0,
        help="Lambda to use to weight localization loss.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="Input",
        choices=["Input", "Final", "Mid1", "Mid2", "Mid3"],
        help="Layer of the model to compute and optimize attributions on.",
    )
    parser.add_argument(
        "--localization_loss_fn",
        type=str,
        default=None,
        choices=["Energy", "L1", "RRR", "PPCE"],
        help="Localization loss function to use.",
    )
    parser.add_argument(
        "--attribution_method",
        type=str,
        default=None,
        choices=["BCos", "GradCam", "IxG"],
        help="Attribution method to use for optimization.",
    )
    parser.add_argument(
        "--optimize_explanations",
        action="store_true",
        default=False,
        help="Flag for optimizing attributions. When False, a model is trained just using the classification loss.",
    )
    parser.add_argument(
        "--min_fscore",
        type=float,
        default=-1,
        help="Minimum F-Score the best model so far must have to continue training. If the best F-Score drops below this threshold, stops training early.",
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        default=False,
        help="Flag to save Pareto front of models based on F-Score, EPG Score, and IoU Score.",
    )
    parser.add_argument(
        "--annotated_fraction",
        type=float,
        default=1.0,
        help="Fraction of training dataset from which bounding box annotations are to be used.",
    )
    parser.add_argument(
        "--evaluation_frequency",
        type=int,
        default=1,
        help="Frequency (number of epochs) at which to evaluate the current model.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size to use for evaluation.",
    )
    parser.add_argument(
        "--box_dilation_percentage",
        type=float,
        default=0,
        help="Fraction of dilation to use for bounding boxes when training.",
    )
    parser.add_argument(
        "--pareto_metric",
        type=str,
        default="EPG_IOU",
        choices=["EPG_IOU", "EPG", "IOU", "seperate"],
        help="Select the metric to create a pareto front with -> F1 vs choice. EPG_IOU evaluates both metrics as one, while [seperate] evaluates both metrics seperate.",
    )
    args = parser.parse_args()
    main(args)
