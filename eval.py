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


def eval_model(
    model,
    attributor,
    loader,
    num_batches,
    num_classes,
    loss_fn,
    writer=None,
    epoch=None,
    mode="bbs",
    vis_flag=False,
):
    """ 
    Evaluates a model, helper function of evaluation_function.
    if mode is set to bbs: only bounding boxes are evaluated.
    if mode is set to segment: The specail segment loader should be added.
    
    Returns: 
        A dictionary with all evaluation metrics.
    """
    model.eval()
    
    # create all metrics
    f1_metric = metrics.MultiLabelMetrics(num_classes=num_classes, threshold=0.0)
    bb_metric = metrics.BoundingBoxEnergyMultiple()
    seg_metric = metrics.SegmentEnergyMultiple()
    frac_metric = metrics.SegmentFractionMultiple()
    iou_metric = metrics.BoundingBoxIoUMultiple(vis_flag=vis_flag)
    total_loss = 0
    
    # for every image, evaluate 
    for batch_idx, data in enumerate(tqdm(loader)):
        if mode == "bbs":
            test_X, test_y, test_bbs = data
        elif mode == "segment":
            test_X, test_y, test_segment, test_bbs = data
            test_segment = test_segment.cuda() # to GPU to do element wise matrix multiplication
        else:
            raise NotImplementedError

        # push to GPU
        test_X.requires_grad = True
        test_X = test_X.cuda()
        test_y = test_y.cuda()
        logits, features = model(test_X)

        loss = loss_fn(logits, test_y).detach()
        total_loss += loss
        f1_metric.update(logits, test_y)

        # add metrics that are attributor specific
        if attributor:
            for img_idx, image in enumerate(test_X):
                class_target = torch.where(test_y[img_idx] == 1)[0]
                for pred_idx, pred in enumerate(class_target):
                    attributions = (
                        attributor(features, logits, pred, img_idx)
                        .detach()
                        .squeeze(0)
                        .squeeze(0)
                    )
                    bb_list = utils.filter_bbs(test_bbs[img_idx], pred)

                    if len(bb_list) == 0:  # if no bounding boxes are found return
                        print(pred)
                        print(test_bbs[img_idx])
                        print(len(bb_list))
                        raise ValueError
                    
                    bb_metric.update(attributions, bb_list)
                    iou_metric.update(attributions, bb_list, image, pred)
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

    # finalize metric calculations, 
    metric_vals = f1_metric.compute()
    if attributor:
        bb_metric_vals = bb_metric.compute()
        iou_metric_vals = iou_metric.compute()

        metric_vals["BB-Loc"] = bb_metric_vals
        if mode == "segment":
            seg_bb_metric_vals = seg_metric.compute()
            seg_bb_metric_frac = frac_metric.compute()
            metric_vals["BB-Loc-segment"] = seg_bb_metric_vals
            metric_vals["BB-Loc-Fraction"] = seg_bb_metric_frac

        metric_vals["BB-IoU"] = iou_metric_vals

    metric_vals["Average-Loss"] = total_loss.item() / num_batches
    print(f"Validation Metrics: {metric_vals}")
    model.train()
    
    # write to tensorboard
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


def evaluation_function(
    model_path,
    fix_layer=None,
    pareto=False,
    eval_batch_size=4,
    data_path="datasets/",
    dataset="VOC2007",
    split="test",
    annotated_fraction=1,
    log_path=None,
    mode="bbs",
    npz=False,
    vis_iou_thr_methods=False,
):
    """
    Function which returns the metrics of a given model in model_path
    on certain data split.

    Note: this function does not save metrics in tensorboard log, use the
    commandine script for that functionality.
    """
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

    # default localistion loss is energy
    if not localization_loss_fn:
        localization_loss_fn = "Energy"

    if fix_layer:
        layer = fix_layer
    print(
        "Found model with specs: ",
        model_backbone,
        localization_loss_fn,
        layer,
        attribution_method,
        dataset,
    )
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
    elif split == "seg_test":  # NEW
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

    # Get loss function to calculate loss of split
    loss_fn = torch.nn.BCEWithLogitsLoss()

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
    if npz:
        # Save metrics as .npz file in log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if pareto:
            epoch = model_path.split("_")[-1].split(".")[0]
            if attribution_method:
                npz_name = f"{dataset}_{split}_{model_backbone}_{localization_loss_fn}_{layer}_{attribution_method}_Pareto_{epoch}.npz"
            else:
                npz_name = f"{dataset}_{split}_{model_backbone}_Pareto_{epoch}.npz"

        else:
            if attribution_method:
                npz_name = f"{dataset}_{split}_{model_backbone}_{localization_loss_fn}_{layer}_{attribution_method}.npz"
            else:
                npz_name = f"{dataset}_{split}_{model_backbone}.npz"

        npz_path = os.path.join(log_path, npz_name)

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

    # print("Baseline:")
    # evaluation_function(
    #     "./BASE/VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput/model_checkpoint_f1_best.pt",
    #     pareto=True,
    #     split="seg_test",
    #     mode="segment",
    # )
    # print("Finetune:")
    # evaluation_function(
    #     "./FT/VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerFinal/model_checkpoint_f1_best.pt",
    #     split="seg_test",
    #     mode="segment",
    # )

    # # Root directory
    # root_dir = "/home/roan/Documents/FACTifAI_2024_3/FT/VOC2007/"

    # # Regular expression to match and extract information from file names
    # pattern = re.compile(
    #     r'(\w+)_finetunedobjlocpareto_attr(\w+)_locloss(\w+)_origmodel_checkpoint_f1_best.pt_(\w+)_lr([0-9.e-]+)_sll([0-9.e-]+)_layer(\w+)/pareto_front/model_checkpoint_pareto_(.+).pt'
    # )

    # # Walk through the directory
    # for subdir, dirs, files in os.walk(root_dir):
    #     for file_name in tqdm(files):
    #         full_path = os.path.join(subdir, file_name)
    #         if file_name.endswith(".pt") and 'pareto_front' in full_path:
    #             match = pattern.search(full_path)
    #             if match:
    #                 attr_type, attr_name, loss_type, _, _, _, layer_type, _ = match.groups()
    #                 model_path = full_path
    #                 pareto = True
    #                 log_path = f"metrics_per_model_{attr_type.lower()}/"

    #                 # # print all the arguments
    #                 # print(f"attr_type: {attr_type}")
    #                 # print(f"attr_name: {attr_name}")
    #                 # print(f"loss_type: {loss_type}")
    #                 # print(f"layer_type: {layer_type}")
    #                 # print(f"pareto: {pareto}")
    #                 # print(f"log_path: {log_path}")

    #                 # Call the evaluation function
    #                 evaluation_function(model_backbone=attr_type.lower(),
    #                                     model_path=model_path,
    #                                     localization_loss_fn=loss_type,
    #                                     layer=layer_type,
    #                                     attribution_method=attr_name,
    #                                     pareto=pareto,
    #                                     log_path=log_path)

    # baseline BCOS
    # model_path = "/home/roan/Documents/FACTifAI_2024_3/BASE/VOC2007/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput/model_checkpoint_f1_best.pt"

    # evaluation_function(model_backbone="vanilla",
    #                     model_path=model_path,
    #                     localization_loss_fn=None,
    #                     layer="Input",
    #                     attribution_method=None,
    #                     pareto=False,
    #                     baseline=True,
    #                     log_path="metrics_per_model_vanilla/")
