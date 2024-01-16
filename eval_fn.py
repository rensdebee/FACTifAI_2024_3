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


def eval_model(
    model,
    attributor,
    loader,
    num_batches,
    num_classes,
    loss_fn,
    writer=None,
    epoch=None,
):
    model.eval()
    f1_metric = metrics.MultiLabelMetrics(num_classes=num_classes, threshold=0.0)
    bb_metric = metrics.BoundingBoxEnergyMultiple()
    vis_flag = False
    iou_metric = metrics.BoundingBoxIoUMultiple(vis_flag=vis_flag)
    total_loss = 0
    for batch_idx, (test_X, test_y, test_bbs) in enumerate(tqdm(loader)):
        test_X.requires_grad = True
        test_X = test_X.cuda()
        test_y = test_y.cuda()
        logits, features = model(test_X)
        loss = loss_fn(logits, test_y).detach()
        total_loss += loss
        f1_metric.update(logits, test_y)

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
                    bb_metric.update(attributions, bb_list)
                    iou_metric.update(attributions, bb_list, image, pred)

    metric_vals = f1_metric.compute()
    if attributor:
        bb_metric_vals = bb_metric.compute()
        iou_metric_vals = iou_metric.compute()
        metric_vals["BB-Loc"] = bb_metric_vals
        metric_vals["BB-IoU"] = iou_metric_vals
    metric_vals["Average-Loss"] = total_loss.item() / num_batches
    print(f"Validation Metrics: {metric_vals}")
    model.train()
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
    model_backbone,
    model_path,
    localization_loss_fn,
    layer,
    attribution_method,
    eval_batch_size=4,
    data_path="datasets/",
    dataset="VOC2007",
    split="test",
    annotated_fraction=1,
):
    """
    Function which returns the metrics of a given model in model_path
    on certain data split.

    Note this function does not save metrics in tensorboard log, use the
    commandine script for that functionality.
    """
    # Get number of classes
    num_classes_dict = {"VOC2007": 20, "COCO2014": 80}
    num_classes = num_classes_dict[dataset]

    # Load correct model
    is_bcos = model_backbone == "bcos"
    is_xdnn = model_backbone == "xdnn"
    is_vanilla = model_backbone == "vanilla"

    if is_bcos:
        model = hubconf.resnet50(pretrained=True)
        model[0].fc = bcos.modules.bcosconv2d.BcosConv2d(
            in_channels=model[0].fc.in_channels, out_channels=num_classes
        )
        layer_dict = {"Input": None, "Mid1": 3, "Mid2": 4, "Mid3": 5, "Final": 6}
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
    elif is_vanilla:
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        )
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features, out_features=num_classes
        )
        layer_dict = {"Input": None, "Mid1": 4, "Mid2": 5, "Mid3": 6, "Final": 7}
    else:
        raise NotImplementedError

    # Get layer to extract atribution layers
    layer_idx = layer_dict[layer]

    # Load model checkpoint
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
    else:
        raise Exception("Model path must be provided for evaluations")

    model = model.cuda()

    writer = None

    # Add transform for BCOS model ekse normalize
    if is_bcos:
        transformer = bcos.data.transforms.AddInverse(dim=0)
    else:
        transformer = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # Load dataset base on --split argument
    root = os.path.join(data_path, dataset, "processed")

    if split == "train":
        train_data = datasets.VOCDetectParsed(
            root=root,
            image_set="train",
            transform=transformer,
            annotated_fraction=annotated_fraction,
        )
        loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=eval_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=datasets.VOCDetectParsed.collate_fn,
        )
        num_batches = len(train_data) / eval_batch_size
    elif split == "val":
        val_data = datasets.VOCDetectParsed(
            root=root, image_set="val", transform=transformer
        )
        loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=datasets.VOCDetectParsed.collate_fn,
        )
        num_batches = len(val_data) / eval_batch_size
    elif split == "test":
        test_data = datasets.VOCDetectParsed(
            root=root, image_set="test", transform=transformer
        )
        loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=datasets.VOCDetectParsed.collate_fn,
        )
        num_batches = len(test_data) / eval_batch_size
    else:
        raise Exception("Data split not valid choose from ['train', 'val', 'test']")

    # Get loss function to calculate loss of split
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_loc = (
        losses.get_localization_loss(localization_loss_fn)
        if localization_loss_fn
        else None
    )

    # Get model activator to procces batches
    model_activator = model_activators.ResNetModelActivator(
        model=model, layer=layer_idx, is_bcos=is_bcos
    )

    # If neede get atribution method to calculate atribution maps
    if attribution_method:
        interpolate = True if layer_idx is not None else False
        eval_attributor = attribution_methods.get_attributor(
            model,
            attribution_method,
            loss_loc.only_positive,
            loss_loc.binarize,
            interpolate,
            (224, 224),
            batch_mode=False,
        )
    else:
        eval_attributor = None

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
    )
    return metric_vals


if __name__ == "__main__":
    evaluation_function(
        "bcos",
        "FT\VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerInput\model_checkpoint_f1_best.pt",
        "Energy",
        "Input",
        "BCos",
    )
