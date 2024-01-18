import torch
import random
import numpy as np
import copy
from collections import OrderedDict
import os
import torchvision
import model_activators
import losses
import datasets
import fixup_resnet
import hubconf
import attribution_methods
import bcos.modules.bcosconv2d
import bcos.data.transforms
import matplotlib.colors as mcolors


def remove_module(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def filter_bbs(bb_coordinates, gt):
    bb_list = []
    for bb in bb_coordinates:
        if bb[0] == gt:
            bb_list.append(bb[1:])
    return bb_list


class BestMetricTracker:
    def __init__(self, metric_name):
        super().__init__()
        self.metric_name = metric_name
        self.best_model_dict = None
        self.best_epoch = None
        self.best_metrics = None
        self.initialized = False

    def update_values(self, metric_dict, model, epoch):
        self.best_model_dict = copy.deepcopy(model.state_dict())
        self.best_metrics = copy.deepcopy(metric_dict)
        self.best_epoch = epoch

    def update(self, metric_dict, model, epoch):
        if not self.initialized:
            self.update_values(metric_dict, model, epoch)
            self.initialized = True
        elif self.best_metrics[self.metric_name] < metric_dict[self.metric_name]:
            self.update_values(metric_dict, model, epoch)

    def get_best(self):
        if not self.initialized:
            return None, None, None, None
        return (
            self.best_metrics[self.metric_name],
            self.best_model_dict,
            self.best_epoch,
            self.best_metrics,
        )


def get_random_optimization_targets(targets):
    probabilities = targets / targets.sum(dim=1, keepdim=True).detach()
    return probabilities.multinomial(num_samples=1).squeeze(1)


class ParetoFrontModels:
    def __init__(self, bin_width=0.005):
        super().__init__()
        self.bin_width = bin_width
        self.pareto_checkpoints = []
        self.pareto_costs = []

    def update(self, model, metric_dict, epoch):
        metric_vals = copy.deepcopy(metric_dict)
        state_dict = copy.deepcopy(model.state_dict())
        metric_vals.update({"model": state_dict, "epochs": epoch + 1})
        self.pareto_checkpoints.append(metric_vals)
        self.pareto_costs.append(
            [metric_vals["F-Score"], metric_vals["BB-Loc"], metric_vals["BB-IoU"]]
        )
        efficient_indices = self.is_pareto_efficient(
            -np.round(np.array(self.pareto_costs) / self.bin_width, 0) * self.bin_width,
            return_mask=False,
        )
        self.pareto_checkpoints = [
            self.pareto_checkpoints[idx] for idx in efficient_indices
        ]
        self.pareto_costs = [self.pareto_costs[idx] for idx in efficient_indices]
        print(f"Current Pareto Front Size: {len(self.pareto_checkpoints)}")
        pareto_str = ""
        for idx, cost in enumerate(self.pareto_costs):
            pareto_str += f"({cost[0]:.4f},{cost[1]:.4f},{cost[2]:.4f},{self.pareto_checkpoints[idx]['epochs']})"
        print(f"Pareto Costs: {pareto_str}")

    def get_pareto_front(self):
        return self.pareto_checkpoints, self.pareto_costs

    def save_pareto_front(self, save_path):
        augmented_path = os.path.join(save_path, "pareto_front")
        os.makedirs(augmented_path, exist_ok=True)
        for idx in range(len(self.pareto_checkpoints)):
            f_score = self.pareto_checkpoints[idx]["F-Score"]
            bb_score = self.pareto_checkpoints[idx]["BB-Loc"]
            iou_score = self.pareto_checkpoints[idx]["BB-IoU"]
            epoch = self.pareto_checkpoints[idx]["epochs"]
            torch.save(
                self.pareto_checkpoints[idx],
                os.path.join(
                    augmented_path,
                    f"model_checkpoint_pareto_{f_score:.4f}_{bb_score:.4f}_{iou_score:.4f}_{epoch}.pt",
                ),
            )

    def is_pareto_efficient(self, costs, return_mask=True):
        """
        Find the pareto-efficient points
        : param costs: An(n_points, n_costs) array
        : param return_mask: True to return a mask
        : return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an(n_points, ) boolean array
            Otherwise it will be a(n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            # Remove dominated points
            is_efficient = is_efficient[nondominated_point_mask]
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient


def enlarge_bb(bb_list, percentage=0):
    en_bb_list = []
    for bb_coord in bb_list:
        xmin, ymin, xmax, ymax = bb_coord
        width = xmax - xmin
        height = ymax - ymin
        w_margin = int(percentage * width)
        h_margin = int(percentage * height)
        new_xmin = max(0, xmin - w_margin)
        new_xmax = min(223, xmax + w_margin)
        new_ymin = max(0, ymin - h_margin)
        new_ymax = min(223, ymax + h_margin)
        en_bb_list.append([new_xmin, new_ymin, new_xmax, new_ymax])
    return en_bb_list


def update_val_metrics(metric_vals):
    metric_vals["Val-Accuracy"] = metric_vals.pop("Accuracy")
    metric_vals["Val-Precision"] = metric_vals.pop("Precision")
    metric_vals["Val-Recall"] = metric_vals.pop("Recall")
    metric_vals["Val-F-Score"] = metric_vals.pop("F-Score")
    metric_vals["Val-Average-Loss"] = metric_vals.pop("Average-Loss")
    if "BB-Loc" in metric_vals:
        metric_vals["Val-BB-Loc"] = metric_vals.pop("BB-Loc")
        metric_vals["Val-BB-IoU"] = metric_vals.pop("BB-IoU")
    return metric_vals


class UnNormalize(torchvision.transforms.Normalize):
    """
    Undo the normalization to get back the original image. For visualization
    """

    def __init__(self, mean, std, *args, **kwargs):
        """
        mean = mean which the images where transformed
        std = std with which the images where transformed
        """
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def get_class_name(class_num):
    """
    Function to map from class number back to classname
    """
    target_dict = {
        "aeroplane": 0,
        "bicycle": 1,
        "bird": 2,
        "boat": 3,
        "bottle": 4,
        "bus": 5,
        "car": 6,
        "cat": 7,
        "chair": 8,
        "cow": 9,
        "diningtable": 10,
        "dog": 11,
        "horse": 12,
        "motorbike": 13,
        "person": 14,
        "pottedplant": 15,
        "sheep": 16,
        "sofa": 17,
        "train": 18,
        "tvmonitor": 19,
    }
    for key, value in target_dict.items():
        if class_num == value:
            return key

    return "class doesn't exist"


def get_model_specs(path):
    """
    Function to get model specs from model path name
    """
    # Model options
    model_backbones = ["bcos", "xdnn", "vanilla"]
    localization_loss_fns = ["Energy", "L1", "RRR", "PPCE"]
    layers = ["Input", "Final", "Mid1", "Mid2", "Mid3"]
    attribution_methods = ["BCos", "GradCam", "IxG"]

    # Check if option in model name
    # If in name break loop and thus option is stored in variable
    for model_backbone in model_backbones:
        if model_backbone in path:
            break

    # If not found set to None
    for localization_loss_fn in localization_loss_fns:
        if localization_loss_fn in path:
            break
        else:
            localization_loss_fn = None

    for layer in layers:
        if layer in path:
            break
        else:
            layer = None

    for attribution_method in attribution_methods:
        if attribution_method in path:
            break
        else:
            attribution_method = None
    return model_backbone, localization_loss_fn, layer, attribution_method


def get_model(
    model_backbone,
    localization_loss_fn,
    layer,
    attribution_method,
    dataset,
    model_path=None,
):
    """
    Load model, attributor and transform from model specs
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
        imagenet_state_dict = remove_module(imagenet_checkpoint["state_dict"])
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
        print("No model path provided using resnet baseline model")

    model = model.cuda()

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
        attributor = attribution_methods.get_attributor(
            model,
            attribution_method,
            loss_loc.only_positive,
            loss_loc.binarize,
            interpolate,
            (224, 224),
            batch_mode=False,
        )
    else:
        attributor = None

    # Get correct transformation
    if is_bcos:
        transformer = bcos.data.transforms.AddInverse(dim=0)
    else:
        transformer = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    return model_activator, attributor, transformer


def get_color_map():
    """
    Returns used color map
    """
    cdict = {
        "red": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        "green": [[0.0, 0.0, 1.0], [0.3, 0.15, 0.15], [1.0, 0.0, 0.0]],
        "blue": [[0.0, 0.0, 1.0], [0.3, 0.15, 0.15], [1.0, 0.0, 0.0]],
    }
    return mcolors.LinearSegmentedColormap("DarkRed", cdict)
