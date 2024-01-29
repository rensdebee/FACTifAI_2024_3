"""
Reproducibility Study of “Studying How to Efficiently and Effectively Guide Models with Explanations”

Description: This file is part of a project aiming to reproduce the study titled "Studying How to Efficiently and 
Effectively Guide Models with Explanations." The project focuses on verifying the results and methodologies 
proposed in the original study, and potentially extending or refining the study's findings.

Based on the code of orginal paper: https://github.com/sukrutrao/Model-Guidance

utils.py
"""

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
from typing import Tensor
from typing import Dict, List, Tuple, Optional, Any


def remove_module(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove the 'module.' prefix from keys in the state_dict.

    Args:
    - state_dict (Dict[str, Any]): A state dictionary from which to remove the module prefix.

    Returns:
    - Dict[str, Any]: The new state dictionary with 'module.' prefix removed from keys.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:7] == "module." else k
        new_state_dict[name] = v
    return new_state_dict


def set_seed(seed: int) -> None:
    """
    Set the seed for random number generators in PyTorch, random module, and NumPy.

    Args:
    - seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def filter_bbs(bb_coordinates: List[Tuple[int, ...]], gt: int) -> List[Tuple[int, ...]]:
    """
    Filter bounding box coordinates based on a given ground truth label.

    Args:
    - bb_coordinates (List[Tuple[int, ...]]): List of tuples where each tuple represents 
      bounding box coordinates and the first element is the ground truth label.
    - gt (int): The ground truth label to filter the bounding boxes.

    Returns:
    - List[Tuple[int, ...]]: A list of filtered bounding box coordinates.
    """
    bb_list = [bb[1:] for bb in bb_coordinates if bb[0] == gt]
    return bb_list


class BestMetricTracker:
    def __init__(self, metric_name: str):
        """
        Initialize the BestMetricTracker object.

        Args:
        - metric_name (str): The name of the metric to track.
        """
        super().__init__()
        self.metric_name = metric_name
        self.best_model_dict = None
        self.best_epoch = None
        self.best_metrics = None
        self.initialized = False


    def update_values(self, metric_dict: Dict[str, float], model: Any, epoch: int) -> None:
        """
        Update the values of the best model, metrics, and epoch.

        Args:
        - metric_dict (Dict[str, float]): A dictionary of metrics.
        - model (Any): The model object.
        - epoch (int): The current epoch.
        """
        self.best_model_dict = copy.deepcopy(model.state_dict())
        self.best_metrics = copy.deepcopy(metric_dict)
        self.best_epoch = epoch


    def update(self, metric_dict: Dict[str, float], model: Any, epoch: int) -> None:
        """
        Update the tracker with new metrics, model, and epoch.

        Args:
        - metric_dict (Dict[str, float]): A dictionary of metrics.
        - model (Any): The model object.
        - epoch (int): The current epoch.
        """
        if not self.initialized:
            self.update_values(metric_dict, model, epoch)
            self.initialized = True
        elif self.best_metrics[self.metric_name] < metric_dict[self.metric_name]:
            self.update_values(metric_dict, model, epoch)


    def get_best(self) -> Optional[tuple]:
        """
        Get the best metric value, model state dict, epoch, and metrics.

        Returns:
        - Optional[tuple]: A tuple containing the best metric value, model state dict, best epoch, 
          and best metrics, or None if not initialized.
        """
        if not self.initialized:
            return None, None, None, None
        return (
            self.best_metrics[self.metric_name],
            self.best_model_dict,
            self.best_epoch,
            self.best_metrics,
        )


def get_random_optimization_targets(targets: Tensor) -> Tensor:
    """
    Get random optimization targets based on the input probabilities.

    Args:
    - targets (Tensor): A tensor of target probabilities for each class.

    Returns:
    - Tensor: A tensor containing indices of selected classes based on the input probabilities.
    """
    summed = targets.sum(dim=1, keepdim=True).detach()
    probabilities = targets / summed
    return probabilities.multinomial(num_samples=1).squeeze(1)


class ParetoFrontModels:
    def __init__(self: Any, epg: bool = True, iou: bool = True, adapt_iou: bool = False, bin_width: float = 0.005) -> None:
        """
        Initialize the ParetoFrontModels object.

        Args:
        - epg (bool, optional): Whether to use EPG. Defaults to True.
        - iou (bool, optional): Whether to use IOU. Defaults to True.
        - adapt_iou (bool, optional): Whether to use adaptive IOU. Defaults to False.
        - bin_width (float, optional): The bin width for Pareto front. Defaults to 0.005.
        """
        super().__init__()
        self.bin_width = bin_width
        self.pareto_checkpoints = []
        self.pareto_costs = []
        self.epg = epg
        self.iou = iou
        self.adapt_iou = adapt_iou


    def update(self, model: Any, metric_dict: Dict[str, float], epoch: int, sll: Optional[float] = None) -> None:
        """
        Update the Pareto front with new model, metrics, and epoch.

        Args:
        - model (Any): The model object.
        - metric_dict (Dict[str, float]): A dictionary of metrics.
        - epoch (int): The current epoch.
        - sll (Optional[float], optional): The SLL value. Defaults to None.
        """

        metric_vals = copy.deepcopy(metric_dict)
        state_dict = copy.deepcopy(model.state_dict())

        self.sll = sll

        if sll is not None:
            metric_vals.update({"model": state_dict, "epochs": epoch + 1, "sll": sll})
        else:
            metric_vals.update({"model": state_dict, "epochs": epoch + 1})
        self.pareto_checkpoints.append(metric_vals)

        # Which metrics to evaluate in making a pareto front
        if self.epg and self.iou:
            self.pareto_costs.append(
                [metric_vals["F-Score"], metric_vals["BB-Loc"], metric_vals["BB-IoU"]]
            )
        elif self.epg:
            self.pareto_costs.append([metric_vals["F-Score"], metric_vals["BB-Loc"]])
        elif self.iou:
            self.pareto_costs.append([metric_vals["F-Score"], metric_vals["BB-IoU"]])
        elif self.adapt_iou:
            self.pareto_costs.append(
                [metric_vals["F-Score"], metric_vals["BB-IoU-Adapt"]]
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

            # Which evaluated metrics for the pareto front to print
            if self.epg and self.iou:
                pareto_str += f"(F1:{cost[0]:.4f}, EPG:{cost[1]:.4f}, IOU:{cost[2]:.4f}, MOD{self.pareto_checkpoints[idx]['epochs']})"
            elif self.epg:
                pareto_str += f"(F1:{cost[0]:.4f}, EPG:{cost[1]:.4f}, MOD{self.pareto_checkpoints[idx]['epochs']})"
            elif self.iou:
                pareto_str += f"(F1:{cost[0]:.4f}, IOU:{cost[1]:.4f}, MOD{self.pareto_checkpoints[idx]['epochs']})"
            elif self.adapt_iou:
                pareto_str += f"(F1:{cost[0]:.4f}, ADAPTIOU:{cost[1]:.4f}, MOD{self.pareto_checkpoints[idx]['epochs']})"

        print(f"Pareto Costs: {pareto_str}")


    def get_pareto_front(self) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """
        Get the Pareto front.

        Returns:
        - Tuple[List[Dict[str, Any]], List[List[float]]]: A tuple containing the Pareto checkpoints and costs.
        """
        return self.pareto_checkpoints, self.pareto_costs
    

    def save_pareto_front(self: Any, save_path: str, npz: bool = False) -> None:
        """
        Save the Pareto front.

        Args:
        - save_path (str): The path to save the Pareto front.
        - npz (bool, optional): Whether to save as npz. Defaults to False.
        """

        # Create folder to save pareto front
        augmented_path = os.path.join(save_path, "pareto_front")
        os.makedirs(augmented_path, exist_ok=True)

        # Loop through all pareto checkpoints and save them
        for idx, _ in enumerate(self.pareto_checkpoints):
            f_score = self.pareto_checkpoints[idx]["F-Score"]
            bb_score = self.pareto_checkpoints[idx]["BB-Loc"]
            iou_score = self.pareto_checkpoints[idx]["BB-IoU"]
            adapt_iou_score = self.pareto_checkpoints[idx]["BB-IoU-Adapt"]
            epoch = self.pareto_checkpoints[idx]["epochs"]

            # If SLL is used, get SLL value from pareto checkpoint
            if self.sll is not None:
                sll = self.pareto_checkpoints[idx]["sll"]
            else:
                sll = None

            if self.epg and self.iou:
                method = "EPG_IOU"
            elif self.epg:
                method = "EPG"
            elif self.iou:
                method = "IOU"
            elif self.adapt_iou:
                method = "ADAPT"

            torch.save(
                self.pareto_checkpoints[idx],
                os.path.join(
                    augmented_path,
                    f"test_pareto_ch_{method}_SLL{sll}_F{f_score:.4f}_EPG{bb_score:.4f}_IOU{iou_score:.4f}_MOD{epoch}.pt",
                ),
            )

            # Save as npz if specified
            if npz:

                # If adapt_iou is used, save as adapt_iou_score
                if self.adapt_iou:
                    npz_name = f"{method}_SLL{sll}_F{f_score:.4f}_EPG{bb_score:.4f}_IOU{adapt_iou_score:.4f}_{epoch}"
                else:
                    npz_name = f"{method}_SLL{sll}_F{f_score:.4f}_EPG{bb_score:.4f}_IOU{iou_score:.4f}_{epoch}"

                # Save npz
                npz_path = os.path.join(save_path, npz_name)
                np.savez(
                    npz_path,
                    f_score=f_score,
                    bb_score=bb_score,
                    iou_score=iou_score,
                    adapt_iou_score=adapt_iou_score,
                    sll=sll,
                )


    def is_pareto_efficient(self: Any, costs: np.ndarray, return_mask: bool = True) -> np.ndarray:
        """
        Find the pareto-efficient points.

        Args:
        - costs (np.ndarray): An (n_points, n_costs) array.
        - return_mask (bool, optional): True to return a mask. Defaults to True.

        Returns:
        - np.ndarray: An array of indices of pareto-efficient points. If return_mask is True, this will be an (n_points, ) boolean array. Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """

        # Create an array of `n_points` indices containing all the points in the frontier
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]

        # Next index in the is_efficient array to search for
        next_point_index = 0

        # Loop through all the points
        while next_point_index < len(costs):
            
            # Get the nondominated point mask of the next point
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)

            # If next_point_index is nondominated
            nondominated_point_mask[next_point_index] = True

            # Remove dominated points
            is_efficient = is_efficient[nondominated_point_mask]
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

        # Return mask or index
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient


def enlarge_bb(bb_list: List[List[int]], percentage: int = 0) -> List[List[int]]:
    """
    Enlarge bounding boxes by a given percentage.

    Args:
    - bb_list (List[List[int]]): A list of bounding boxes.
    - percentage (int, optional): The percentage to enlarge the bounding boxes. Defaults to 0.

    Returns:
    - List[List[int]]: A list of enlarged bounding boxes.
    """

    en_bb_list = []

    # Loop through all bounding boxes and enlarge them
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


def update_val_metrics(metric_vals: Dict[str, float]) -> Dict[str, float]:
    """
    Update the validation metrics.

    Args:
    - metric_vals (Dict[str, float]): A dictionary of metrics.

    Returns:
    - Dict[str, float]: A dictionary of updated metrics.
    """

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
    Undo the normalization to get back the original image. For visualization purposes.

    Args:
    - mean (List[float]): The mean used for normalizing the images.
    - std (List[float]): The standard deviation used for normalizing the images.
    """

    def __init__(self, mean: List[float], std: List[float], *args, **kwargs) -> None:
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the unnormalization transformation to the given tensor.

        Args:
        - tensor (torch.Tensor): The tensor to be unnormalized.

        Returns:
        - torch.Tensor: The unnormalized tensor.
        """
        return super().__call__(tensor.clone())


def get_class_name(class_num: int) -> str:
    """
    Function to map from class number back to classname (VOC2007).

    Args:
    - class_num (int): The class number to be mapped to a classname.

    Returns:
    - str: The classname corresponding to the given class number. Returns 
           "class doesn't exist" if the class number is not found.
    """

    target_dict = {
        "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4,
        "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10,
        "dog": 11, "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15,
        "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19,
    }

    for key, value in target_dict.items():
        if class_num == value:
            return key

    return "class doesn't exist"


def get_class_number(class_name: str) -> int:
    """
    Function to map from classname to class number.

    Args:
    - class_name (str): The classname to be mapped to a class number.

    Returns:
    - int: The class number corresponding to the given classname.
    """

    target_dict = {
        "aeroplane": 0, "bicycle": 1, "bird": 2, "boat": 3, "bottle": 4,
        "bus": 5, "car": 6, "cat": 7, "chair": 8, "cow": 9, "diningtable": 10,
        "dog": 11, "horse": 12, "motorbike": 13, "person": 14, "pottedplant": 15,
        "sheep": 16, "sofa": 17, "train": 18, "tvmonitor": 19,
    }
    
    return target_dict[class_name]


def get_model_specs(path: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Function to get model specs from model path name.

    Args:
    - path (str): The path of the model.

    Returns:
    - Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]: A tuple containing 
      the model backbone, localization loss function, layer, and attribution method 
      extracted from the path. Each element is None if the corresponding spec is not found.
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
    model_backbone: str,
    localization_loss_fn: str,
    layer: str,
    attribution_method: str,
    dataset: str,
    model_path: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """
    Load model, attributor, and transform from model specs.

    Args:
    - model_backbone (str): The backbone of the model.
    - localization_loss_fn (str): The localization loss function used in the model.
    - layer (str): The layer of the model to consider for processing.
    - attribution_method (str): The method used for attribution in the model.
    - dataset (str): The dataset name.
    - model_path (Optional[str]): The path to the model's weights. Defaults to None.

    Returns:
    - Tuple[Any, Any, Any]: A tuple containing the model activator, attributor, and transformer.
    """

    # Get number of classes
    num_classes_dict = {"VOC2007": 20, "COCO2014": 80, "WATERBIRDS": 2}
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

    # If needed get atribution method to calculate atribution maps
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


def get_color_map() -> mcolors.LinearSegmentedColormap:
    """
    Returns used color map.

    Returns:
    - mcolors.LinearSegmentedColormap: The colormap object.
    """

    cdict = {
        "red": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        "green": [[0.0, 0.0, 1.0], [0.3, 0.15, 0.15], [1.0, 0.0, 0.0]],
        "blue": [[0.0, 0.0, 1.0], [0.3, 0.15, 0.15], [1.0, 0.0, 0.0]],
    }

    return mcolors.LinearSegmentedColormap("DarkRed", cdict)


def get_waterbird_name(class_num: int) -> str:
    """
    Function to translate class number into label for waterbirds.

    Args:
    - class_num (int): The class number.

    Returns:
    - str: The class name ('Waterbird' or 'Landbird').
    """

    class_name = "Waterbird" if class_num == 1 else "Landbird"

    return class_name


def switch_best_to_last(path_name: str, epochs: int = 350) -> str:
    """
    Function to switch model path from best epoch to last epoch save.

    Args:
    - path_name (str): The original path name of the model.
    - epochs (int, optional): The number of epochs. Defaults to 350.

    Returns:
    - str: The modified path name.
    """

    return f"final_{epochs}".join(path_name.rsplit("f1_best", 1))


def load_data_from_folders_with_npz_files(
    root_folder: str, 
    metrics: Tuple[str, str] = ('f_score', 'bb_score'),
) -> Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]:
    """
    Loads and organizes data from specified folders for plotting Pareto curves, including baseline data as a separate data type.

    Args:
        root_folder (str): Root directory containing the data folders.
        metrics (Tuple[str, str], optional): A tuple specifying the metrics to be used. Defaults to ('f_score', 'bb_score').

    Returns:
        Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]: A dictionary containing organized data for plotting, including baseline data as a separate type.
    """

    valid_metrics = ['f_score', 'bb_score', 'iou_score', 'adapt_iou_score']
    data_types = ['energy', 'l1', 'ppce', 'rrr', 'baseline']
    data_structure = {data_type: [] for data_type in data_types}

    data_dict = {'vanilla': {'input': dict(data_structure), 'final': dict(data_structure), 'mid2': dict(data_structure)},
                 'bcos': {'input': dict(data_structure), 'final': dict(data_structure), 'mid2': dict(data_structure)}}
    
    if not all(metric in valid_metrics for metric in metrics):
        raise ValueError(f'Invalid metrics: {metrics}. Valid options are {valid_metrics}')
    
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.npz'):

                category = 'vanilla' if 'vanilla' in subdir.lower() else 'bcos' if 'bcos' in subdir.lower() else None
                if category is None:
                    continue

                filepath = os.path.join(subdir, file)
                data = np.load(filepath)

                if 'adapt_iou_score' in metrics and 'mid2' in subdir.lower():
                    continue
                else:
                    data = {metric: data[metric] * 100 for metric in metrics}

                # Replace adapt_iou_score with iou_score for mid2 data
                metric = "iou_score" if 'mid2' in subdir.lower() and 'adapt_iou_score' in metrics else metrics[1]

                if 'baseline' in subdir.lower() and 'input' in file.lower():

                    data_dict[category]['input']['baseline'] = (data[metrics[0]], data[metric])

                if 'baseline' in subdir.lower() and 'final' in file.lower():

                    data_dict[category]['final']['baseline'] = (data[metrics[0]], data[metric])

                if 'bb_score' in metrics and file.split('_')[0] == 'EPG':

                    if 'input' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['input']['energy'].append((data[metrics[0]], data[metric]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['input']['l1'].append((data[metrics[0]], data[metric]))
                        elif 'ppce' in subdir.lower():
                            data_dict[category]['input']['ppce'].append((data[metrics[0]], data[metric]))
                        elif 'rrr' in subdir.lower():
                            data_dict[category]['input']['rrr'].append((data[metrics[0]], data[metric]))

                    if 'final' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['final']['energy'].append((data[metrics[0]], data[metric]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['final']['l1'].append((data[metrics[0]], data[metric]))
                        elif 'ppce' in subdir.lower():
                            data_dict[category]['final']['ppce'].append((data[metrics[0]], data[metric]))
                        elif 'rrr' in subdir.lower():
                            data_dict[category]['final']['rrr'].append((data[metrics[0]], data[metric]))

                    if 'mid2' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['mid2']['energy'].append((data[metrics[0]], data[metric]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['mid2']['l1'].append((data[metrics[0]], data[metric]))
                        elif 'ppce' in subdir.lower():
                            data_dict[category]['mid2']['ppce'].append((data[metrics[0]], data[metric]))
                        elif 'rrr' in subdir.lower():
                            data_dict[category]['mid2']['rrr'].append((data[metrics[0]], data[metric]))

                if 'iou_score' in metrics and file.split('_')[0] == 'IOU':
                        
                    if 'input' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['input']['energy'].append((data[metrics[0]], data[metric]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['input']['l1'].append((data[metrics[0]], data[metric]))
                        elif 'ppce' in subdir.lower():
                            data_dict[category]['input']['ppce'].append((data[metrics[0]], data[metric]))
                        elif 'rrr' in subdir.lower():
                            data_dict[category]['input']['rrr'].append((data[metrics[0]], data[metric]))

                    if 'final' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['final']['energy'].append((data[metrics[0]], data[metric]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['final']['l1'].append((data[metrics[0]], data[metric]))
                        elif 'ppce' in subdir.lower():
                            data_dict[category]['final']['ppce'].append((data[metrics[0]], data[metric]))
                        elif 'rrr' in subdir.lower():
                            data_dict[category]['final']['rrr'].append((data[metrics[0]], data[metric]))

                    if 'mid2' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['mid2']['energy'].append((data[metrics[0]], data[metric]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['mid2']['l1'].append((data[metrics[0]], data[metric]))
                        elif 'ppce' in subdir.lower():
                            data_dict[category]['mid2']['ppce'].append((data[metrics[0]], data[metric]))
                        elif 'rrr' in subdir.lower():
                            data_dict[category]['mid2']['rrr'].append((data[metrics[0]], data[metric]))

                if 'adapt_iou_score' in metrics and file.split('_')[0] == 'ADAPT':
                
                    if 'input' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['input']['energy'].append((data[metrics[0]], data[metric]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['input']['l1'].append((data[metrics[0]], data[metric]))
                        elif 'ppce' in subdir.lower():
                            data_dict[category]['input']['ppce'].append((data[metrics[0]], data[metric]))
                        elif 'rrr' in subdir.lower():
                            data_dict[category]['input']['rrr'].append((data[metrics[0]], data[metric]))

                    if 'final' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['final']['energy'].append((data[metrics[0]], data[metric]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['final']['l1'].append((data[metrics[0]], data[metric]))
                        elif 'ppce' in subdir.lower():
                            data_dict[category]['final']['ppce'].append((data[metrics[0]], data[metric]))
                        elif 'rrr' in subdir.lower():
                            data_dict[category]['final']['rrr'].append((data[metrics[0]], data[metric]))

                    if 'mid2' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['mid2']['energy'].append((data[metrics[0]], data[metric]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['mid2']['l1'].append((data[metrics[0]], data[metric]))
                        elif 'ppce' in subdir.lower():
                            data_dict[category]['mid2']['ppce'].append((data[metrics[0]], data[metric]))
                        elif 'rrr' in subdir.lower():
                            data_dict[category]['mid2']['rrr'].append((data[metrics[0]], data[metric]))

        # Sorting data for each category
        for category in ['vanilla', 'bcos']:
            for data_type in data_types:
                if data_type != 'baseline':
                    data_dict[category]['input'][data_type] = sorted(data_dict[category]['input'][data_type])
                    data_dict[category]['final'][data_type] = sorted(data_dict[category]['final'][data_type])
                    data_dict[category]['mid2'][data_type] = sorted(data_dict[category]['mid2'][data_type])

    return data_dict

def load_data_from_folders_with_npz_files_with_limited_ann(
    root_folder: str, 
    metrics: Tuple[str, str] = ('f_score', 'bb_score')
) -> Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]:
    """
    Loads and organizes data from specified folders with lim0.01, lim0.1, and lim1.0 instead of input, final, and mid2.

    Args:
        root_folder (str): Root directory containing the data folders.
        metrics (Tuple[str, str], optional): A tuple specifying the metrics to be used. Defaults to ('f_score', 'bb_score').

    Returns:
        Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]: A dictionary containing organized data.
    """
    valid_metrics = ['f_score', 'bb_score', 'iou_score', 'adapt_iou_score']
    data_types = ['energy', 'l1', 'baseline']
    data_structure = {data_type: [] for data_type in data_types}

    data_dict = {'vanilla': {'lim0.01': dict(data_structure), 'lim0.1': dict(data_structure), 'lim1.0': dict(data_structure)}, 
                 'bcos': {'lim0.01': dict(data_structure), 'lim0.1': dict(data_structure), 'lim1.0': dict(data_structure)}}
    
    if not all(metric in valid_metrics for metric in metrics):
        raise ValueError(f'Invalid metrics: {metrics}. Valid options are {valid_metrics}')
    
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.npz'):

                category = 'vanilla' if 'vanilla' in subdir.lower() else 'bcos' if 'bcos' in subdir.lower() else None
                if category is None:
                    continue

                filepath = os.path.join(subdir, file)
                data = np.load(filepath)

                data = {metric: data[metric] * 100 for metric in metrics}

                if 'baseline' in subdir.lower() and 'input' in file.lower():

                    data_dict[category]['lim0.01']['baseline'] = (data[metrics[0]], data[metrics[1]])

                if 'baseline' in subdir.lower() and 'input' in file.lower():

                    data_dict[category]['lim0.1']['baseline'] = (data[metrics[0]], data[metrics[1]])

                if 'baseline' in subdir.lower() and 'input' in file.lower():
                        
                    data_dict[category]['lim1.0']['baseline'] = (data[metrics[0]], data[metrics[1]])

                if 'bb_score' in metrics and file.split('_')[0] == 'EPG':

                    if 'lim0.01' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['lim0.01']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['lim0.01']['l1'].append((data[metrics[0]], data[metrics[1]]))

                    if 'lim0.1' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['lim0.1']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['lim0.1']['l1'].append((data[metrics[0]], data[metrics[1]]))

                    if 'lim1.0' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['lim1.0']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['lim1.0']['l1'].append((data[metrics[0]], data[metrics[1]]))


        # Sorting data for each category
        for category in ['vanilla', 'bcos']:
            for data_type in data_types:
                if data_type != 'baseline':
                    data_dict[category]['lim0.01'][data_type] = sorted(data_dict[category]['lim0.01'][data_type])
                    data_dict[category]['lim0.1'][data_type] = sorted(data_dict[category]['lim0.1'][data_type])
                    data_dict[category]['lim1.0'][data_type] = sorted(data_dict[category]['lim1.0'][data_type])
                    
    return data_dict

def load_data_from_folders_with_npz_files_with_dilation(
    root_folder: str, 
    metrics: Tuple[str, str] = ('f_score', 'bb_score')
) -> Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]:
    """
    Loads and organizes data from specified folders with lim0.01, lim0.1, and lim1.0 instead of input, final, and mid2.

    Args:
        root_folder (str): Root directory containing the data folders.
        metrics (Tuple[str, str], optional): A tuple specifying the metrics to be used. Defaults to ('f_score', 'bb_score').

    Returns:
        Dict[str, Dict[str, Dict[str, List[Tuple[float, float]]]]]: A dictionary containing organized data.
    """
    valid_metrics = ['f_score', 'bb_score', 'iou_score', 'adapt_iou_score']
    data_types = ['energy', 'l1', 'baseline']
    data_structure = {data_type: [] for data_type in data_types}

    data_dict = {'vanilla': {'dil0': dict(data_structure), 'dil0.1': dict(data_structure), 'dil0.25': dict(data_structure), 'dil0.5': dict(data_structure),
                             'dil0_not_pareto': dict(data_structure), 'dil0.1_not_pareto': dict(data_structure), 'dil0.25_not_pareto': dict(data_structure), 'dil0.5_not_pareto': dict(data_structure)},
                    'bcos': {'dil0': dict(data_structure), 'dil0.1': dict(data_structure), 'dil0.25': dict(data_structure), 'dil0.5': dict(data_structure),
                             'dil0_not_pareto': dict(data_structure), 'dil0.1_not_pareto': dict(data_structure), 'dil0.25_not_pareto': dict(data_structure), 'dil0.5_not_pareto': dict(data_structure)}}
    
    if not all(metric in valid_metrics for metric in metrics):
        raise ValueError(f'Invalid metrics: {metrics}. Valid options are {valid_metrics}')
    
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.npz'):


                category = 'vanilla' if 'vanilla' in subdir.lower() else 'bcos' if 'bcos' in subdir.lower() else None
                if category is None:
                    continue

                filepath = os.path.join(subdir, file)
                data = np.load(filepath)

                data = {metric: data[metric] * 100 for metric in metrics}

                if 'baseline' in subdir.lower() and 'input' in file.lower():

                    data_dict[category]['dil0']['baseline'] = (data[metrics[0]], data[metrics[1]])

                if 'baseline' in subdir.lower() and 'input' in file.lower():

                    data_dict[category]['dil0.1']['baseline'] = (data[metrics[0]], data[metrics[1]])

                if 'baseline' in subdir.lower() and 'input' in file.lower():

                    data_dict[category]['dil0.25']['baseline'] = (data[metrics[0]], data[metrics[1]])

                if 'baseline' in subdir.lower() and 'input' in file.lower():

                    data_dict[category]['dil0.5']['baseline'] = (data[metrics[0]], data[metrics[1]])

                if ('bb_score' in metrics and file.split('_')[0] == 'EPG') or ('adapt_iou_score' in metrics and file.split('_')[0] == 'ADAPT'):\

                    if 'dil0/yes_par' in subdir.lower():
                    
                        if 'energy' in subdir.lower():
                            data_dict[category]['dil0']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['dil0']['l1'].append((data[metrics[0]], data[metrics[1]]))

                    if 'dil0.1/yes_par' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['dil0.1']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['dil0.1']['l1'].append((data[metrics[0]], data[metrics[1]]))

                    if 'dil0.25/yes_par' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['dil0.25']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['dil0.25']['l1'].append((data[metrics[0]], data[metrics[1]]))

                    if 'dil0.5/yes_par' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['dil0.5']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['dil0.5']['l1'].append((data[metrics[0]], data[metrics[1]]))

                    if 'dil0/not_par' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['dil0_not_pareto']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['dil0_not_pareto']['l1'].append((data[metrics[0]], data[metrics[1]]))

                    if 'dil0.1/not_par' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['dil0.1_not_pareto']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['dil0.1_not_pareto']['l1'].append((data[metrics[0]], data[metrics[1]]))

                    if 'dil0.25/not_par' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['dil0.25_not_pareto']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['dil0.25_not_pareto']['l1'].append((data[metrics[0]], data[metrics[1]]))

                    if 'dil0.5/not_par' in subdir.lower():
                        if 'energy' in subdir.lower():
                            data_dict[category]['dil0.5_not_pareto']['energy'].append((data[metrics[0]], data[metrics[1]]))
                        elif 'l1' in subdir.lower():
                            data_dict[category]['dil0.5_not_pareto']['l1'].append((data[metrics[0]], data[metrics[1]]))


        # Sorting data for each category
        for category in ['vanilla', 'bcos']:
            for data_type in data_types:
                if data_type != 'baseline':
                    data_dict[category]['dil0'][data_type] = sorted(data_dict[category]['dil0'][data_type])
                    data_dict[category]['dil0.1'][data_type] = sorted(data_dict[category]['dil0.1'][data_type])
                    data_dict[category]['dil0.25'][data_type] = sorted(data_dict[category]['dil0.25'][data_type])
                    data_dict[category]['dil0.5'][data_type] = sorted(data_dict[category]['dil0.5'][data_type])

                    data_dict[category]['dil0_not_pareto'][data_type] = sorted(data_dict[category]['dil0_not_pareto'][data_type])
                    data_dict[category]['dil0.1_not_pareto'][data_type] = sorted(data_dict[category]['dil0.1_not_pareto'][data_type])
                    data_dict[category]['dil0.25_not_pareto'][data_type] = sorted(data_dict[category]['dil0.25_not_pareto'][data_type])
                    data_dict[category]['dil0.5_not_pareto'][data_type] = sorted(data_dict[category]['dil0.5_not_pareto'][data_type])
                    
    return data_dict