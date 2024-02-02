"""
Reproducibility Study of “Studying How to Efficiently and Effectively Guide Models with Explanations”

Description: This file is part of a project aiming to reproduce the study titled "Studying How to Efficiently and 
Effectively Guide Models with Explanations." The project focuses on verifying the results and methodologies 
proposed in the original study, and potentially extending or refining the study's findings.

Based on the code of orginal paper: https://github.com/sukrutrao/Model-Guidance

visualize.py
! Note we commented out the font family since it does not come standard on linux !
"""

import torch
import argparse
import datasets
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import numpy as np
from typing import Dict, List, Tuple, Optional


def visualize_fig9(
    model_paths,
    fix_layer=None,
    data_path="datasets/",
    dataset="VOC2007",
    image_set="test",
    last=False,
):
    num_images = 15
    # Get dataset path
    root = os.path.join(data_path, dataset, "processed")

    # Create figure
    fig, axs = plt.subplots(
        num_images,
        1 + len(model_paths),
        figsize=(5 * (1 + len(model_paths)), 5 * num_images),
    )

    # Create custom color map
    dark_red_cmap = utils.get_color_map()

    # Load images
    data = datasets.VOCDetectParsed(root=root, image_set=image_set)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=num_images,
        shuffle=True,
        num_workers=0,
        collate_fn=datasets.VOCDetectParsed.collate_fn,
    )

    # Get one random batch
    inputs, classes, bb_box_list = next(iter(loader))
    # Class picked per image (images contain multiple classes)
    chosen_classes = []

    for i in range(num_images):
        # Get image
        image = inputs[i]
        # Get all bboxes for this image
        bbs = bb_box_list[i]

        # Pick one random class from all classes in image
        clas = np.random.choice(torch.where(classes[i] == 1)[0])
        chosen_classes.append(clas)

        # Get class name from class number
        class_name = utils.get_class_name(clas)

        # Get bboxes from specific classes
        class_bbs = utils.filter_bbs(bbs, clas)

        # Show original image
        axs[i][0].imshow(torch.movedim(image[:3, :, :], 0, -1))
        axs[i][0].set_title(
            "Input",
            fontsize=45,
            pad=20,
            # fontname="Times New Roman",
            fontweight=650,
        )
        axs[i][0].set_ylabel(
            class_name,
            fontsize=45,
            # fontname="Times New Roman",
            fontweight=650,
        )

        # Plot boundingboxes
        for coords in class_bbs:
            xmin, ymin, xmax, ymax = coords
            axs[i][0].add_patch(
                patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fc="none",
                    ec="#00FFFF",
                    lw=5,
                )
            )
        for row_idx, path in enumerate(model_paths):
            # Get model spec from path
            (
                model_backbone,
                localization_loss_fn,
                layer,
                attribution_method,
            ) = utils.get_model_specs(path)
            # If no attribution method set default values
            if not attribution_method:
                if model_backbone == "bcos":
                    attribution_method = "BCos"
                elif model_backbone == "vanilla":
                    attribution_method = "IxG"

            og_loss_fn = localization_loss_fn
            # default localistion loss is energy
            if not localization_loss_fn:
                localization_loss_fn = "Energy"

            if last:
                # If fixed layers are given set layer
                if fix_layer:
                    layer = fix_layer
                epochs = 50
                if og_loss_fn is None:
                    epochs = 300
                path = utils.switch_best_to_last(path, epochs)
            # Get model, attributor, en transform based on specs
            model, attributor, transformer = utils.get_model(
                model_backbone,
                localization_loss_fn,
                layer,
                attribution_method,
                dataset,
                path,
            )
            model.eval()

            # apply transform
            transformer.dim = -3
            X = transformer(inputs.clone())
            X = X[i : i + 1]
            # Get output from model
            X.requires_grad = True
            X = X.cuda()
            logits, features = model(X)

            # Get attributions per image
            for img_idx, image in enumerate(X):
                pred = chosen_classes[i]
                attributions = (
                    attributor(features, logits, pred, img_idx)
                    .detach()
                    .squeeze(0)
                    .squeeze(0)
                )
                positive_attributions = attributions.clamp(min=0).cpu()
                bb = class_bbs

                # Plot attribution map
                axs[i][row_idx + 1].imshow(positive_attributions, cmap=dark_red_cmap)
                # Plot boundingbox
                for coords in bb:
                    xmin, ymin, xmax, ymax = coords
                    axs[i][row_idx + 1].add_patch(
                        patches.Rectangle(
                            (xmin, ymin),
                            xmax - xmin,
                            ymax - ymin,
                            fc="none",
                            ec="#4169E1",
                            lw=5,
                        )
                    )
                # Set row title
                if og_loss_fn is None:
                    localization_loss_fn = "Baseline"
                axs[i][row_idx + 1].set_title(
                    localization_loss_fn,
                    pad=20,
                    fontsize=45,
                    # fontname="Times New Roman",
                    fontweight=650,
                )

    # Disable ticks
    for _ax in axs:
        for ax in _ax:
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
    # Save figure
    last = "Last" if last else "Best"
    fig.tight_layout()
    plt.savefig(f"./images/Figure9_{last}.png")


def visualize_fig2(
    models_names,
    models_modes,
    models_paths,
    fix_layers=None,
    data_path="datasets/",
    dataset="VOC2007",
    image_set="test",
    last=False,
):
    """
    Function to recreate figure 2 of the paper
    model_names: list of model names to visualize [BCos, IxG]
    models_modes: list of modes [baseline vs guided]
    model_paths: list in list of path to models
    fix_layers: optional list to fix layer [Input, Final]
    data_path: optional path to dataset
    num_images: optional number of images to plot
    dataset: optional dataset to use
    image_set: optional dataset split to use
    """
    # Get dataset path
    root = os.path.join(data_path, dataset, "processed")

    # Create figure
    fig, axs = plt.subplots(
        1 + len(models_names) * len(models_modes),
        15,
        figsize=(5 * 15, 20),
    )

    # Create custom color map
    dark_red_cmap = utils.get_color_map()

    # Load images
    data = datasets.VOCDetectParsed(root=root, image_set=image_set)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=datasets.VOCDetectParsed.collate_fn,
    )

    # Get image of same class in same order as paper
    class_list = [
        "cat",
        "train",
        "boat",
        "person",
        "motorbike",
        "sheep",
        "tvmonitor",
        "chair",
        "cow",
        "horse",
        "person",
        "sheep",
        "boat",
        "train",
        "cow",
    ]
    # Class picked per image (images contain multiple classes)
    chosen_classes = []

    # Init batch of correct image with correct shape but batch size 0
    inputs, classes, bb_box_list = next(iter(loader))
    inputs = torch.zeros((0, *inputs.shape[1:]))
    classes = torch.zeros((0, *classes.shape[1:]))
    bb_box_list = []

    # Find correct images in dataloader
    for class_name in class_list:
        class_number = utils.get_class_number(class_name)
        chosen_classes.append(class_number)
        # Get one random image
        input, clas, bb_box = next(iter(loader))
        while clas[0, class_number] != 1:
            input, clas, bb_box = next(iter(loader))
            if clas[0, class_number] == 1:
                break
        inputs = torch.cat((inputs, input), dim=0)
        classes = torch.cat((classes, clas), dim=0)
        bb_box_list.append(bb_box[0])

    # Loop over plots in upper row
    for i, ax in enumerate(axs[0]):
        # Get image
        image = inputs[i]
        # Get all bboxes for this image
        bbs = bb_box_list[i]

        # Pick choosen
        clas = chosen_classes[i]

        # Get class name from class number
        class_name = utils.get_class_name(clas)

        # Get bboxes from specific classes
        class_bbs = utils.filter_bbs(bbs, clas)

        # Show original image
        ax.imshow(torch.movedim(image[:3, :, :], 0, -1))
        ax.set_title(
            class_name,
            fontsize=45,
            # fontname="Times New Roman",
            pad=20,
            fontweight=650,
        )

        # Plot boundingboxes
        for coords in class_bbs:
            xmin, ymin, xmax, ymax = coords
            ax.add_patch(
                patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fc="none",
                    ec="#00FFFF",
                    lw=5,
                )
            )

    # Loop over model names and modes
    for i, model_name in enumerate(models_names):
        for j, model_mode in enumerate(models_modes):
            print(model_name, model_mode)
            # Calculate row index
            row_idx = (i * len(models_modes)) + (j + 1)
            # Get model path
            path = models_paths[i][j]
            if last:
                epoch = 50
                if model_mode == "Baseline":
                    epoch = 300
                path = utils.switch_best_to_last(path, epoch)
            # Get model spec from path
            (
                model_backbone,
                localization_loss_fn,
                layer,
                attribution_method,
            ) = utils.get_model_specs(path)

            # If no attribution method set default values
            if not attribution_method:
                if model_backbone == "bcos":
                    attribution_method = "BCos"
                elif model_backbone == "vanilla":
                    attribution_method = "IxG"

            # default localistion loss is energy
            if not localization_loss_fn:
                localization_loss_fn = "Energy"

            # If fixed layers are given set layer
            if fix_layers:
                layer = fix_layers[i]

            # Get model, attributor, en transform based on specs
            model, attributor, transformer = utils.get_model(
                model_backbone,
                localization_loss_fn,
                layer,
                attribution_method,
                dataset,
                path,
            )
            model.eval()

            # apply transform
            transformer.dim = -3
            X = transformer(inputs.clone())

            # Get output from model
            X.requires_grad = True
            X = X.cuda()
            logits, features = model(X)

            # Get attributions per image
            for img_idx, image in enumerate(X):
                pred = chosen_classes[img_idx]
                attributions = (
                    attributor(features, logits, pred, img_idx)
                    .detach()
                    .squeeze(0)
                    .squeeze(0)
                )
                positive_attributions = attributions.clamp(min=0).cpu()
                bb = utils.filter_bbs(bb_box_list[img_idx], pred)

                # Plot attribution map
                axs[row_idx][img_idx].imshow(positive_attributions, cmap=dark_red_cmap)
                # Plot boundingbox
                for coords in bb:
                    xmin, ymin, xmax, ymax = coords
                    axs[row_idx][img_idx].add_patch(
                        patches.Rectangle(
                            (xmin, ymin),
                            xmax - xmin,
                            ymax - ymin,
                            fc="none",
                            ec="#4169E1",
                            lw=5,
                        )
                    )
                # Set row title
                if img_idx == 0:
                    axs[row_idx][img_idx].set_ylabel(
                        f"{model_name} \n {model_mode}",
                        fontsize=45,
                        # fontname="Times New Roman",
                    )

    # Remove plot ticks
    for ax_ in axs:
        for ax in ax_:
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

    # Save figure
    last = "Last" if last else "Best"
    fig.tight_layout()
    plt.savefig(f"./images/Figure2_{last}.png")


def visualize_fig13(base_model_pth, ft_energy_path, ft_l1_path, last=True):
    # define constants
    fix_layer = "Input"
    data_path = "datasets/"
    dataset = "WATERBIRDS"
    image_set = "test"
    models = [base_model_pth, ft_energy_path, ft_l1_path]
    model_names = ["Baseline", "Energy", "L1"]
    # Get dataset path
    root = os.path.join(data_path, dataset, "processed")

    if last:
        for i, path in enumerate(models):
            models[i] = utils.switch_best_to_last(path, 350)

    num_images = 5

    # Create figure
    fig, axs = plt.subplots(
        num_images,
        4,
        figsize=(num_images * 2.5, 20),
    )

    # Create custom color map
    dark_red_cmap = utils.get_color_map()

    # Load images
    data = datasets.VOCDetectParsed(root=root, image_set=image_set)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=num_images,
        shuffle=True,
        num_workers=0,
        collate_fn=datasets.VOCDetectParsed.collate_fn,
    )

    # Init batch of correct image with correct shape but batch size 0
    inputs, classes, bb_box_list = next(iter(loader))

    # Loop over plots in upper row
    for i, ax in enumerate(axs[:, 0]):
        # Get image
        image = inputs[i]
        # Get all bboxes for this image
        bbs = bb_box_list[i]

        # Pick choosen
        clas = torch.where(classes[i] == 1)[0]

        # Get class name from class number
        class_name = utils.get_waterbird_name(clas)

        # Get bboxes from specific classes
        class_bbs = utils.filter_bbs(bbs, clas)

        # Show original image
        ax.imshow(torch.movedim(image[:3, :, :], 0, -1))
        ax.set_title(
            class_name,
            fontsize=20,
            # fontname="Times New Roman",
        )

        # Plot boundingboxes
        for coords in class_bbs:
            xmin, ymin, xmax, ymax = coords
            ax.add_patch(
                patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fc="none",
                    ec="#00FFFF",
                    lw=5,
                )
            )

    # Loop over model names and modes
    for i, model_path in enumerate(models, 1):
        print(model_path)
        # Get model spec from path
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

        # If fixed layers are given set layer
        if fix_layer:
            layer = fix_layer

        # Get model, attributor, en transform based on specs
        model, attributor, transformer = utils.get_model(
            model_backbone,
            localization_loss_fn,
            layer,
            attribution_method,
            dataset,
            model_path,
        )
        model.eval()
        # apply transform
        transformer.dim = -3
        X = transformer(inputs.clone())

        # Get output from model
        X.requires_grad = True
        X = X.cuda()
        logits, features = model(X)

        # Get attributions per image
        for img_idx, image in enumerate(X):
            pred = torch.where(classes[img_idx] == 1)[0]
            attributions = (
                attributor(features, logits, pred, img_idx)
                .detach()
                .squeeze(0)
                .squeeze(0)
            )
            positive_attributions = attributions.clamp(min=0).cpu()
            bb = utils.filter_bbs(bb_box_list[img_idx], pred)

            # Plot attribution map
            axs[img_idx][i].imshow(positive_attributions, cmap=dark_red_cmap)
            # Plot boundingbox
            for coords in bb:
                xmin, ymin, xmax, ymax = coords
                axs[img_idx][i].add_patch(
                    patches.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fc="none",
                        ec="#4169E1",
                        lw=5,
                    )
                )
            # Set row title
            axs[img_idx][i].set_title(
                f"{model_names[i-1]}",
                fontsize=20,
                # fontname="Times New Roman",
            )
            pred_clas = int(logits[img_idx].argmax().item())
            confidence = logits[img_idx][pred_clas].sigmoid().item() * 100
            pred_clas_name = utils.get_waterbird_name(pred_clas)
            axs[img_idx][i].set_xlabel(
                f"{pred_clas_name}\nConf.:{confidence:.0f}%",
                fontsize=20,
                # fontname="Times New Roman",
            )

    # Remove plot ticks
    for ax_ in axs:
        for ax in ax_:
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

    # Save figure
    last = "Last" if last else "Best"
    fig.tight_layout()
    plt.savefig(f"./images/Figure13_{last}.png")


def get_one_laytout(i):
    j = str(i + 1)
    i = str(i)
    layout = [
        ["A" + i, "B" + i, "C" + i, "A" + j, "B" + j, "C" + j],
        ["D" + i, "E" + i, "F" + i, "D" + j, "E" + j, "F" + j],
        ["G" + i, "H" + i, "F" + i, "G" + j, "H" + j, "F" + j],
    ]

    return layout


def visualize_fig11(
    base_path,
    energy_paths,
    L1_paths,
    last=False
):
    if last:
        base_path = utils.switch_best_to_last(base_path, 300)

        for i, path in enumerate(energy_paths):
            energy_paths[i] = utils.switch_best_to_last(path, 50)

        for i, path in enumerate(L1_paths):
            L1_paths[i] = utils.switch_best_to_last(path, 50)

    # define constants
    num_images = 6
    fix_layer = "Input"
    data_path = "datasets/"
    dataset = "VOC2007"
    image_set = "test"
    titles = ["50%", "0%", "Baseline"]
    # Get dataset path
    root = os.path.join(data_path, dataset, "processed")

    assert num_images % 2 == 0
    assert num_images >= 2
    layout = []
    for i in range(0, num_images, 2):
        layout += get_one_laytout(i)

    # Create custom color map
    dark_red_cmap = utils.get_color_map()
    fig, axd = plt.subplot_mosaic(layout, figsize=(15, num_images * 4))

    # Load images
    data = datasets.VOCDetectParsed(root=root, image_set=image_set)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=num_images,
        shuffle=True,
        num_workers=0,
        collate_fn=datasets.VOCDetectParsed.collate_fn,
    )
    # Init batch of correct image with correct shape but batch size 0
    inputs, classes, bb_box_list = next(iter(loader))

    chosen_classes = []

    for i in range(num_images):
        # Get image
        image = inputs[i]
        # Get all bboxes for this image
        bbs = bb_box_list[i]
        # Pick one random class from all classes in image
        clas = np.random.choice(torch.where(classes[i] == 1)[0])
        chosen_classes.append(clas)

        # Get class name from class number
        class_name = utils.get_class_name(clas)

        # Get bboxes from specific classes
        class_bbs = utils.filter_bbs(bbs, clas)
        for j, letter in enumerate(["A", "B", "C"]):
            i = str(i)
            # Show original image
            axd[letter + i].imshow(torch.movedim(image[:3, :, :], 0, -1))
            axd[letter + i].axes.get_xaxis().set_ticks([])
            axd[letter + i].axes.get_yaxis().set_ticks([])
            axd[letter + i].set_title(
                titles[j],
                fontsize=20,
                # fontname="Times New Roman",
            )
            if letter == "A":
                axd[letter + i].set_ylabel(
                    class_name,
                    fontsize=20,
                    # fontname="Times New Roman",
                )
            if letter != "C":
                if letter == "A":
                    en_class_bbs = utils.enlarge_bb(class_bbs, 0.5)
                else:
                    en_class_bbs = class_bbs
                # Plot boundingboxes
                for coords in en_class_bbs:
                    xmin, ymin, xmax, ymax = coords
                    axd[letter + i].add_patch(
                        patches.Rectangle(
                            (xmin, ymin),
                            xmax - xmin,
                            ymax - ymin,
                            fc="none",
                            ec="#00FFFF",
                            lw=5,
                        )
                    )
    # Loop over model names and modes
    models = [base_path] + L1_paths + energy_paths

    i_to_letter = ["F", "E", "D", "H", "G"]
    for i, model_path in enumerate(models):
        print(model_path)
        # Get model spec from path
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

        # If fixed layers are given set layer
        if fix_layer:
            layer = fix_layer

        # Get model, attributor, en transform based on specs
        model, attributor, transformer = utils.get_model(
            model_backbone,
            localization_loss_fn,
            layer,
            attribution_method,
            dataset,
            model_path,
        )
        model.eval()
        # apply transform
        transformer.dim = -3
        X = transformer(inputs.clone())

        # Get output from model
        X.requires_grad = True
        X = X.cuda()
        logits, features = model(X)

        # Get attributions per image
        for img_idx, image in enumerate(X):
            pred = chosen_classes[img_idx]
            attributions = (
                attributor(features, logits, pred, img_idx)
                .detach()
                .squeeze(0)
                .squeeze(0)
            )
            positive_attributions = attributions.clamp(min=0).cpu()
            bb = utils.filter_bbs(bb_box_list[img_idx], pred)

            img_idx = str(img_idx)
            letter = i_to_letter[i]
            # Plot attribution map
            axd[letter + img_idx].imshow(positive_attributions, cmap=dark_red_cmap)
            axd[letter + img_idx].axes.get_xaxis().set_ticks([])
            axd[letter + img_idx].axes.get_yaxis().set_ticks([])
            loss_fn = None
            if letter == "D":
                loss_fn = "L1"
            if letter == "G":
                loss_fn = "Energy"
            if loss_fn:
                axd[letter + img_idx].set_ylabel(
                    loss_fn,
                    fontsize=20,
                    # fontname="Times New Roman",
                )
            # Plot boundingbox
            for coords in bb:
                xmin, ymin, xmax, ymax = coords
                axd[letter + img_idx].add_patch(
                    patches.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fc="none",
                        ec="#4169E1",
                        lw=5,
                    )
                )

    # Save figure
    last = "Last" if last else "Best"
    fig.tight_layout()
    plt.savefig(f"./images/Figure11_{last}.png")

def plot_pareto_curve(
    baseline_data: Tuple[float, float] = (50, 50),
    energy_data: List[Tuple[float, float]] = [],
    l1_data: List[Tuple[float, float]] = [],
    ppce_data: List[Tuple[float, float]] = [],
    rrr_data: List[Tuple[float, float]] = [],
    x_label: str = 'F1 Score (%)', 
    y_label: str = 'EPG Score (%)', 
    title: str = '', 
    save_path: Optional[str] = None, 
    figsize: Tuple[int, int] = (10, 6),
    set_xlim: Optional[Tuple[float, float]] = None,
    set_ylim: Optional[Tuple[float, float]] = None,
    step_size_xticks: Optional[int] = None,
    step_size_yticks: Optional[int] = None,
    hide_y_ticks: bool = False,
    hide_x_ticks: bool = False,
    fontsize: int = 20,
    attribution_method: str = "IxG",
    plot_demo_data: bool = False,
    demo_data: List[Tuple[float, float]] = [],
    ) -> None:
    """
    Plots a Pareto curve with given data points.

    Args:
        baseline_data (Tuple[float, float]): A tuple containing x and y coordinates for the baseline point.
        energy_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the energy data points.
        l1_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the L1 data points.
        ppce_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the PPCE data points.
        rrr_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the RRR* data points.
        x_label (str, optional): Label for the x-axis. Defaults to 'F1 Score (%)'.
        y_label (str, optional): Label for the y-axis. Defaults to 'EPG Score (%)'.
        title (str, optional): Title of the plot. Defaults to 'Pareto Curve'.
        save_path (Optional[str], optional): Path to save the plot image. If None, the plot is displayed. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).

    Returns:
        None
    """
    
    fig, ax = plt.subplots(figsize=figsize)

    # Helper function for plotting data points and lines
    def plot_data(data, marker, color, label):
        if data:
            x, y = zip(*data)
            ax.scatter(x, y, marker=marker, color=color, label=label, s=fontsize*10, edgecolors='black', zorder=3)
            ax.plot(x, y, color=color, linestyle='--')

    if not plot_demo_data:
        # Plot each dataset
        plot_data(energy_data, 'o', '#FF006F', 'Energy')
        plot_data(l1_data, 'v', '#00E49F', 'L1')
        plot_data(ppce_data, 'p', '#FFD562', 'PPCE')
        plot_data(rrr_data, 'D', '#008AB3', 'RRR*')
    
    else:

        # Remove points from l1_data from demo_data
        demo_data = [point for point in demo_data if point not in l1_data]

        x, y = zip(*demo_data)
        ax.scatter(x, y, marker="v", color="#00E49F", label="L1 (All Checkpoints)", s=fontsize*10, edgecolors='black', zorder=2, alpha=0.5)

        x, y = zip(*l1_data)
        ax.scatter(x, y, marker="v", color="#00E49F", label="L1 (Pareto Front Highlighted)", s=fontsize*10, edgecolors='black', zorder=3)
        ax.plot(x, y, color="#00E49F", linestyle='--', zorder=3)


    # if baseline isn't empty, plot the baseline data
    if baseline_data != []:

        # Plot Baseline
        ax.plot(baseline_data[0], baseline_data[1], marker='X', color='white', markersize=fontsize, markeredgewidth=2, markeredgecolor='black', label='Baseline', zorder=2)

        # Find the best limits for the plot
        x_values = [x for data in [energy_data, l1_data, ppce_data, rrr_data] for x, y in data] + [baseline_data[0]]
        y_values = [y for data in [energy_data, l1_data, ppce_data, rrr_data] for x, y in data] + [baseline_data[1]]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

    else:
        
        # Find the best limits for the plot
        x_values = [x for data in [energy_data, l1_data, ppce_data, rrr_data] for x, y in data]
        y_values = [y for data in [energy_data, l1_data, ppce_data, rrr_data] for x, y in data]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

    # Set the limits with some padding
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15

    if set_xlim:
        ax.set_xlim(set_xlim[0], set_xlim[1])

    else:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

    if set_ylim:
        ax.set_ylim(set_ylim[0], set_ylim[1])

    else:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    if baseline_data != []:
        # Dominated region (gray)
        ax.plot([0, baseline_data[0]], [baseline_data[1], baseline_data[1]], '--', color='gray', alpha=0.5, zorder=1)
        ax.plot([baseline_data[0], baseline_data[0]], [-20, baseline_data[1]], '--', color='gray', alpha=0.5, zorder=1)
        ax.fill_between([baseline_data[0], -20], [baseline_data[1], baseline_data[1]], [-20, -20], color='gray', alpha=0.1, zorder=1)
        
        # Dominating region (green)
        ax.plot([baseline_data[0], 120], [baseline_data[1], baseline_data[1]], '--', color='green', alpha=0.5, zorder=1)
        ax.plot([baseline_data[0], baseline_data[0]], [120, baseline_data[1]], '--', color='green', alpha=0.5, zorder=1)
        ax.fill_between([120, baseline_data[0]], [baseline_data[1], baseline_data[1]], [120, 120], color='green', alpha=0.1, zorder=1)

        # Place text using axes fraction
        ax.text(0.015, 0.05, 'Dominated', transform=ax.transAxes, fontsize=15, color='gray', alpha=1, fontstyle='italic')
        ax.text(0.02, 0.77, attribution_method, transform=ax.transAxes, fontsize=fontsize + 6, color='darkblue', alpha=1)
        ax.text(0.99, 0.56, 'Dominating', transform=ax.transAxes, fontsize=15, color='green', alpha=1, fontstyle='italic', rotation=90, ha='right')

    # Legend, Axes, and Labels
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=5, fancybox=True, shadow=True, fontsize=fontsize, columnspacing=0, handletextpad=0.0)

    if not hide_x_ticks:
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
    else:
        # Hide x-axis tick labels
        ax.set_xlabel('')
        
        # Make the x-axis ticks white
        ax.tick_params(axis='x', labelsize=fontsize, colors='white', length=0)

    if not hide_y_ticks:
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
    else:
        # Hide y-axis tick labels
        ax.set_ylabel('')

        # Make the y-axis ticks white
        ax.tick_params(axis='y', labelsize=fontsize, colors='white', length=0)

    if step_size_xticks:
        ax.xaxis.set_major_locator(plt.MultipleLocator(step_size_xticks))
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))


    if step_size_yticks:
        ax.yaxis.set_major_locator(plt.MultipleLocator(step_size_yticks))
    else:
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))

    # Set title and grid
    ax.set_title(title, fontsize=16, fontstyle='italic')
    ax.grid(True)

    # Adjust layout for consistent PNG height
    plt.tight_layout()

    # Save or show plot depending on save_path
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pareto_curve_speed_up(
        baseline_data: Tuple[float, float] = (50, 50),
        energy_data_input_layer: List[Tuple[float, float]] = [], 
        energy_data: List[Tuple[float, float]] = [],
        speed_up_text: str = '',
        x_label: str = 'F1 Score (%)', 
        y_label: str = 'EPG Score (%)', 
        title: str = '', 
        save_path: Optional[str] = None, 
        figsize: Tuple[int, int] = (10, 6),
        set_xlim: Optional[Tuple[float, float]] = None,
        set_ylim: Optional[Tuple[float, float]] = None,
        step_size_xticks: Optional[int] = None,
        step_size_yticks: Optional[int] = None,
        hide_x_ticks: bool = False,
        hide_y_ticks: bool = False,
        fontsize: int = 20,
    ) -> None:
    """
    Plots a Pareto curve with given data points.

    Args:
        baseline_data (Tuple[float, float]): A tuple containing x and y coordinates for the baseline point.
        energy_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the energy data points.
        l1_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the L1 data points.
        ppce_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the PPCE data points.
        rrr_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the RRR* data points.
        x_label (str, optional): Label for the x-axis. Defaults to 'F1 Score (%)'.
        y_label (str, optional): Label for the y-axis. Defaults to 'EPG Score (%)'.
        title (str, optional): Title of the plot. Defaults to 'Pareto Curve'.
        save_path (Optional[str], optional): Path to save the plot image. If None, the plot is displayed. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Energy Input Layer
    if energy_data_input_layer:
        energy_x, energy_y = zip(*energy_data_input_layer)
        ax.plot(energy_x, energy_y, color='black', linestyle='--', label='@Input', zorder=2)

    # Energy
    if energy_data:
        energy_x, energy_y = zip(*energy_data)
        ax.scatter(energy_x, energy_y, marker='o', color='#FF006F', label='Energy', s=fontsize*10, edgecolors='black', zorder=3)
        ax.plot(energy_x, energy_y, color='#FF006F', linestyle='--', zorder=2)

    # Plot Baseline
    ax.plot(baseline_data[0], baseline_data[1], marker='X', color='white', markersize=fontsize, markeredgewidth=2, markeredgecolor='black', label='Baseline', zorder=2)

    # Find the best limits for the plot
    x_values = [x for data in [energy_data] for x, y in data] + [baseline_data[0]]
    y_values = [y for data in [energy_data] for x, y in data] + [baseline_data[1]]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Set the limits with some padding
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15

    if set_xlim:
        ax.set_xlim(set_xlim[0], set_xlim[1])
    else:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

    if set_ylim:
        ax.set_ylim(set_ylim[0], set_ylim[1])
    else:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Dominated region (gray)
    ax.plot([0, baseline_data[0]], [baseline_data[1], baseline_data[1]], '--', color='gray', alpha=0.5, zorder=1)
    ax.plot([baseline_data[0], baseline_data[0]], [-20, baseline_data[1]], '--', color='gray', alpha=0.5, zorder=1)
    ax.fill_between([baseline_data[0], -20], [baseline_data[1], baseline_data[1]], [-20, -20], color='gray', alpha=0.1, zorder=1)
    
    # Dominating region (green)
    ax.plot([baseline_data[0], 120], [baseline_data[1], baseline_data[1]], '--', color='green', alpha=0.5, zorder=1)
    ax.plot([baseline_data[0], baseline_data[0]], [120, baseline_data[1]], '--', color='green', alpha=0.5, zorder=1)
    ax.fill_between([120, baseline_data[0]], [baseline_data[1], baseline_data[1]], [120, 120], color='green', alpha=0.1, zorder=1)

    # Place text using axes fraction
    ax.text(0.015, 0.05, 'Dominated', transform=ax.transAxes, fontsize=15, color='gray', alpha=1, fontstyle='italic')
    # ax.text(0.02, 0.77, attribution_method, transform=ax.transAxes, fontsize=fontsize + 6, color='darkblue', alpha=1)
    ax.text(0.99, 0.56, 'Dominating', transform=ax.transAxes, fontsize=15, color='green', alpha=1, fontstyle='italic', rotation=90, ha='right')

    # Place speed-up
    ax.text(0.02, 0.175, speed_up_text, transform=ax.transAxes, fontsize=24, color='black', alpha=1, fontstyle='italic')

    # Legend, Axes, and Labels
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=5, fancybox=True, shadow=True, fontsize=fontsize, columnspacing=0, handletextpad=0.0)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)

    if not hide_x_ticks:
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
    else:
        # Hide x-axis tick labels
        ax.set_xlabel('')
        
        # Make the x-axis ticks white
        ax.tick_params(axis='x', labelsize=fontsize, colors='white', length=0)

    if not hide_y_ticks:
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
    else:
        # Hide y-axis tick labels
        ax.set_ylabel('')

        # Make the y-axis ticks white
        ax.tick_params(axis='y', labelsize=fontsize, colors='white', length=0)

    if step_size_xticks:
        ax.xaxis.set_major_locator(plt.MultipleLocator(step_size_xticks))
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))


    if step_size_yticks:
        ax.yaxis.set_major_locator(plt.MultipleLocator(step_size_yticks))
    else:
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))

    # Set title and grid
    ax.set_title(title, fontsize=16, fontstyle='italic')
    ax.grid(True)

    # Adjust layout for consistent PNG height
    plt.tight_layout()

    # Save or show plot depending on save_path
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_pareto_curve_dilation(
    baseline_data: Tuple[float, float] = (50, 50),
    data_0: List[Tuple[float, float]] = [],
    data_01: List[Tuple[float, float]] = [],
    data_025: List[Tuple[float, float]] = [],
    data_05: List[Tuple[float, float]] = [],
    data_0_not_pareto : List[Tuple[float, float]] = [],
    data_01_not_pareto : List[Tuple[float, float]] = [],
    data_025_not_pareto : List[Tuple[float, float]] = [],
    data_05_not_pareto : List[Tuple[float, float]] = [],
    loss: str = 'Energy',
    x_label: str = 'F1 Score (%)', 
    y_label: str = 'EPG Score (%)', 
    title: str = '', 
    save_path: Optional[str] = None, 
    figsize: Tuple[int, int] = (10, 6),
    set_xlim: Optional[Tuple[float, float]] = None,
    set_ylim: Optional[Tuple[float, float]] = None,
    step_size_xticks: Optional[int] = None,
    step_size_yticks: Optional[int] = None,
    hide_x_ticks: bool = False,
    hide_y_ticks: bool = False,
    fontsize: int = 20
    ) -> None:
    """
    Plots a Pareto curve with given data points.

    Args:
        baseline_data (Tuple[float, float]): A tuple containing x and y coordinates for the baseline point.
        energy_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the energy data points.
        l1_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the L1 data points.
        ppce_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the PPCE data points.
        rrr_data (List[Tuple[float, float]]): A list of tuples containing x and y coordinates for the RRR* data points.
        x_label (str, optional): Label for the x-axis. Defaults to 'F1 Score (%)'.
        y_label (str, optional): Label for the y-axis. Defaults to 'EPG Score (%)'.
        title (str, optional): Title of the plot. Defaults to 'Pareto Curve'.
        save_path (Optional[str], optional): Path to save the plot image. If None, the plot is displayed. Defaults to None.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).

    Returns:
        None
    """
    
    fig, ax = plt.subplots(figsize=figsize)

    # Plot Baseline
    ax.plot(baseline_data[0], baseline_data[1], marker='X', color='white', markersize=fontsize, markeredgewidth=2, markeredgecolor='black', label='Baseline', zorder=2)

    # Helper function for plotting data points and lines
    def plot_data(data, marker, color, label, alpha, zorder=3):
        if data:
            x, y = zip(*data)
            ax.scatter(x, y, marker=marker, color=color, label=label, s=fontsize*10, edgecolors='black', alpha=alpha, zorder=zorder)

    # Plot each dataset
    if loss == 'Energy':
        plot_data(data_0, 'o', '#09236c', '0%', 1, zorder=3)
        plot_data(data_01, 'o', '#0082c1', '10%', 1, zorder=3)
        plot_data(data_025, 'o', '#9fd0e6', '25%', 1, zorder=3)
        plot_data(data_05, 'o', '#f6fbff', '50%', 1, zorder=3)
        plot_data(data_0_not_pareto, 'o', '#09236c', '', 0.5, zorder=2)
        plot_data(data_01_not_pareto, 'o', '#0082c1', '', 0.5, zorder=2)
        plot_data(data_025_not_pareto, 'o', '#9fd0e6', '', 0.5, zorder=2)
        plot_data(data_05_not_pareto, 'o', '#f6fbff', '', 0.5, zorder=2)

    elif loss == 'L1':
        plot_data(data_0, 'v', '#09236c', '0%', 1, zorder=3)
        plot_data(data_01, 'v', '#0082c1', '10%', 1, zorder=3)
        plot_data(data_025, 'v', '#9fd0e6', '25%', 1, zorder=3)
        plot_data(data_05, 'v', '#f6fbff', '50%', 1, zorder=3)
        plot_data(data_0_not_pareto, 'v', '#09236c', '', 0.5, zorder=2)
        plot_data(data_01_not_pareto, 'v', '#0082c1', '', 0.5, zorder=2)
        plot_data(data_025_not_pareto, 'v', '#9fd0e6', '', 0.5, zorder=2)
        plot_data(data_05_not_pareto, 'v', '#f6fbff', '', 0.5, zorder=2)

    # Find the best limits for the plot
    x_values = [x for data in [data_0] for x, y in data] + [baseline_data[0]]
    y_values = [y for data in [data_0] for x, y in data] + [baseline_data[1]]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    # Set the limits with some padding
    x_pad = (x_max - x_min) * 0.15
    y_pad = (y_max - y_min) * 0.15

    if set_xlim:
        ax.set_xlim(set_xlim[0], set_xlim[1])

    else:
        ax.set_xlim(x_min - x_pad, x_max + x_pad)

    if set_ylim:
        ax.set_ylim(set_ylim[0], set_ylim[1])

    else:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    # Dominated region (gray)
    ax.plot([0, baseline_data[0]], [baseline_data[1], baseline_data[1]], '--', color='gray', alpha=0.5, zorder=1)
    ax.plot([baseline_data[0], baseline_data[0]], [-20, baseline_data[1]], '--', color='gray', alpha=0.5, zorder=1)
    ax.fill_between([baseline_data[0], -20], [baseline_data[1], baseline_data[1]], [-20, -20], color='gray', alpha=0.1, zorder=1)
    
    # Dominating region (green)
    ax.plot([baseline_data[0], 120], [baseline_data[1], baseline_data[1]], '--', color='green', alpha=0.5, zorder=1)
    ax.plot([baseline_data[0], baseline_data[0]], [120, baseline_data[1]], '--', color='green', alpha=0.5, zorder=1)
    ax.fill_between([120, baseline_data[0]], [baseline_data[1], baseline_data[1]], [120, 120], color='green', alpha=0.1, zorder=1)

    # Place text using axes fraction
    ax.text(0.015, 0.02, 'Dominated', transform=ax.transAxes, fontsize=15, color='gray', alpha=1, fontstyle='italic')

    ax.text(0.99, 0.76, 'Dominating', transform=ax.transAxes, fontsize=15, color='green', alpha=1, fontstyle='italic', rotation=90, ha='right')

    # Legend, Axes, and Labels
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=5, fancybox=True, shadow=True, fontsize=fontsize, columnspacing=0, handletextpad=0.0)
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)

    if not hide_x_ticks:
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
    else:
        # Hide x-axis tick labels
        ax.set_xlabel('')

        # Make the x-axis ticks white
        ax.tick_params(axis='x', labelsize=fontsize, colors='white', length=0)

    if not hide_y_ticks:
        ax.set_ylabel(y_label, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
    else:
        # Hide y-axis tick labels
        ax.set_ylabel('')

        # Make the y-axis ticks white
        ax.tick_params(axis='y', labelsize=fontsize, colors='white', length=0)

    if step_size_xticks:
        ax.xaxis.set_major_locator(plt.MultipleLocator(step_size_xticks))
    else:
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))


    if step_size_yticks:
        ax.yaxis.set_major_locator(plt.MultipleLocator(step_size_yticks))
    else:
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        
            
    ax.set_title(title, fontsize=16, fontstyle='italic')
    ax.grid(True)

    # Adjust layout for consistent PNG height
    plt.tight_layout()

    # Save or show plot depending on save_path
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_names",
        type=list,
        default=["B-cos", "IxG"],
        help="Model backbones to plot.",
    )
    parser.add_argument(
        "--models_modes",
        type=list,
        default=["Baseline", "Guided"],
        help="Model modes to plot.",
    )
    parser.add_argument(
        "--models_paths",
        type=list,
        default=[
            [
                "BASE/VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt",
                "FT/VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.005_layerInput/model_checkpoint_f1_best.pt",
            ],
            [
                "BASE/VOC2007/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt",
                "FT/VOC2007/vanilla_finetunedobjlocpareto_attrIxG_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.005_layerFinal/model_checkpoint_f1_best.pt",
            ],
        ],
        help="Model modes to plot.",
    )
    parser.add_argument("--fix_layers", type=list, default=["Input", "Final"])
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
        "--image_set",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        default=False,
        help="Flag to indicate to show last model epoch instead of best",
    )
    args = parser.parse_args()
    args = vars(args)
    visualize_fig2(**args)
    visualize_fig9(
        [
            "BASE/VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput/model_checkpoint_f1_best.pt",
            "FT/VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.005_layerInput/model_checkpoint_f1_best.pt",
            "FT/VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossL1_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.01_layerInput/model_checkpoint_f1_best.pt",
            "FT/VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossPPCE_origmodel_checkpoint_f1_best.pt_resnet50_lr0.001_sll0.001_layerInput/model_checkpoint_f1_best.pt",
            "FT/VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossRRR_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll5e-05_layerInput/model_checkpoint_f1_best.pt",
        ],
        last=False,
    )

    # visualize_fig13(
    #     "WBBASE\WATERBIRDS/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput\model_checkpoint_f1_best.pt",
    #     "WBFT\WATERBIRDS/bcos_finetunedobjlocpareto_limited_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0005_sll0.1_layerInputlimited0.01\model_checkpoint_f1_best.pt",
    #     "WBFT\WATERBIRDS/bcos_finetunedobjlocpareto_limited_attrBCos_loclossL1_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0005_sll0.1_layerInputlimited0.01\model_checkpoint_f1_best.pt",
    #     last=False,
    # )

    # visualize_fig11(
    #     base_path="BASE\VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.0001_sll1.0_layerInput\model_checkpoint_f1_best.pt",
    #     energy_paths=[
    #         "FT\DIL/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.005_layerInput\model_checkpoint_f1_best.pt",
    #         "FT\DIL/bcos_FT_dilated_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.005_layerInput_dil0.5\model_checkpoint_f1_best.pt",
    #     ],
    #     L1_paths=[
    #         "FT\DIL/bcos_finetunedobjlocpareto_attrBCos_loclossL1_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.005_layerInput\model_checkpoint_f1_best.pt",
    #         "FT\DIL/bcos_FT_dilated_attrBCos_loclossL1_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.005_layerInput_dil0.5\model_checkpoint_f1_best.pt",
    #     ],
    #     last=False,
    # )

    plot_pareto_curves = True

    if plot_pareto_curves:

        # Metric F1 vs EPG for VOC2007
        root_folder = './p_curves/VOC2007'
        data_f1_epg = utils.load_data_from_folders_with_npz_files(root_folder, metrics=('f_score', 'bb_score'))

        x_lim_range = (65, 85)
        y_lim_range = (31, 90)
        step_size_xticks = 5
        step_size_yticks = 10

        plot_pareto_curve(
            baseline_data=data_f1_epg['vanilla']['input']['baseline'],
            energy_data=data_f1_epg['vanilla']['input']['energy'],
            l1_data=data_f1_epg['vanilla']['input']['l1'],
            ppce_data=data_f1_epg['vanilla']['input']['ppce'],
            rrr_data=data_f1_epg['vanilla']['input']['rrr'],
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            title='',
            save_path='./images/fig_5_voc2007_vanilla_resnet50_input_layer_f1_epg_pareto_curve.png',
            figsize=(10, 4),
            set_xlim=x_lim_range,
            set_ylim=y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_x_ticks=True,
            hide_y_ticks=False,
            fontsize=20,
            attribution_method="IxG")

        plot_pareto_curve(
            baseline_data=data_f1_epg['bcos']['input']['baseline'],
            energy_data=data_f1_epg['bcos']['input']['energy'],
            l1_data=data_f1_epg['bcos']['input']['l1'],
            ppce_data=data_f1_epg['bcos']['input']['ppce'],
            rrr_data=data_f1_epg['bcos']['input']['rrr'],
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            title='',
            save_path='./images/fig_5_voc2007_bcos_resnet50_input_layer_f1_epg_pareto_curve.png',
            figsize=(10, 4),
            set_xlim=x_lim_range,
            set_ylim=y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_x_ticks=True,
            hide_y_ticks=True,
            fontsize=20,
            attribution_method="B-cos")

        plot_pareto_curve(
            baseline_data=data_f1_epg['vanilla']['final']['baseline'],
            energy_data=data_f1_epg['vanilla']['final']['energy'],
            l1_data=data_f1_epg['vanilla']['final']['l1'],
            ppce_data=data_f1_epg['vanilla']['final']['ppce'],
            rrr_data=data_f1_epg['vanilla']['final']['rrr'],
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            title='',
            save_path='./images/fig_5_voc2007_vanilla_resnet50_final_layer_f1_epg_pareto_curve.png',
            figsize=(10, 4),
            set_xlim=x_lim_range,
            set_ylim=y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_x_ticks=False,
            hide_y_ticks=False,
            fontsize=20,
            attribution_method="IxG")

        plot_pareto_curve(
            baseline_data=data_f1_epg['bcos']['final']['baseline'],
            energy_data=data_f1_epg['bcos']['final']['energy'],
            l1_data=data_f1_epg['bcos']['final']['l1'],
            ppce_data=data_f1_epg['bcos']['final']['ppce'],
            rrr_data=data_f1_epg['bcos']['final']['rrr'],
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            title='',
            save_path='./images/fig_5_voc2007_bcos_resnet50_final_layer_f1_epg_pareto_curve.png',
            figsize=(10, 4),
            set_xlim=x_lim_range,
            set_ylim=y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=(step_size_yticks),
            hide_x_ticks=False,
            hide_y_ticks=True,
            fontsize=20,
            attribution_method="B-cos")
        
        # Metric F1 vs IOU for VOC2007
        root_folder = './p_curves/VOC2007'
        data_f1_iou = utils.load_data_from_folders_with_npz_files(root_folder, metrics=('f_score', 'iou_score'))

        x_lim_range = (65, 85)
        y_lim_range = (11, 60)
        step_size_xticks = 5
        step_size_yticks = 10

        plot_pareto_curve(
            baseline_data=data_f1_iou['vanilla']['input']['baseline'],
            energy_data=data_f1_iou['vanilla']['input']['energy'],
            l1_data=data_f1_iou['vanilla']['input']['l1'],
            ppce_data=data_f1_iou['vanilla']['input']['ppce'],
            rrr_data=data_f1_iou['vanilla']['input']['rrr'],
            x_label='F1 Score (%)',
            y_label='IoU Score (%)',
            title='',
            save_path='./images/fig_6_voc2007_vanilla_resnet50_input_layer_f1_iou_pareto_curve.png',
            figsize=(10, 4),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=False,
            fontsize=20,
            attribution_method="IxG")

        plot_pareto_curve(
            baseline_data=data_f1_iou['bcos']['input']['baseline'],
            energy_data=data_f1_iou['bcos']['input']['energy'],
            l1_data=data_f1_iou['bcos']['input']['l1'],
            ppce_data=data_f1_iou['bcos']['input']['ppce'],
            rrr_data=data_f1_iou['bcos']['input']['rrr'],
            x_label='F1 Score (%)',
            y_label='IoU Score (%)',
            title='',
            save_path='./images/fig_6_voc2007_bcos_resnet50_input_layer_f1_iou_pareto_curve.png',
            figsize=(10, 4),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=True,
            fontsize=20,
            attribution_method="B-cos")

        plot_pareto_curve(
            baseline_data=data_f1_iou['vanilla']['final']['baseline'],
            energy_data=data_f1_iou['vanilla']['final']['energy'],
            l1_data=data_f1_iou['vanilla']['final']['l1'],
            ppce_data=data_f1_iou['vanilla']['final']['ppce'],
            rrr_data=data_f1_iou['vanilla']['final']['rrr'],
            x_label='F1 Score (%)',
            y_label='IoU Score (%)',
            title='',
            save_path='./images/fig_6_voc2007_vanilla_resnet50_final_layer_f1_iou_pareto_curve.png',
            figsize=(10, 4),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=False,
            fontsize=20,
            attribution_method="IxG")

        plot_pareto_curve(
            baseline_data=data_f1_iou['bcos']['final']['baseline'],
            energy_data=data_f1_iou['bcos']['final']['energy'],
            l1_data=data_f1_iou['bcos']['final']['l1'],
            ppce_data=data_f1_iou['bcos']['final']['ppce'],
            rrr_data=data_f1_iou['bcos']['final']['rrr'],
            x_label='F1 Score (%)',
            y_label='IoU Score (%)',
            save_path='./images/fig_6_voc2007_bcos_resnet50_final_layer_f1_iou_pareto_curve.png',
            figsize=(10, 4),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=True,
            fontsize=20,
            attribution_method="B-cos")

        # Metric F1 vs Adapted IOU for VOC2007
        root_folder = './p_curves/VOC2007'
        data_f1_adapt_iou = utils.load_data_from_folders_with_npz_files(root_folder, metrics=('f_score', 'adapt_iou_score'))

        x_lim_range = (65, 85)
        y_lim_range = (11, 40)
        step_size_xticks = 5
        step_size_yticks = 10

        plot_pareto_curve(
            baseline_data=data_f1_adapt_iou['vanilla']['input']['baseline'],
            energy_data=data_f1_adapt_iou['vanilla']['input']['energy'],
            l1_data=data_f1_adapt_iou['vanilla']['input']['l1'],
            ppce_data=data_f1_adapt_iou['vanilla']['input']['ppce'],
            rrr_data=data_f1_adapt_iou['vanilla']['input']['rrr'],
            x_label='F1 Score (%)',
            y_label='Adapted IoU Score (%)',
            save_path='./images/fig_6_voc2007_vanilla_resnet50_input_layer_f1_adapt_iou_pareto_curve.png',
            figsize=(10, 4),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=False,
            fontsize=20,
            attribution_method="IxG")

        plot_pareto_curve(
            baseline_data=data_f1_adapt_iou['bcos']['input']['baseline'],
            energy_data=data_f1_adapt_iou['bcos']['input']['energy'],
            l1_data=data_f1_adapt_iou['bcos']['input']['l1'],
            ppce_data=data_f1_adapt_iou['bcos']['input']['ppce'],
            rrr_data=data_f1_adapt_iou['bcos']['input']['rrr'],
            x_label='F1 Score (%)',
            y_label='Adapted IoU Score (%)',
            save_path='./images/fig_6_voc2007_bcos_resnet50_input_layer_f1_adapt_iou_pareto_curve.png',
            figsize=(10, 4),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=True,
            fontsize=20,
            attribution_method="B-cos")

        plot_pareto_curve(
            baseline_data=data_f1_adapt_iou['vanilla']['final']['baseline'],
            energy_data=data_f1_adapt_iou['vanilla']['final']['energy'],
            l1_data=data_f1_adapt_iou['vanilla']['final']['l1'],
            ppce_data=data_f1_adapt_iou['vanilla']['final']['ppce'],
            rrr_data=data_f1_adapt_iou['vanilla']['final']['rrr'],
            x_label='F1 Score (%)',
            y_label='Adapted IoU Score (%)',
            save_path='./images/fig_6_voc2007_vanilla_resnet50_final_layer_f1_adapt_iou_pareto_curve.png',
            figsize=(10, 4),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=False,
            fontsize=20,
            attribution_method="IxG")

        plot_pareto_curve(
            baseline_data=data_f1_adapt_iou['bcos']['final']['baseline'],
            energy_data=data_f1_adapt_iou['bcos']['final']['energy'],
            l1_data=data_f1_adapt_iou['bcos']['final']['l1'],
            ppce_data=data_f1_adapt_iou['bcos']['final']['ppce'],
            rrr_data=data_f1_adapt_iou['bcos']['final']['rrr'],
            x_label='F1 Score (%)',
            y_label='Adapted IoU Score (%)',
            save_path='./images/fig_6_voc2007_bcos_resnet50_final_layer_f1_adapt_iou_pareto_curve.png',
            figsize=(10, 4),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=True,
            fontsize=20,
            attribution_method="B-cos")
        

        # Metric F1 vs EPG for different layers for VOC2007 (Speed up)
        root_folder = './p_curves/VOC2007'
        data_speed_up_f1_epg = utils.load_data_from_folders_with_npz_files(root_folder, metrics=('f_score', 'bb_score'))

        x_lim_range = (74, 81)
        y_lim_range = (41, 90)
        step_size_xticks = 2
        step_size_yticks = 10

        plot_pareto_curve_speed_up(
            baseline_data=data_speed_up_f1_epg['bcos']['input']['baseline'],
            energy_data_input_layer=data_speed_up_f1_epg['bcos']['input']['energy'],
            energy_data=data_speed_up_f1_epg['bcos']['mid2']['energy'],
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            speed_up_text='Speed-up: 1.25x',
            save_path='./images/fig_8_voc2007_bcos_resnet50_speed_up_f1_epg_input_layer_mid2_layer.png',
            figsize=(10, 4),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=False,
            fontsize=20)

        plot_pareto_curve_speed_up(
            baseline_data=data_speed_up_f1_epg['bcos']['input']['baseline'],
            energy_data_input_layer=data_speed_up_f1_epg['bcos']['input']['energy'],
            energy_data=data_speed_up_f1_epg['bcos']['final']['energy'],
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            speed_up_text='Speed-up: 2.0x',
            save_path='./images/fig_8_voc2007_bcos_resnet50_speed_up_f1_epg_input_layer_final_layer.png',
            figsize=(10, 4),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=True,
            fontsize=20)
        
        # Plots for limited annotations
        root_folder = './p_c_ann'
        data_limited_ann_f1_epg = utils.load_data_from_folders_with_npz_files_with_limited_ann(root_folder, metrics=('f_score', 'bb_score'))

        x_lim_range = (75, 81)
        y_lim_range = (41, 90)
        step_size_xticks = 2
        step_size_yticks = 10

        plot_pareto_curve(
            baseline_data=data_limited_ann_f1_epg['bcos']['lim0.01']['baseline'],
            energy_data=data_limited_ann_f1_epg['bcos']['lim0.01']['energy'],
            l1_data=data_limited_ann_f1_epg['bcos']['lim0.01']['l1'],
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            save_path='./images/fig_12_voc2007_bcos_resnet50_limited_ann_0.01_f1_epg_pareto_curve.png',
            figsize=(8, 4),
            set_xlim=x_lim_range,
            set_ylim=y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=False,
            fontsize=20,
            attribution_method="")

        plot_pareto_curve(
            baseline_data=data_limited_ann_f1_epg['bcos']['lim0.1']['baseline'],
            energy_data=data_limited_ann_f1_epg['bcos']['lim0.1']['energy'],
            l1_data=data_limited_ann_f1_epg['bcos']['lim0.1']['l1'],
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            save_path='./images/fig_12_voc2007_bcos_resnet50_limited_ann_0.1_f1_epg_pareto_curve.png',
            figsize=(8, 4),
            set_xlim=x_lim_range,
            set_ylim=y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=True,
            fontsize=20,
            attribution_method="")

        plot_pareto_curve(
            baseline_data=data_limited_ann_f1_epg['bcos']['lim1.0']['baseline'],
            energy_data=data_limited_ann_f1_epg['bcos']['lim1.0']['energy'],
            l1_data=data_limited_ann_f1_epg['bcos']['lim1.0']['l1'],
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            save_path='./images/fig_12_voc2007_bcos_resnet50_limited_ann_1.0_f1_epg_pareto_curve.png',
            figsize=(8, 4),
            set_xlim=x_lim_range,
            set_ylim=y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_y_ticks=True,
            fontsize=20,
            attribution_method="")
        
        root_folder = './p_c_dil'
        data_dilation_f1_epg = utils.load_data_from_folders_with_npz_files_with_dilation(root_folder, metrics=('f_score', 'bb_score'))

        x_lim_range = (75, 81)
        y_lim_range = (41, 80)
        step_size_xticks = 2
        step_size_yticks = 10

        plot_pareto_curve_dilation(
            baseline_data=data_dilation_f1_epg['bcos']['dil0']['baseline'],
            data_0=data_dilation_f1_epg['bcos']['dil0']['energy'],
            data_01=data_dilation_f1_epg['bcos']['dil0.1']['energy'],
            data_025=data_dilation_f1_epg['bcos']['dil0.25']['energy'],
            data_05=data_dilation_f1_epg['bcos']['dil0.5']['energy'],
            data_0_not_pareto=data_dilation_f1_epg['bcos']['dil0_not_pareto']['energy'],
            data_01_not_pareto=data_dilation_f1_epg['bcos']['dil0.1_not_pareto']['energy'],
            data_025_not_pareto=data_dilation_f1_epg['bcos']['dil0.25_not_pareto']['energy'],
            data_05_not_pareto=data_dilation_f1_epg['bcos']['dil0.5_not_pareto']['energy'],
            loss='Energy',
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            save_path='./images/fig_10_voc2007_bcos_resnet50_dilation_loss_energy_f1_epg_pareto_curve.png',
            figsize=(8, 6),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_x_ticks=True,
            hide_y_ticks=False,
            fontsize=20)

        plot_pareto_curve_dilation(
            baseline_data=data_dilation_f1_epg['bcos']['dil0']['baseline'],
            data_0=data_dilation_f1_epg['bcos']['dil0']['l1'],
            data_01=data_dilation_f1_epg['bcos']['dil0.1']['l1'],
            data_025=data_dilation_f1_epg['bcos']['dil0.25']['l1'],
            data_05=data_dilation_f1_epg['bcos']['dil0.5']['l1'],
            data_0_not_pareto=data_dilation_f1_epg['bcos']['dil0_not_pareto']['l1'],
            data_01_not_pareto=data_dilation_f1_epg['bcos']['dil0.1_not_pareto']['l1'],
            data_025_not_pareto=data_dilation_f1_epg['bcos']['dil0.25_not_pareto']['l1'],
            data_05_not_pareto=data_dilation_f1_epg['bcos']['dil0.5_not_pareto']['l1'],
            loss='L1',
            x_label='F1 Score (%)',
            y_label='EPG Score (%)',
            save_path='./images/fig_10_voc2007_bcos_resnet50_dilation_loss_l1_f1_epg_pareto_curve.png',
            figsize=(8, 6),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_x_ticks=True,
            hide_y_ticks=True,
            fontsize=20)
        
        root_folder = './p_c_dil'
        data_dilation_f1_epg = utils.load_data_from_folders_with_npz_files_with_dilation(root_folder, metrics=('f_score', 'adapt_iou_score'))

        x_lim_range = (75, 81)
        y_lim_range = (11, 35)
        step_size_xticks = 2
        step_size_yticks = 10

        plot_pareto_curve_dilation(
            baseline_data=data_dilation_f1_epg['bcos']['dil0']['baseline'],
            data_0=data_dilation_f1_epg['bcos']['dil0']['energy'],
            data_01=data_dilation_f1_epg['bcos']['dil0.1']['energy'],
            data_025=data_dilation_f1_epg['bcos']['dil0.25']['energy'],
            data_05=data_dilation_f1_epg['bcos']['dil0.5']['energy'],
            data_0_not_pareto=data_dilation_f1_epg['bcos']['dil0_not_pareto']['energy'],
            data_01_not_pareto=data_dilation_f1_epg['bcos']['dil0.1_not_pareto']['energy'],
            data_025_not_pareto=data_dilation_f1_epg['bcos']['dil0.25_not_pareto']['energy'],
            data_05_not_pareto=data_dilation_f1_epg['bcos']['dil0.5_not_pareto']['energy'],
            loss='Energy',
            x_label='F1 Score (%)',
            y_label='Adapted IoU Score (%)',
            save_path='./images/fig_10_voc2007_bcos_resnet50_dilation_loss_energy_f1_adapt_iou_pareto_curve.png',
            figsize=(8, 6),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_x_ticks=False,
            hide_y_ticks=False,
            fontsize=20)

        plot_pareto_curve_dilation(
            baseline_data=data_dilation_f1_epg['bcos']['dil0']['baseline'],
            data_0=data_dilation_f1_epg['bcos']['dil0']['l1'],
            data_01=data_dilation_f1_epg['bcos']['dil0.1']['l1'],
            data_025=data_dilation_f1_epg['bcos']['dil0.25']['l1'],
            data_05=data_dilation_f1_epg['bcos']['dil0.5']['l1'],
            data_0_not_pareto=data_dilation_f1_epg['bcos']['dil0_not_pareto']['l1'],
            data_01_not_pareto=data_dilation_f1_epg['bcos']['dil0.1_not_pareto']['l1'],
            data_025_not_pareto=data_dilation_f1_epg['bcos']['dil0.25_not_pareto']['l1'],
            data_05_not_pareto=data_dilation_f1_epg['bcos']['dil0.5_not_pareto']['l1'],
            loss='L1',
            x_label='F1 Score (%)',
            y_label='Adapted IoU Score (%)',
            save_path='./images/fig_10_voc2007_bcos_resnet50_dilation_loss_l1_f1_adapt_iou_pareto_curve.png',
            figsize=(8, 6),
            set_xlim = x_lim_range,
            set_ylim = y_lim_range,
            step_size_xticks=step_size_xticks,
            step_size_yticks=step_size_yticks,
            hide_x_ticks=False,
            hide_y_ticks=True,
            fontsize=20)