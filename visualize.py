import torch
import argparse
import datasets
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utils
import numpy as np


def visualize_fig9(
    model_paths,
    fix_layer=None,
    data_path="datasets/",
    dataset="VOC2007",
    image_set="test",
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
            fontname="Times New Roman",
            fontweight=650,
        )
        axs[i][0].set_ylabel(
            class_name,
            fontsize=45,
            fontname="Times New Roman",
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
            print(i, model_backbone, localization_loss_fn)
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
                path,
            )
            model.eval()

            # apply transform
            transformer.dim = -3
            X = transformer(inputs.clone())
            X = X[i : i + 1]
            print(X.shape)
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
                    fontname="Times New Roman",
                    fontweight=650,
                )

    # Disable ticks
    for _ax in axs:
        for ax in _ax:
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])
    # Save figure
    fig.tight_layout()
    plt.savefig("Figure9.png")


def visualize_fig2(
    models_names,
    models_modes,
    models_paths,
    fix_layers=None,
    data_path="datasets/",
    dataset="VOC2007",
    image_set="test",
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
            fontname="Times New Roman",
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
                        fontname="Times New Roman",
                    )

    # Remove plot ticks
    for ax_ in axs:
        for ax in ax_:
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

    # Save figure
    fig.tight_layout()
    plt.savefig("Figure2.png")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--models_names",
    #     type=list,
    #     default=["B-cos", "IxG"],
    #     help="Model backbones to plot.",
    # )
    # parser.add_argument(
    #     "--models_modes",
    #     type=list,
    #     default=["Baseline", "Guided"],
    #     help="Model modes to plot.",
    # )
    # parser.add_argument(
    #     "--models_paths",
    #     type=list,
    #     default=[
    #         [
    #             "BASE\VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput\model_checkpoint_f1_best.pt",
    #             "FT\VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerInput\model_checkpoint_f1_best.pt",
    #         ],
    #         [
    #             "BASE\VOC2007/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput\model_checkpoint_f1_best.pt",
    #             "FT\VOC2007/vanilla_finetunedobjlocpareto_attrIxG_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerFinal\model_checkpoint_f1_best.pt",
    #         ],
    #     ],
    #     help="Model modes to plot.",
    # )
    # parser.add_argument("--fix_layers", type=list, default=["Input", "Final"])
    # parser.add_argument(
    #     "--data_path", type=str, default="datasets/", help="Path to datasets."
    # )
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="VOC2007",
    #     choices=["VOC2007", "COCO2014"],
    #     help="Dataset to train on.",
    # )
    # parser.add_argument(
    #     "--image_set",
    #     type=str,
    #     default="test",
    #     choices=["train", "val", "test"],
    #     help="Dataset to train on.",
    # )
    # args = parser.parse_args()
    # args = vars(args)
    # visualize_fig2(**args)
    visualize_fig9(
        [
            "BASE\VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput\model_checkpoint_f1_best.pt",
            "FT\VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerInput\model_checkpoint_f1_best.pt",
            "FT\VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossL1_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.005_layerInput\model_checkpoint_f1_best.pt",
            "FT\VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossPPCE_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.0005_layerInput\model_checkpoint_f1_best.pt",
            "FT\VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossRRR_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll1e-05_layerInput\model_checkpoint_f1_best.pt",
        ]
    )
