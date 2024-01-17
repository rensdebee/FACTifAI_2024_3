import torch
import torchvision
import datasets
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import bcos.data.transforms
import utils
import numpy as np
import matplotlib.colors as mcolors

if __name__ == "__main__":
    # Need to enter
    data_path = "datasets/"
    num_images = 15
    dataset = "VOC2007"
    models_names = ["B-cos", "IxG"]
    models_modes = ["Baseline", "Guided"]

    # Default
    fix_layers = ["Input", "Final"]  # Default None
    image_set = "test"
    models_paths = [
        [
            "BASE\VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput\model_checkpoint_f1_best.pt",
            "FT\VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerInput\model_checkpoint_f1_best.pt",
        ],
        [
            "BASE\VOC2007/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput\model_checkpoint_f1_best.pt",
            "FT\VOC2007/vanilla_finetunedobjlocpareto_attrIxG_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerFinal\model_checkpoint_f1_best.pt",
        ],
    ]

    root = os.path.join(data_path, dataset, "processed")
    fig, axs = plt.subplots(
        1 + len(models_names) * len(models_modes),
        num_images,
        figsize=(5 * num_images, 20),
    )
    # Create custom color map
    cdict = {
        "red": [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
        "green": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
        "blue": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
    }

    dark_red_cmap = mcolors.LinearSegmentedColormap("DarkRed", cdict)

    test_data = datasets.VOCDetectParsed(root=root, image_set=image_set)
    loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=num_images,
        shuffle=True,
        num_workers=1,
        collate_fn=datasets.VOCDetectParsed.collate_fn,
    )

    inputs, classes, bb_box_list = next(iter(loader))
    chosen_classes = []
    for i, ax in enumerate(axs[0]):
        image = inputs[i]
        bbs = bb_box_list[i]
        clas = np.random.choice(torch.where(classes[i] == 1)[0])
        chosen_classes.append(clas)

        class_name = utils.get_class_name(clas)
        class_bbs = utils.filter_bbs(bbs, clas)
        ax.imshow(torch.movedim(image[:3, :, :], 0, -1))
        ax.set_title(
            class_name,
            fontsize=45,
            fontname="Times New Roman",
            pad=20,
            fontweight=650,
        )
        for coords in class_bbs:
            xmin, ymin, xmax, ymax = coords
            ax.add_patch(
                patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fc="none",
                    ec="royalblue",
                    lw=2,
                )
            )
    for i, model_name in enumerate(models_names):
        for j, model_mode in enumerate(models_modes):
            print(model_name, model_mode)
            row_idx = (i * len(models_modes)) + (j + 1)
            path = models_paths[i][j]

            (
                model_backbone,
                localization_loss_fn,
                layer,
                attribution_method,
            ) = utils.get_model_specs(path)

            if not attribution_method:
                if model_backbone == "bcos":
                    attribution_method = "BCos"
                elif model_backbone == "vanilla":
                    attribution_method = "IxG"
            if not localization_loss_fn:
                localization_loss_fn = "Energy"
            if fix_layers:
                layer = fix_layers[i]

            model, attributor, transformer = utils.get_model(
                path,
                model_backbone,
                localization_loss_fn,
                layer,
                attribution_method,
                dataset,
            )
            model.eval()
            transformer.dim = -3
            X = transformer(inputs.clone())
            X.requires_grad = True
            X = X.cuda()
            logits, features = model(X)
            for img_idx, image in enumerate(X):
                class_target = torch.where(classes[img_idx] == 1)[0]
                pred = chosen_classes[img_idx]
                attributions = (
                    attributor(features, logits, pred, img_idx)
                    .detach()
                    .squeeze(0)
                    .squeeze(0)
                )
                positive_attributions = attributions.clamp(min=0).cpu()
                bb = utils.filter_bbs(bb_box_list[img_idx], pred)
                axs[row_idx][img_idx].imshow(positive_attributions, cmap=dark_red_cmap)
                for coords in bb:
                    xmin, ymin, xmax, ymax = coords
                    axs[row_idx][img_idx].add_patch(
                        patches.Rectangle(
                            (xmin, ymin),
                            xmax - xmin,
                            ymax - ymin,
                            fc="none",
                            ec="royalblue",
                            lw=2,
                        )
                    )
                if img_idx == 0:
                    axs[row_idx][img_idx].set_ylabel(
                        f"{model_name} \n {model_mode}",
                        fontsize=45,
                        fontname="Times New Roman",
                    )
    for ax_ in axs:
        for ax in ax_:
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

    fig.tight_layout()
    # fig.suptitle("Comparision of threshold method for IoU score:    ")
    plt.savefig("Figure2.png")
