import torch
import os
import datasets
import utils
import bcos.models
import model_activators
import attribution_methods
import hubconf
import bcos
import bcos.modules
import bcos.data
from eval import eval_model
import re

def main(bin_width=0.005,
        layer="Final",
        data_split="test",
        attribution_method="BCos",
        eval_batch_size=4,
        model_dir="./FT/VOC2007/bcos/fin/l1/bcos_finetunedobjlocpareto_attrBCos_loclossL1_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.005_layerFinal/pareto_front",
        output_dir="./p_curves_demo/VOC2007/bcos/Final/L1"):
    """
    """

    # Extract the used localization loss
    pattern = re.compile(r"sll([\d.]+)")

    # Search for the pattern in the path name
    match = pattern.search(model_dir)

    if match:
        # Retrieve the numerical value after "sll"
        sll = match.group(1)

    # Initialize a pareto front tracker for the EPG vs F1 score.
    pareto_front_tracker_EPG = utils.ParetoFrontModels(
                    epg=True, iou=False, adapt_iou=False, bin_width=bin_width
                )
    
    num_model = 0
    utils.set_seed(0)

    # Loop over all pareto_chechpoints
    for pareto_ch in os.listdir(model_dir):
        full_path = os.path.join(model_dir, pareto_ch)

        # VOC num of classes
        num_classes = 20

        #Load bcos model
        model = hubconf.resnet50(pretrained=True)
        model[0].fc = bcos.modules.bcosconv2d.BcosConv2d(
            in_channels=model[0].fc.in_channels,
            out_channels=num_classes,
        )

        layer_dict = {
            "Input": None,
            "Mid1": 3,
            "Mid2": 4,
            "Mid3": 5,
            "Final": 6,
        }

        # Get layer to extract atribution layers
        layer_idx = layer_dict[layer]

        # Load model checkpoint
        checkpoint = torch.load(full_path)
        model.load_state_dict(checkpoint["model"])
        model = model.cuda()

        # Add transform for BCOS model
        transformer = bcos.data.transforms.AddInverse(dim=0)

        # Load dataset base on --split argument
        data_path = "datasets/VOC2007/"
        root = os.path.join(
            data_path, "processed"
        )
        if data_split == "train":
            train_data = datasets.VOCDetectParsed(
                root=root,
                image_set="train",
                transform=transformer,
                annotated_fraction=1.0,
            )
            loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=eval_batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=datasets.VOCDetectParsed.collate_fn,
            )
            num_batches = len(train_data) / eval_batch_size
        elif data_split == "val":
            val_data = datasets.VOCDetectParsed(
                root=root, image_set="val", transform=transformer
            )
            loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=datasets.VOCDetectParsed.collate_fn,
            )
            num_batches = len(val_data) / eval_batch_size
        elif data_split == "test":
            test_data = datasets.VOCDetectParsed(
                root=root, image_set="test", transform=transformer
            )
            loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=datasets.VOCDetectParsed.collate_fn,
            )
            num_batches = len(test_data) / eval_batch_size
        else:
            raise Exception(
                "Data split not valid choose from ['train', 'val', 'test']"
            )
        
        # Use BCE classification loss
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Attribution method
        model_activator = model_activators.ResNetModelActivator(
            model=model, layer=layer_idx, is_bcos=True
        )

        # Create attribution map
        interpolate = True if layer_idx is not None else False
        eval_attributor = attribution_methods.get_attributor(
            model,
            attribution_method,
            False,  # loss_loc.only_positive,
            False,  # loss_loc.binarize,
            interpolate,
            (224, 224),
            batch_mode=False,
        )

        # Evaluate model
        metric_vals = eval_model(
            model_activator,
            eval_attributor,
            loader,
            num_batches,
            num_classes,
            loss_fn,
            None,  # writer,
            1,
            mode="bbs",
            vis_flag=False,
        )

        # Save evaluation of pareto checkpoint in tracker
        pareto_front_tracker_EPG.update(
            model, metric_vals, num_model, sll
        )

    os.makedirs(output_dir, exist_ok=True)
    # Create and save pareto front out of all evaluated checkpoint in the tracker
    pareto_front_tracker_EPG.save_pareto_front(output_dir, npz=True)

if __name__ == "__main__":
    main()