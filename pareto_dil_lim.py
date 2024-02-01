"""
Reproducibility Study of “Studying How to Efficiently and Effectively Guide Models with Explanations”

Description: This file is part of a project aiming to reproduce the study titled "Studying How to Efficiently and 
Effectively Guide Models with Explanations." The project focuses on verifying the results and methodologies 
proposed in the original study, and potentially extending or refining the study's findings.

Based on the code of orginal paper: https://github.com/sukrutrao/Model-Guidance

pareto_dil_lim.py
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
from eval import eval_model
import re
import numpy as np


def main(args):
    bin_width = 0.002
    root_dir = args.pareto_path

    for loss in os.listdir(root_dir):
        loss_path = os.path.join(root_dir, loss)

        for eff_metric in os.listdir(loss_path):
            eff_path = os.path.join(loss_path, eff_metric)

            print("######### Dilation/Annotation Frac ##########")
            print(eff_metric)
            print("##################################################")

            pareto_front_tracker_EPG = utils.ParetoFrontModels(
                epg=True, iou=False, adapt_iou=False, bin_width=bin_width
            )
            pareto_front_tracker_IOU = utils.ParetoFrontModels(
                epg=False, iou=True, adapt_iou=False, bin_width=bin_width
            )
            pareto_front_tracker_ADAPTIOU = utils.ParetoFrontModels(
                epg=False, iou=False, adapt_iou=True, bin_width=bin_width
            )

            output_dir = ""
            num_model = 0

            # Loop over directories of fine tuned models for different lambda's of a specific dilation
            for model_dir in os.listdir(eff_path):
                model_path = os.path.join(eff_path, model_dir)
                print("----------")
                print(model_dir)
                print("----------")

                # Extract the used localization loss
                pattern = re.compile(r"sll([\d.]+)")
                # Search for the pattern in the path name
                match = pattern.search(model_dir)
                if match:
                    # Retrieve the numerical value after "sll"
                    sll = match.group(1)

                # Extract the used dilation
                dil = model_dir.split("_")[-1]

                # Look in model dir for the pareto_front map
                for pareto_dir in os.listdir(model_path):
                    if pareto_dir != "pareto_front":
                        continue
                    pareto_path = os.path.join(model_path, pareto_dir)

                    # Loop over all pareto_ch as resulted from training and evaluated on the val set
                    for pareto_ch in os.listdir(pareto_path):

                        full_path = os.path.join(pareto_path, pareto_ch)
                        print("@@@@ Evaluating individual pareto dom model @@@@")
                        print(full_path)
                        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

                        (
                            model_backbone,
                            localization_loss_fn,
                            layer,
                            attribution_method,
                        ) = utils.get_model_specs(full_path)
                        output_dir = os.path.join(
                            model_backbone, layer, localization_loss_fn, dil
                        )

                        utils.set_seed(args.seed)

                        # Get number of classes
                        num_classes_dict = {"VOC2007": 20, "COCO2014": 80}
                        num_classes = num_classes_dict[args.dataset]

                        # Load correct model
                        is_bcos = model_backbone == "bcos"
                        is_xdnn = model_backbone == "xdnn"
                        is_vanilla = model_backbone == "vanilla"

                        if is_bcos:
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
                        elif is_xdnn:
                            model = fixup_resnet.xfixup_resnet50()
                            imagenet_checkpoint = torch.load(
                                os.path.join(
                                    "weights/xdnn/xfixup_resnet50_model_best.pth.tar"
                                )
                            )
                            imagenet_state_dict = utils.remove_module(
                                imagenet_checkpoint["state_dict"]
                            )
                            model.load_state_dict(imagenet_state_dict)
                            model.fc = torch.nn.Linear(
                                in_features=model.fc.in_features,
                                out_features=num_classes,
                            )
                            layer_dict = {
                                "Input": None,
                                "Mid1": 3,
                                "Mid2": 4,
                                "Mid3": 5,
                                "Final": 6,
                            }
                        elif is_vanilla:
                            model = torchvision.models.resnet50(
                                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
                            )
                            model.fc = torch.nn.Linear(
                                in_features=model.fc.in_features,
                                out_features=num_classes,
                            )
                            layer_dict = {
                                "Input": None,
                                "Mid1": 4,
                                "Mid2": 5,
                                "Mid3": 6,
                                "Final": 7,
                            }
                        else:
                            raise NotImplementedError

                        # Get layer to extract atribution layers
                        layer_idx = layer_dict[layer]

                        # Load model checkpoint
                        checkpoint = torch.load(full_path)
                        model.load_state_dict(checkpoint["model"])
                        model = model.cuda()

                        # Add transform for BCOS model else normalize
                        if is_bcos:
                            transformer = bcos.data.transforms.AddInverse(dim=0)
                        else:
                            transformer = torchvision.transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                            )

                        # Load dataset base on --split argument
                        root = os.path.join(args.data_path, args.dataset, "processed")

                        if args.split == "train":
                            train_data = datasets.VOCDetectParsed(
                                root=root,
                                image_set="train",
                                transform=transformer,
                                annotated_fraction=args.annotated_fraction,
                            )
                            loader = torch.utils.data.DataLoader(
                                train_data,
                                batch_size=args.eval_batch_size,
                                shuffle=True,
                                num_workers=0,
                                collate_fn=datasets.VOCDetectParsed.collate_fn,
                            )
                            num_batches = len(train_data) / args.eval_batch_size
                        elif args.split == "val":
                            val_data = datasets.VOCDetectParsed(
                                root=root, image_set="val", transform=transformer
                            )
                            loader = torch.utils.data.DataLoader(
                                val_data,
                                batch_size=args.eval_batch_size,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=datasets.VOCDetectParsed.collate_fn,
                            )
                            num_batches = len(val_data) / args.eval_batch_size
                        elif args.split == "test":
                            test_data = datasets.VOCDetectParsed(
                                root=root, image_set="test", transform=transformer
                            )
                            loader = torch.utils.data.DataLoader(
                                test_data,
                                batch_size=args.eval_batch_size,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=datasets.VOCDetectParsed.collate_fn,
                            )
                            num_batches = len(test_data) / args.eval_batch_size
                        else:
                            raise Exception(
                                "Data split not valid choose from ['train', 'val', 'test']"
                            )

                        # Get loss function to calculate loss of split
                        loss_fn = torch.nn.BCEWithLogitsLoss()

                        # Get model activator to procces batches
                        model_activator = model_activators.ResNetModelActivator(
                            model=model, layer=layer_idx, is_bcos=is_bcos
                        )

                        # If needed get atribution method to calculate atribution maps
                        if attribution_method:
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
                            None,  # writer,
                            1,
                            mode="bbs",
                            vis_flag=False,
                        )

                        pareto_front_tracker_EPG.update(
                            model, metric_vals, num_model, sll
                        )
                        pareto_front_tracker_IOU.update(
                            model, metric_vals, num_model, sll
                        )

                        pareto_front_tracker_ADAPTIOU.update(
                            model, metric_vals, num_model, sll
                        )
                        num_model += 1

                        if args.dilated:
                            # save all metrics of evaluated models on the test set as npz
                            save_all_path = os.path.join(
                                args.save_path, output_dir, "not_par"
                            )
                            os.makedirs(save_all_path, exist_ok=True)

                            # Save as npz
                            npz_name = f"EPG_SLL{sll}_F{metric_vals['F-Score']:.4f}_EPG{metric_vals['BB-Loc']:.4f}_IOU{metric_vals['BB-IoU']:.4f}"
                            npz_path = os.path.join(save_all_path, npz_name)
                            np.savez(
                                npz_path,
                                f_score=metric_vals["F-Score"],
                                bb_score=metric_vals["BB-Loc"],
                                iou_score=metric_vals["BB-IoU"],
                                adapt_iou_score=metric_vals["BB-IoU-Adapt"],
                                sll=sll,
                            )

                            npz_name = f"ADAPT_SLL{sll}_F{metric_vals['F-Score']:.4f}_EPG{metric_vals['BB-Loc']:.4f}_IOU{metric_vals['BB-IoU-Adapt']:.4f}"
                            npz_path = os.path.join(save_all_path, npz_name)
                            np.savez(
                                npz_path,
                                f_score=metric_vals["F-Score"],
                                bb_score=metric_vals["BB-Loc"],
                                iou_score=metric_vals["BB-IoU"],
                                adapt_iou_score=metric_vals["BB-IoU-Adapt"],
                                sll=sll,
                            )

            # # Create and save the pareto front of the resulting evaluated models
            if args.dilated:
                save_path = os.path.join(args.save_path, output_dir, "yes_par")
            else:
                save_path = os.path.join(args.save_path, output_dir)
            os.makedirs(save_path, exist_ok=True)

            pareto_front_tracker_EPG.save_pareto_front(save_path, npz=True)
            pareto_front_tracker_IOU.save_pareto_front(save_path, npz=True)
            pareto_front_tracker_ADAPTIOU.save_pareto_front(save_path, npz=True)

            if args.dilated:
                # remove pareto front models from "not_par"
                yes_pareto_path = save_path
                not_pareto_path = os.path.join(args.save_path, output_dir, "not_par")

                for par_model in os.listdir(yes_pareto_path):
                    if par_model == "pareto_front":
                        continue

                    par_model_no_suffix = par_model.replace("_" + par_model.split("_")[-1], "")
                    print(par_model_no_suffix)
                    not_pareto_models_list = os.listdir(not_pareto_path)

                    for not_par_model in not_pareto_models_list:
                        # Delete pareto front models from not_par directory
                        if not_par_model.startswith(par_model_no_suffix):
                            print("------------------")
                            print("--- Pareto CH: ---")
                            print(par_model)
                            print("--- Delete CH: ---")
                            print(not_par_model)
                            print("------------------")
                            check_point_npz = os.path.join(not_pareto_path, not_par_model)
                            if os.path.exists(check_point_npz):
                                os.remove(check_point_npz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pareto_path", 
        type=str, 
        default="./FT/ANN", 
        help="Path to the pareto front models main directory."
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="datasets/", 
        help="Path to datasets."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="p_curves/Test/",
        help="Path to save the pareto front at.",
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
        choices=["train", "val", "test"],
        help="Set to evaluate on",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size to use for evaluation.",
    )
    parser.add_argument(
        "--dilated",
        type=bool, 
        default=False, 
        help="Specify Whether you fine-tuned with dilated bounding boxes."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0, 
        help="Random seed to use."
    )
    args = parser.parse_args()
    main(args)
