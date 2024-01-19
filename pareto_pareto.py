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
from old_eval import eval_model

def main(args):
    pareto_front_tracker = utils.ParetoFrontModels()

    root_dir = "./FT/ptest/VOC2007/"

    num_model = 0

    #TODO WEG
    test_counter = 0

    # Loop over directories of fine tuned models for different lambda's
    for model_dir in os.listdir(root_dir):
        #TODO WEG
        if test_counter == 1:
            break

        model_path = os.path.join(root_dir, model_dir)

        #TODO WEG
        test_counter += 1

        # Look in model dir for the pareto_front map
        for pareto_dir in os.listdir(model_path):
            if pareto_dir != "pareto_front":
                continue
            pareto_path = os.path.join(model_path, pareto_dir)

            # Loop over all pareto_ch as resulted from training and evaluated on the val set
            for pareto_ch in os.listdir(pareto_path):
                full_path = os.path.join(pareto_path, pareto_ch)
                print(full_path)
                
                model_backbone, localization_loss_fn, layer, attribution_method = utils.get_model_specs(full_path)

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
                checkpoint = torch.load(full_path)
                model.load_state_dict(checkpoint["model"])
                model = model.cuda()

                # Add transform for BCOS model else normalize
                if is_bcos:
                    transformer = bcos.data.transforms.AddInverse(dim=0)
                else:
                    transformer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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
                    raise Exception("Data split not valid choose from ['train', 'val', 'test']")

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
                        False, #loss_loc.only_positive,
                        False, #loss_loc.binarize,
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
                    None, #writer,
                    1,
                    None, #args,
                )

                # Find the used learning rate
                if "sll0.001" in model_path:
                    sll = "0.001"
                if "sll0.005" in model_path:
                    sll = "0.005"
                if "sll0.0005" in model_path:
                    sll = "0.0005"
                
                pareto_front_tracker.update(model, metric_vals, num_model, sll)
                num_model += 1

    save_path = os.path.join(args.save_path, args.dataset, "TEST")
    os.makedirs(save_path, exist_ok=True)

    pareto_front_tracker.save_pareto_front(save_path, npz=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="datasets/", help="Path to datasets."
    )
    parser.add_argument(
        "--log_path", type=str, default=None, help="Path to save TensorBoard logs."
    )
    parser.add_argument(
        "--save_path", type=str, default="ptest7", help="Path to save the pareto front at."
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed to use.")
    args = parser.parse_args()
    main(args)