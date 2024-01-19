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
import re
import numpy as np

from eval import evaluation_function

print("Baseline:")
evaluation_function(
    "./BASE/VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput/model_checkpoint_f1_best.pt",
    pareto=True,
    split="seg_test",
    mode="segment",
)

print("Finetune:")
evaluation_function(
    "./FT/VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerFinal/model_checkpoint_f1_best.pt",
    split="seg_test",
    mode="segment",
)

# Root directory
root_dir = "./FT/VOC2007"

# Regular expression to match and extract information from file names
pattern = re.compile(
    r'(\w+)_finetunedobjlocpareto_attr(\w+)_locloss(\w+)_origmodel_checkpoint_f1_best.pt_(\w+)_lr([0-9.e-]+)_sll([0-9.e-]+)_layer(\w+)/pareto_front/model_checkpoint_pareto_(.+).pt'
)

# Walk through the directory
for subdir, dirs, files in os.walk(root_dir):
    for file_name in tqdm(files):
        full_path = os.path.join(subdir, file_name)
        if file_name.endswith(".pt") and 'pareto_front' in full_path:
            match = pattern.search(full_path)
            if match:
                attr_type, attr_name, loss_type, _, _, _, layer_type, _ = match.groups()
                model_path = full_path
                pareto = True
                log_path = f"metrics_per_model_{attr_type.lower()}/"

                # # print all the arguments
                # print(f"attr_type: {attr_type}")
                # print(f"attr_name: {attr_name}")
                # print(f"loss_type: {loss_type}")
                # print(f"layer_type: {layer_type}")
                # print(f"pareto: {pareto}")
                # print(f"log_path: {log_path}")

                # Call the evaluation function
                evaluation_function(model_backbone=attr_type.lower(),
                                    model_path=model_path,
                                    localization_loss_fn=loss_type,
                                    layer=layer_type,
                                    attribution_method=attr_name,
                                    pareto=pareto,
                                    log_path=log_path)

# baseline BCOS
model_path = "/home/roan/Documents/FACTifAI_2024_3/BASE/VOC2007/vanilla_standard_attrNone_loclossNone_origNone_resnet50_lr1e-05_sll1.0_layerInput/model_checkpoint_f1_best.pt"

evaluation_function(model_backbone="vanilla",
                    model_path=model_path,
                    localization_loss_fn=None,
                    layer="Input",
                    attribution_method=None,
                    pareto=False,
                    baseline=True,
                    log_path="metrics_per_model_vanilla/")