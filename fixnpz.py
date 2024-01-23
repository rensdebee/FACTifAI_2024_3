import torch
import numpy as np
import os


def save_pareto_front(load_path, save_path, pt_name, npz=False):
    pareto_checkpoints = torch.load(load_path)
    f_score = pareto_checkpoints["F-Score"]
    bb_score = pareto_checkpoints["BB-Loc"]
    iou_score = pareto_checkpoints["BB-IoU"]
    adapt_iou_score = pareto_checkpoints["BB-IoU-Adapt"]
    epoch = pareto_checkpoints["epochs"]
    sll = pareto_checkpoints["sll"]
    method = pt_name.split("_")[3]
    print(method)
    if npz:
        # Save as npz
        if method == "ADAPT":
            npz_name = f"{method}_SLL{sll}_F{f_score:.4f}_EPG{bb_score:.4f}_IOU{adapt_iou_score:.4f}_{epoch}"
        else:
            npz_name = f"{method}_SLL{sll}_F{f_score:.4f}_EPG{bb_score:.4f}_IOU{iou_score:.4f}_{epoch}"
        npz_path = os.path.join(save_path, npz_name)
        np.savez(
            npz_path,
            f_score=f_score,
            bb_score=bb_score,
            iou_score=iou_score,
            adapt_iou_score=adapt_iou_score,
            sll=sll,
        )


# save_pareto_front(path, npz=False)

root_dir = "./p_curves_rens/VOC2007/"
for base in os.listdir(root_dir):
    base_path = os.path.join(root_dir, base)
    for layer in os.listdir(base_path):
        layer_path = os.path.join(base_path, layer)
        for loss in os.listdir(layer_path):
            loss_path = os.path.join(layer_path, loss)
            for model_dir in os.listdir(loss_path):
                if not model_dir.endswith(".npz"):
                    model_path = os.path.join(loss_path, model_dir)
                    for pareto_dir in os.listdir(model_path):
                        pareto_path = os.path.join(model_path, pareto_dir)
                        print("----------")
                        print(pareto_path, loss_path)
                        save_pareto_front(pareto_path, loss_path, pareto_dir, npz=True)
                        print("----------")
