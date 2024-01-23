import torch
import os
from tqdm import tqdm
import datasets
import argparse
import torch.utils.tensorboard
import statistics
import utils
import losses
import metrics
import bcos.modules
import bcos.data
import numpy as np
import csv

from eval import evaluation_function
from utils import get_model_specs, get_model, get_class_name
import matplotlib.pyplot as plt 
from collections import defaultdict

class Class_Fairness:
    def __init__(self, model_pathBase, 
                 model_pathFN50,
                 model_pathFNbest, dataset, metric, split) -> None:
        
        mode = 'segment' if split == "seg_test" else "bbs"
        if mode == "segment" and metric in ["BB-Loc-segment", "BB-Loc-Fraction"]:
            raise TypeError('Can not do segmentaion for non segmented dataset')
        
        bl_metrics, labels = evaluation_function(model_pathBase,
                                         fix_layer="Final",
                                         pareto=False,
                                         eval_batch_size=4,
                                         data_path=f"datasets/",
                                         dataset=dataset,
                                         split=split,
                                         annotated_fraction=1,
                                         log_path=None,
                                         mode=mode,
                                         npz=False,
                                         vis_iou_thr_methods=False,
                                         return_per_class=True)
        
        ft50_metrics, labels = evaluation_function(model_pathFN50,
                                         fix_layer="Final",
                                         pareto=False,
                                         eval_batch_size=4,
                                         data_path=f"datasets/",
                                         dataset=dataset,
                                         split=split,
                                         annotated_fraction=1,
                                         log_path=None,
                                         mode=mode,
                                         npz=False,
                                         vis_iou_thr_methods=False,
                                         return_per_class=True)
        
        ftbest_metrics, labels = evaluation_function(model_pathFNbest,
                                         fix_layer="Final",
                                         pareto=False,
                                         eval_batch_size=4,
                                         data_path=f"datasets/",
                                         dataset=dataset,
                                         split=split,
                                         annotated_fraction=1,
                                         log_path=None,
                                         mode=mode,
                                         npz=False,
                                         vis_iou_thr_methods=False,
                                         return_per_class=True)
        
        bl_means = self.compute_metric(bl_metrics[metric], labels)
        ft50_means = self.compute_metric(ft50_metrics[metric], labels)
        ftbest_means = self.compute_metric(ftbest_metrics[metric], labels)
        
        percentage50_diff = [round((ft - bl) / bl, 4) for bl, ft in zip(bl_means, ft50_means)]
        percentagebest_diff = [round((ft - bl) / bl, 4) for bl, ft in zip(bl_means, ftbest_means)]
        
        class_names = [get_class_name(i) for i in range(20)]
        
        bl_means= [round(value, 4) for value in bl_means]
        ft50_means = [round(value, 4) for value in ft50_means]
        ftbest_means = [round(value, 4) for value in ftbest_means]
        
        label_counts = defaultdict(int)

        for label in labels:
            label_counts[label.item()] += 1
            
        sample_count = list(label_counts.values())
        print(sample_count)
        
        # Zip the data together
        data = zip(class_names, sample_count, bl_means, ft50_means, ftbest_means,  percentage50_diff, percentagebest_diff)

        # Save to CSV file
        csv_file_path = f'{dataset}_{metric}_{split}.csv'
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header
            writer.writerow(['class-name', 'num-samples', 'bl-means', 'ft50-means', 'ftbest-means', 'percentage50_diff', 'percentagebest_diff'])
            
            # Write data
            for row in data:
                writer.writerow(row)
        
                
    def compute_metric(self, data, labels):
        """
        Compute the mean score per class for a given metric.

        Args:
            data (list): List of metric values corresponding to each data point.
            labels (list): List of class labels corresponding to each data point.

        Returns:
            list: Mean score per class.
        """
        metric_score = [[] for _ in range(20)]

        # Group metric values by class
        _ = [metric_score[label].append(data[i]) for i, label in enumerate(labels)]

        # Compute the mean score for each class
        mean_scores = [statistics.fmean(sublist) for sublist in metric_score]

        return mean_scores


def main(args):
    model_path1 = args.model_pathBase
    model_path2 = args.model_pathFN50
    model_path3 = args.model_pathFNbest
    dataset = args.dataset
    metric = args.metric
    split = args.split
    Class_Fairness(model_pathBase=model_path1, 
                 model_pathFN50=model_path2,
                 model_pathFNbest=model_path3, 
                 dataset=dataset, 
                 metric=metric,
                 split=split)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_pathBase",
        type=str,
        default=None,
        help="Path to checkpoint to eval for a baseline model",
        required=True,
    )
    parser.add_argument(
        "--model_pathFN50",
        type=str,
        default=None,
        help="Path to checkpoint to eval for a finetuned model",
        required=True,
    )
    parser.add_argument(
        "--model_pathFNbest",
        type=str,
        default=None,
        help="Path to checkpoint to eval for a finetuned model",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="VOC2007",
        help="dataset to choose",
        choices=[
            "VOC2007",
            "COCO2014",
        ],
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="BB-Loc"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="seg_test",
        choices=["train", "val", "test", "seg_test"],
        help="Set to evaluate on",
    )
    
    args = parser.parse_args()
    main(args)
    
    # python fairness.py --model_pathBase ./BASE/VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.001_sll1.0_layerInput/model_checkpoint_f1_best.pt --model_pathFN50 ./FT/VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerFinal/model_checkpoint_f1_best.pt  --model_pathFNbest ./FT/VOC2007/bcos_finetunedobjlocpareto_attrBCos_loclossEnergy_origmodel_checkpoint_f1_best.pt_resnet50_lr0.0001_sll0.001_layerFinal/model_checkpoint_f1_best.pt --dataset VOC2007
    