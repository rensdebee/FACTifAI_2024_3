<p align="center">
<h1 align="center">
Reproducing: Studying How to Efficiently and Effectively Guide Models with Explanations
</h1>
 
## Setup

### Prerequisites

All the required packages can be installed using conda with the provided [environment.yml](environment.yml) file.

### Data

Scripts to download and preprocess the VOC2007 and COCO2014 datasets have been provided in the [datasets](datasets) directory. Please refer to the README file provided there.

### ImageNet Pre-trained Weights

A script to download the pre-trained ImageNet weights for B-cos and X-DNN backbones has been provided in the [weights](weights) directory. Please refer to the README file provided there.


## Training Models

To train a model, use:

```bash
python train.py [options]
```

The list of options and their descriptions can be found by using:

```bash
python train.py -h
```

### Training without Model Guidance

For example, to train a B-cos model on VOC2007, use:

```bash
python train.py --model_backbone bcos --dataset VOC2007 --learning_rate 1e-4 --train_batch_size 64 --total_epochs 300
```

### Fine-tuning with Model Guidance

For example, to optimize B-cos attributions using the Energy loss at the Input layer, use:

```bash
python train.py --model_backbone bcos --dataset VOC2007 --learning_rate 1e-4 --train_batch_size 64 --total_epochs 50 --optimize_explanations --model_path models/VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr1e-04_sll1.0_layerInput/model_checkpoint_f1_best.pt --localization_loss_lambda 1e-3 --layer Input --localization_loss_fn Energy --pareto
```

### Evaluating and displaying different IoU thresholds

For example, to optimize B-cos attributions using the Energy loss at the Input layer, use:

```bash
python eval.py --model_path BASE/VOC2007/bcos_standard_attrNone_loclossNone_origNone_resnet50_lr0.001_sll1.0_layerInput/model_checkpoint_final_300.pt --log_path ./base_logs/VOC2007/EVAL/ --dataset VOC2007 --fix_layer Input --vis_iou_thr_methods
```


---
## Acknowledgements

This repository uses and builds upon code from the following repositories:
* [B-cos/B-cos-v2](https://github.com/B-cos/B-cos-v2)
* [stevenstalder/NN-Explainer](https://github.com/stevenstalder/NN-Explainer)
* [visinf/fast-axiomatic-attribution](https://github.com/visinf/fast-axiomatic-attribution)
* [sukrutrao/Model-Guidance](https://github.com/sukrutrao/Model-Guidance)



