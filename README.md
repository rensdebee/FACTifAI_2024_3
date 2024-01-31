# Reproducibility Study of “Studying How to Efficiently and Effectively Guide Models with Explanations”

This repository contains the code for a project aiming to reproduce the study titled (["Studying How to Efficiently and 
Effectively Guide Models with Explanations."](https://openaccess.thecvf.com/content/ICCV2023/papers/Rao_Studying_How_to_Efficiently_and_Effectively_Guide_Models_with_Explanations_ICCV_2023_paper.pdf)). If you want to use this code, please cite the orginal paper:

```
@inproceedings{rao2023studying,
  title={Studying How to Efficiently and Effectively Guide Models with Explanations},
  author={Rao, Sukrut and B{\"o}hle, Moritz and Parchami-Araghi, Amin and Schiele, Bernt},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1922--1933},
  year={2023}
}
```
 
## Table of Contents
* [About the Project](#about-the-project)
* [Folder Structure](#folder-structure)
* [Getting Started](#getting-started)
  * [Install Packages](#install-packages)
  * [Download the Data](#download-the-data)
  * [ImageNet Pre-trained Weights](#imagenet-pre-trained-weights)
* [Training models](#training-models)
  * [Training without Model Guidance](#training-without-model-guidance)
  * [Fine-tuning with Model Guidance](#fine-tuning-with-model-guidance)
  * [Evaluating and displaying different IoU thresholds](#evaluating-and-displaying-different-iou-thresholds)
* [Acknowledgements](#acknowledgements)

## About the Project

While deep neural networks have excelled in diverse research domains, there is no assurance that these models are learning the correct features. Certain models may rely on spurious correlations in their predictions, such as putting undue attention on the background. This leads to unfair and inexplicable decisions and consequently limits a models ability to generalize. To inspect if models rely on spurious correlations in their decision-making, attribution methods have been developed. When incorporated alongside a classification model, these methods can steer the model's attention toward relevant features, ensure that the model is right for the right reasons.

This study aims to replicate the original paper, and investigates the reproducibility of the main claims in the paper

![Intro Teaser](images/Figure2_Best.png)

## Folder Structure
```
├── README.md
├── environment.yml                                   - environment to run the code
├── datasets
│   ├── VOC2007                                       - will hold VOC-2007 data (automatically downloaded)
│   │   └── preprocess.py                             - download file for VOC-2007 data (will create new directories)
│   └── WATERBIRDS                                    - will hold Waterbirds-100% data (automatically downloaded)
│       └── preprocess.py                             - download file for Waterbirds-100% data (will create new directories)
├── examples                                          - example images for this README
├── BASE                                              - will hold the trained baseline models
│   ├── VOC2007
│   └── WATERBIRDS 
├── base_logs                                         - will hold the tensorboard logs for the baseline models
├── FT                                                - will hold the trained fine-tuned models
│   ├── VOC2007
│   └── WATERBIRDS
├── ft_logs                                           - will hold the tensorboard logs for the fine-tuned models
├── images                                            - will hold the images for this reproducibility study
├── p_c_ann                                           - will hold the .npz files for the paretto front of the models trained with the coarse annotations
├── p_c_dil                                           - will hold the .npz files for the paretto front of the models trained with the dilated annotations
├── p_curves                                          - will hold the .npz files for the paretto front of the models trained on different datasets
├── p_curves_bin0.002                                 - will hold the .npz files for the paretto front of the models trained on different datasets with a bin size of 0.002
├── weights                                           - will hold a script to download the pre-trained ImageNet weights
├── attribution_methods.py                            - defines the attribution methods
├── datasets.py                                       - defines the datasets
├── eval.py                                           - will hold the code to evaluate the models
├── fairness.py                                       - will hold the code to evaluate the fairness of the model on different classes
├── fixnpz.py                                         - will hold the code to fix the .npz files
├── fixup_resnet.py                                   - defines the fixup_resnet model
├── hubconf.py                                        - defines the configs for the models
├── losses.py                                         - defines the loss functions used for training
├── metrics.py                                        - defines metrics to be used during training and testing
├── model_activators.py                               - defines the ResNet model activators
├── pareto_dil_lim.py                                 - will hold the code to compute the paretto front of the models trained with the dilated annotations
├── pareto_FT.py                                      - will hold the code to compute the paretto front of the models trained on different datasets
├── tensorboard_proccessing.py                        - will hold the code to process the tensorboard logs
├── test_transform.py                                 - will hold the code to transform the test images
├── train.py                                          - main script to train the models
├── utils.py                                          - defines utility functions
└── visualize.py                                      - defines functions to visualize the results
```

## Getting Started
### Install Packages

All required packages can be found in the environment.yml file. They are most easily installed with [conda/miniconda](https://docs.conda.io/en/latest/miniconda.html), where a new environment can be easily created like this: 
```bash
conda env create -f environment.yml 
```
After that, activate the new environment (needs to be done everytime the shell is reloaded):
```bash
conda activate repro_model_guidance
```
Should you get any errors during the environment setup, make sure to set
```bash
conda config --set channel_priority false
```
and then try again.

### Download the Data

The Pascal VOC-2007 training, validation and training sets should be automatically downloaded to [`datasets/VOC2007/`](datasets/VOC2007) after you run the `preprocess.py` script in the corresponding directory. If that does not work, the training and validation data is available [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar) and the testing data is available [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar). 

The Waterbirds-100% dataset should be automatically downloaded to [`datasets/WATERBIRDS/`](datasets/WATERBIRDS) after you run the `preprocess.py` script in the corresponding directory. If that does not work, the Waterbirds-100% dataset is available [here](https://drive.google.com/file/d/1zJpQYGEt1SuwitlNfE06TFyLaWX-st1k/view) and the Caltech-UCSD Birds-200-2011 dataset, which contains the bounding_boxes.txt file, is available [here](https://www.vision.caltech.edu/datasets/cub_200_2011/).

For either dataset, the `dataset` folder is set as the default base directory in the code. If you move (or want to download) the datasets to a different place, make sure to point the code to the new directory as follows (additional arguments omitted, for more information on how to run the code, see the [Training and Testing the Explainer](#training-and-testing-the-explainer) section:
```bash
python train.py --data_path="<path to your directory containing the VOC2007 or WATERBIRDS subdirectories>"
```
The `--data_base_path` can either be an absolute path or a relative path from the `src` directory (which contains the `main.py` script).

To download the training, validation and training sets, run the following command:
```bash
python preprocess.py --split train
python preprocess.py --split val
python preprocess.py --split test
```

### ImageNet Pre-trained Weights

A script to download the pre-trained ImageNet weights for B-cos and X-DNN backbones has been provided in the [weights](weights) directory. Store ImageNet pre-trained weights for X-DNN and B-cos models there. To download them, run [weights/download.sh](weigths/download.sh)

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



