import torch
import torchvision
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

target_dict = {'background': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7,
                       'cat': 8, 'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                       'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20}
target_classes = list(target_dict.keys())
color_classes = [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]


def encode_segmap(mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        # https://github.com/meetps/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        unique_labels = torch.zeros(20, dtype=int)
        for idx, label in enumerate(color_classes):
            indexes = np.where(np.all(mask == label, axis=-1))[:2]
            if list(indexes[0]) and idx != 0: # if no labels in this class
                unique_labels[idx-1] = 1
            label_mask[indexes] = idx

        label_mask = torch.tensor(label_mask.astype(int))
        return label_mask, unique_labels


def main(args):
    split = args.split
        
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(
        size=(224, 224)), torchvision.transforms.ToTensor()])
    
    # to be able to evaluate on all images profided in VOC 
    if split == "all":
        train = torchvision.datasets.VOCSegmentation(
                root=args.data_root, year="2007", download=True, 
                image_set="train", transform=transform)
        val =  torchvision.datasets.VOCSegmentation(
                root=args.data_root, year="2007", download=True, 
                image_set="val", transform=transform)
        test = torchvision.datasets.VOCSegmentation(
                root=args.data_root, year="2007", download=True, 
                image_set="test", transform=transform)
        
        data = torch.utils.data.ConcatDataset([train, val, test])
    else:
        data = torchvision.datasets.VOCSegmentation(
                root=args.data_root, year="2007", download=True, 
                image_set=split, transform=transform)
    
    # # show segmantation over original image
    # non_black_pixels = (np.asarray(img_rgba)[:, :, :3] > 0).any(axis=2)
    # copy_img = data[curr_idx][0].permute(1, 2, 0).numpy().copy()
    # copy_img[non_black_pixels] = np.asarray(img_rgba)[non_black_pixels]
    
    # plt.imshow(copy_img)
    # plt.savefig("overlay")
    
    images = torch.zeros(len(data), 3, 224, 224)
    label_masks = torch.zeros(len(data), 1, 224, 224)
    labels = torch.zeros((len(data), 20))
    for i, (image, segmentation_info) in tqdm(enumerate(data), total=len(data)):
        segmentation = np.asarray(segmentation_info.resize((224, 224)).convert(mode='RGB'))
        label_mask, unique_labels = encode_segmap(segmentation)
        
        images[i] = image
        label_masks[i] = label_mask
        labels[i] = unique_labels
    
    dataset = {"data": images, "labels": labels, "mask": label_masks}
    
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(dataset, os.path.join(args.save_path, split + "_segment.pt"))
    
    

    
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default=".")
parser.add_argument("--split", type=str,
                    choices=["train", "val", "test", "trainval", "all"], required=True)
parser.add_argument("--save_path", type=str, default="processed/")
args = parser.parse_args()
main(args)
