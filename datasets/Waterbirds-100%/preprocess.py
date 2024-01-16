import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import argparse

# https://github.com/spetryk/GALS/blob/main/datasets/waterbirds.py

def preprocess_waterbirds(args):
    cub_dataset_root = args.cub_dataset_root
    waterbirds_dataset_root = args.waterbirds_dataset_root
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    metadata_df = pd.read_csv(os.path.join(waterbirds_dataset_root, 'metadata.csv'))
    bbox_df = pd.read_csv(os.path.join(cub_dataset_root, 'bounding_boxes.txt'), sep=' ', header=None, names=['img_id', 'x', 'y', 'width', 'height'])

    split_dict = {'train': 0, 'val': 1, 'test': 2}
    split_mask = metadata_df['split'] == split_dict[args.split]

    images = []
    labels = []
    bboxes = []

    for _, row in tqdm(metadata_df[split_mask].iterrows(), total=sum(split_mask)):
        img_path = os.path.join(waterbirds_dataset_root, row['img_filename'])
        img = Image.open(img_path).convert('RGB')
        original_size = img.size
        img = transform(img)
        images.append(img)

        labels.append(row['y'])

        if row['img_id'] not in bbox_df['img_id'].values:
            print(f"Warning: img_id {row['img_id']} not found in bbox_df")
            continue

        bbox_row = bbox_df[bbox_df['img_id'] == row['img_id']].iloc[0]

        bbox = [bbox_row['x'], bbox_row['y'], bbox_row['width'], bbox_row['height']]
        bbox_scaled = scale_bbox(bbox, original_size)
        bboxes.append(bbox_scaled)

        # # visualize the image with bounding box
        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # fig, ax = plt.subplots(1)
        # ax.imshow(img.permute(1, 2, 0))
        # rect = patches.Rectangle((bbox_scaled[0], bbox_scaled[1]), bbox_scaled[2], bbox_scaled[3], linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # plt.show()

        # # Updated code to ensure the figure is displayed in the popup before it closes after 2 seconds
        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # import time

        # fig, ax = plt.subplots(1)
        # ax.imshow(img.permute(1, 2, 0))
        # rect = patches.Rectangle((bbox_scaled[0], bbox_scaled[1]), bbox_scaled[2], bbox_scaled[3], linewidth=1, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        # plt.draw()  # Draw the figure before showing
        # plt.pause(2)  # Show it for 2 seconds
        # plt.close()  # Close the figure


    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.float32)
    bboxes = torch.tensor(bboxes, dtype=torch.float32)

    dataset = {"data": images, "labels": labels, "bbs": bboxes, "mask": None}

    save_path = os.path.join(args.save_path, args.split + ".pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(dataset, save_path)

def scale_bbox(bbox, img_size):
    orig_width, orig_height = img_size  # Original image size
    new_width, new_height = 224, 224
    scale_x, scale_y = new_width / orig_width, new_height / orig_height

    x, y, width, height = bbox
    x = int(x * scale_x)
    y = int(y * scale_y)
    width = int(width * scale_x)
    height = int(height * scale_y)

    return [x, y, width, height]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub_dataset_root", type=str, required=True, help="Root directory of CUB dataset")
    parser.add_argument("--waterbirds_dataset_root", type=str, required=True, help="Root directory of Waterbirds-100 dataset")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    parser.add_argument("--save_path", type=str, default="processed/")
    args = parser.parse_args()

    preprocess_waterbirds(args)
