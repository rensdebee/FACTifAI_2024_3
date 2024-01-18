import torch
import torchvision
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import xml.etree.ElementTree as ET


target_dict = {
    "background": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}
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
        if list(indexes[0]) and idx != 0:  # if no labels in this class
            unique_labels[idx - 1] = 1
        label_mask[indexes] = idx

    label_mask = torch.tensor(label_mask.astype(int))
    return label_mask, unique_labels

# def get_bounding_boxes(label_mask, unique_labels):
#     indxs = torch.where(unique_labels == 1)[0]
#     bbxs = []
#     for label in indxs:
#         indices = torch.where(label_mask == label + 1)
#         min_x = torch.min(indices[1])
#         max_x = torch.max(indices[1])
#         min_y = torch.min(indices[0])
#         max_y = torch.max(indices[0])
#         bbxs.append([label, min_x, min_y, max_x, max_y])
#     return bbxs

# def parse_xml_annotation(xml_file_path):
#     tree = ET.parse(xml_file_path)
#     root = tree.getroot()

#     bounding_boxes = []
#     for obj in root.findall('object'):
#         class_name = obj.find('name').text
#         class_idx = target_dict[class_name]
#         bbox = obj.find('bndbox')
#         xmin = int(bbox.find('xmin').text)
#         ymin = int(bbox.find('ymin').text)
#         xmax = int(bbox.find('xmax').text)
#         ymax = int(bbox.find('ymax').text)
        
#         annotation = [class_idx, xmin, ymin, xmax, ymax]
#         bounding_boxes.append(annotation)


#     return bounding_boxes

def parse_xml_annotation(xml_file_path, target_size=224):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    rescaled_annotations = []
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    wscale = target_size / image_width
    hscale = target_size / image_height

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_idx = target_dict[class_name]
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        new_xmin = int(min(max(xmin * wscale, 0), target_size - 1))
        new_xmax = int(min(max(xmax * wscale, 0), target_size - 1))
        new_ymin = int(min(max(ymin * hscale, 0), target_size - 1))
        new_ymax = int(min(max(ymax * hscale, 0), target_size - 1))

        rescaled_annotation = [class_idx - 1, new_xmin, new_ymin, new_xmax, new_ymax]
        rescaled_annotations.append(rescaled_annotation)

    return rescaled_annotations


def main(args):
    split = args.split

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )

    # to be able to evaluate on all images profided in VOC
    if split == "all":
        val = torchvision.datasets.VOCSegmentation(
            root=args.data_root,
            year="2007",
            download=True,
            image_set="val",
            transform=transform,
        )
        test = torchvision.datasets.VOCSegmentation(
            root=args.data_root,
            year="2007",
            download=True,
            image_set="test",
            transform=transform,
        )
        image_files = []
        with open("./VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt", "r") as f:
            for line in f:
                image_files.append(line.strip() + ".xml")
        with open("./VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt", "r") as f:
            for line in f:
                image_files.append(line.strip() + ".xml")

        data = torch.utils.data.ConcatDataset([val, test])
    else:
        data = torchvision.datasets.VOCSegmentation(
            root=args.data_root,
            year="2007",
            download=True,
            image_set=split,
            transform=transform,
        )

    # # show segmantation over original image
    # non_black_pixels = (np.asarray(img_rgba)[:, :, :3] > 0).any(axis=2)
    # copy_img = data[curr_idx][0].permute(1, 2, 0).numpy().copy()
    # copy_img[non_black_pixels] = np.asarray(img_rgba)[non_black_pixels]

    # plt.imshow(copy_img)
    # plt.savefig("overlay")

    images = torch.zeros(len(data), 3, 224, 224)
    label_masks = torch.zeros(len(data), 1, 224, 224)
    labels = torch.zeros((len(data), 20))
    bounding_boxes = [[] for _ in range(len(data))]
    path_root = "./VOCdevkit/VOC2007/Annotations/"
    
    for i, (d, image_file) in enumerate(zip(tqdm(data), image_files)):
        image, segmentation_info = d
        segmentation = np.asarray(
            segmentation_info.resize((224, 224)).convert(mode="RGB")
        )
        label_mask, unique_labels = encode_segmap(segmentation)

        images[i] = image
        label_masks[i] = label_mask
        labels[i] = unique_labels
        
        bounding_boxes[i] = parse_xml_annotation(path_root + image_file)
        
        # plt.imshow(image.permute(1, 2, 0).numpy())
        # for (_, min_x, min_y, max_x, max_y) in bounding_boxes[i]:
        #     rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
        #                             linewidth=1, edgecolor='r', facecolor='none')
        #     plt.gca().add_patch(rect)
        # plt.savefig(f"./temp_imgs/{image_file[:-4]}.png")
        # plt.clf()
        # if i > 5:
        #     exit()

    dataset = {"data": images, "labels": labels, "mask": label_masks, "bbs": bounding_boxes}
    print(f"saving {len(data)} images")

    os.makedirs(args.save_path, exist_ok=True)
    torch.save(dataset, os.path.join(args.save_path, split + "_segment.pt"))


parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default=".")
parser.add_argument(
    "--split",
    type=str,
    choices=["train", "val", "test", "trainval", "all"],
    required=True,
)
parser.add_argument("--save_path", type=str, default="processed/")
args = parser.parse_args()
main(args)
