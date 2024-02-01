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

# In the segmentation images every class has a color, these can be accesed per
# class with these colors.
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
        code from:
        # https://github.com/meetps/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py
    """
    
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    unique_labels = torch.zeros(20, dtype=int)
    
    # for every class find the corresponding pixels and set them to this class
    for idx, label in enumerate(color_classes):
        indexes = np.where(np.all(mask == label, axis=-1))[:2]
        if list(indexes[0]) and idx != 0:  # if no labels in this class
            unique_labels[idx - 1] = 1
        label_mask[indexes] = idx

    label_mask = torch.tensor(label_mask.astype(int))
    return label_mask, unique_labels

def parse_xml_annotation(xml_file_path, target_size=224):
    """ Retrieves relevant information from the annotation xml file
    
    Args:
        xml_file_path: The relative file path to search for to retrieve the xml
          file. Example: ./VOCdevkit/VOC2007/Annotations/0001.xml
        target_size: the square size the output images will have. 
          ResNet requires 224x224 images.
    Returns:
        list(list): a list of lists  of every bounding box in the image,
          where every list is structured in the format:
          [class index, min x-value, min y-value, max x-value, max y-value]
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    rescaled_annotations = []
    size = root.find('size')
    image_width = int(size.find('width').text)
    image_height = int(size.find('height').text)
    wscale = target_size / image_width
    hscale = target_size / image_height

    # loop over every bounding box
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_idx = target_dict[class_name]
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # scale bounding box from original size to target_size X target_size
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
        # retrieve segmentation image IDs
        image_files = []
        with open(f"{args.data_root}/VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt", "r") as f:
            for line in f:
                image_files.append(line.strip() + ".xml")
        with open(f"{args.data_root}/VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt", "r") as f:
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
        # retrieve segmentation image IDs
        image_files = []
        with open(f"{args.data_root}/VOCdevkit/VOC2007/ImageSets/Segmentation/{split}.txt", "r") as f:
            for line in f:
                image_files.append(line.strip() + ".xml")

    images = torch.zeros(len(data), 3, 224, 224)
    label_masks = torch.zeros(len(data), 1, 224, 224)
    labels = torch.zeros((len(data), 20))
    bounding_boxes = [[] for _ in range(len(data))]
    path_root = f"{args.data_root}/VOCdevkit/VOC2007/Annotations/"
    
    # for every image data, filename; create the data object
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
        
        if args.show_img:
            # show segmantation over original image
            plot_img = image.permute(1,2,0).numpy()
            
            mask = ~np.all(segmentation == 0, axis=-1)
            
            # Modify plot_img where segmentation != 0
            plot_img[mask] = (segmentation[mask] / 255) * 0.5 + plot_img[mask] * 0.5
            plt.imshow(plot_img)
            
            for (class_idx, min_x, min_y, max_x, max_y) in bounding_boxes[i]:
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                        linewidth=3, 
                                        edgecolor=np.array(color_classes[class_idx + 1]) / 255 , 
                                        facecolor='none')
                plt.gca().add_patch(rect)
            if i <= args.img_amount: # increase to see more images
                os.makedirs("temp_imgs", exist_ok=True)
                plt.savefig(f"./temp_imgs/{image_file[:-4]}.png")
            plt.clf()
        
        # Visualize the image "sheep" from appendix Visual EPG metrics comparison
        # note that --img_amount should be said to at least 20.
        if args.show_img and image_file == "000676.xml":
            os.makedirs("temp_imgs", exist_ok=True)

            blue = np.array([30, 170, 80]) /255
            light_blue =  np.array([118,130,20]) / 255
            red = np.array([255, 118, 95]) / 255
            light_red =  np.array([225, 85, 65]) / 255
            # show segmantation over original image
            plot_img = image.permute(1,2,0).numpy()
            og_img = plot_img.copy()
                                    
            mask = np.all(np.isclose(segmentation, np.array([224, 224, 192])), axis=-1)
            
            # Modify plot_img where segmentation != 0
            og_img[mask] = blue
            plt.imshow(og_img)

            
            for (class_idx, min_x, min_y, max_x, max_y) in bounding_boxes[i]:
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                        linewidth=5, 
                                        edgecolor= blue, 
                                        facecolor='none')
                plt.gca().add_patch(rect)
            
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"./temp_imgs/{image_file[:-4]}_original.png")
            mask = ~np.all(segmentation == 0, axis=-1)
            
            # Bounding box
            bbs_img = np.zeros_like(plot_img)
            bbs_img[:, :] = red
            for (class_idx, min_x, min_y, max_x, max_y) in bounding_boxes[i]:
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                        linewidth=5, 
                                        edgecolor=light_blue , 
                                        facecolor='none')
                plt.gca().add_patch(rect)
                bbs_img[min_y:max_y, min_x:max_x] = blue
            
            plt.axis("off")
            plt.tight_layout()
            plt.imshow(bbs_img)
            plt.savefig(f"./temp_imgs/{image_file[:-4]}_bbs.png")
            plt.clf()

            segment_img = np.zeros_like(plot_img)
            segment_img[:, :] = red
            for (class_idx, min_x, min_y, max_x, max_y) in bounding_boxes[i]:
                segment_img[mask] = blue
            plt.axis("off")
            plt.tight_layout()
            plt.imshow(segment_img)
            plt.savefig(f"./temp_imgs/{image_file[:-4]}_segment.png")


            # Fraction
            for (class_idx, min_x, min_y, max_x, max_y) in bounding_boxes[i]:
                plot_img[min_y:max_y, min_x:max_x] = red
                plot_img[mask] = blue
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                        linewidth=5, 
                                        edgecolor=light_red , 
                                        facecolor='none')
                plt.gca().add_patch(rect)
            plt.imshow(plot_img)

            plt.axis("off")
            plt.tight_layout()

            if i <= args.img_amount: # increase to see more images
                os.makedirs("temp_imgs", exist_ok=True)
                plt.savefig(f"./temp_imgs/{image_file[:-4]}_fraction.png")
            plt.clf()

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
parser.add_argument("--show_img", type=bool, default=False)
parser.add_argument("--img_amount", type=int, default=10)


args = parser.parse_args()
main(args)
