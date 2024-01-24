import numpy as np
import os
import torch
import utils
from torchvision import tv_tensors
import numpy as np


class VOCDetectParsed(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        image_set,
        transform=None,
        annotated_fraction=1.0,
        bbs_transform=False,
        plot=False,
    ):
        super().__init__()
        data_dict = torch.load(os.path.join(root, image_set + ".pt"))
        self.data = data_dict["data"]
        self.labels = data_dict["labels"]
        self.bbs = data_dict["bbs"]
        self.bbs_transform = bbs_transform
        self.plot = plot
        assert len(self.data) == len(self.labels)
        assert len(self.data) == len(self.bbs)
        self.annotated_fraction = annotated_fraction

        if self.annotated_fraction < 1.0:
            annotated_indices = np.random.choice(
                len(self.bbs),
                size=int(self.annotated_fraction * len(self.bbs)),
                replace=False,
            )
            for bb_idx in range(len(self.bbs)):
                if bb_idx not in annotated_indices:
                    self.bbs[bb_idx] = None
        self.transform = transform

    def __getitem__(self, idx):
        if self.bbs_transform is not None:
            bbs = self.bbs[idx]
            none_flag = False
            if bbs is None:
                none_flag = True
                bbs = [[0, 0, 0, 0, 0]]
            bbs = np.asarray(bbs)
            img = self.data[idx]
            boxes = tv_tensors.BoundingBoxes(
                bbs[:, 1:],
                format="XYXY",
                canvas_size=img.shape[-2:],
            )
            out_img, out_boxes = self.bbs_transform(img, boxes)
            bbs = np.hstack((bbs[:, :1], np.asarray(out_boxes)))
            if self.plot:
                return (
                    out_img,
                    self.labels[idx],
                    bbs,
                    self.data[idx],
                    self.labels[idx],
                    self.bbs[idx],
                )
            if none_flag:
                bbs = None
            elif self.transform is not None:
                return self.transform(out_img), self.labels[idx], bbs
            else:
                return out_img, self.labels[idx], bbs
        if self.transform is not None:
            return self.transform(self.data[idx]), self.labels[idx], self.bbs[idx]
        return self.data[idx], self.labels[idx], self.bbs[idx]

    def load_data(self, idx, pred_class):
        img, labels, bbs = self.__getitem__(idx)
        label = labels[pred_class]
        bb = utils.filter_bbs(bbs, pred_class)
        return img, label, bb

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        data = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        bbs = [item[2] for item in batch]
        if len(batch[0]) == 6:
            og_data = torch.stack([item[3] for item in batch])
            og_labels = torch.stack([item[4] for item in batch])
            og_bbs = [item[5] for item in batch]
            return data, labels, bbs, og_data, og_labels, og_bbs
        return data, labels, bbs


class SegmentDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.all_data = torch.load(root)
        self.data = self.all_data["data"]
        self.labels = self.all_data["labels"]
        self.masks = self.all_data["mask"]
        self.bbs = self.all_data["bbs"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {
            "data": self.transform(self.data[index]),  # transform data
            "label": self.labels[index],
            "mask": self.masks[index],
            "bbs": self.bbs[index],
        }
        return sample

    @staticmethod
    def collate_fn(batch):
        datas = [item["data"] for item in batch]
        labels = [item["label"] for item in batch]
        masks = [item["mask"] for item in batch]
        bbs = [item["bbs"] for item in batch]

        batched_datas = torch.stack(datas)
        batched_labels = torch.stack(labels)
        batched_masks = torch.stack(masks)

        return [batched_datas, batched_labels, batched_masks, bbs]
