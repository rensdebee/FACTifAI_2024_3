import torch
import torchmetrics
import statistics
import torchmetrics.classification
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np


class EnergyPointingGameBase(torchmetrics.Metric):
    def __init__(self, include_undefined=True):
        super().__init__()

        self.include_undefined = include_undefined

        self.add_state("fractions", default=[])
        self.add_state("defined_idxs", default=[])

    def update(self, attributions, mask_or_coords):
        raise NotImplementedError

    def compute(self):
        if len(self.fractions) == 0:
            return None
        if self.include_undefined:
            return statistics.fmean(self.fractions)
        if len(self.defined_idxs) == 0:
            return None
        return statistics.fmean([self.fractions[idx] for idx in self.defined_idxs])


class BoundingBoxEnergyMultiple(EnergyPointingGameBase):
    def __init__(self, include_undefined=True, min_box_size=None, max_box_size=None):
        super().__init__(include_undefined=include_undefined)
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size

    def update(self, attributions, bb_coordinates):
        positive_attributions = attributions.clamp(min=0)
        bb_mask = torch.zeros_like(positive_attributions, dtype=torch.long)
        for coords in bb_coordinates:
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1
        bb_size = len(torch.where(bb_mask == 1)[0])
        if self.min_box_size is not None and bb_size < self.min_box_size:
            return
        if self.max_box_size is not None and bb_size >= self.max_box_size:
            return
        energy_inside = positive_attributions[torch.where(bb_mask == 1)].sum()
        energy_total = positive_attributions.sum()
        assert energy_inside >= 0, energy_inside
        assert energy_total >= 0, energy_total
        if energy_total < 1e-7:
            self.fractions.append(torch.tensor(0.0))
        else:
            self.defined_idxs.append(len(self.fractions))
            self.fractions.append(energy_inside / energy_total)


class BoundingBoxIoUMultiple(EnergyPointingGameBase):
    def __init__(
        self,
        include_undefined=True,
        iou_threshold=0.5,
        min_box_size=None,
        max_box_size=None,
        vis_flag=False,
    ):
        super().__init__(include_undefined=include_undefined)
        self.iou_threshold = iou_threshold
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size
        self.visualize_flag = vis_flag
        # Create subplots and figure for the amount of images to visaluze
        self.amount_img = 10
        if self.visualize_flag:
            # Create figure
            self.fig, self.axs = plt.subplots(
                self.amount_img, 6, figsize=(20, 5 * self.amount_img)
            )
        # Set index of image visualized count to 0
        self.j = 0

    def binarize(self, attributions):
        # Normalize attributions
        attr_max = attributions.max()
        attr_min = attributions.min()
        if attr_max == 0:
            return attributions
        if torch.abs(attr_max - attr_min) < 1e-7:
            return attributions / attr_max
        return (attributions - attr_min) / (attr_max - attr_min)

    def visualize(self, binarized_attributions, bb_coordinates, image=None):
        """
        Function for visualizing different threshold methods for the IoU score
        Takes:
        binarized_attributions: The binarized atribution map
        bb_coordinates: list of list of coordinates
        image: original RGB images (HxWxC)
        """
        # Create custom color map
        cdict = {
            "red": [(0.0, 1.0, 1.0), (1.0, 1.0, 1.0)],
            "green": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
            "blue": [(0.0, 1.0, 1.0), (1.0, 0.0, 0.0)],
        }

        dark_red_cmap = mcolors.LinearSegmentedColormap("DarkRed", cdict)

        # Get the boundingbox mask
        bb_mask = torch.zeros_like(binarized_attributions, dtype=torch.long)

        # Get coordinates of boundingbox
        for coords in bb_coordinates:
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1

        # If og image is given plot it in first subplot
        if image is not None:
            self.axs[self.j][0].imshow(torch.movedim(image[:3, :, :], 0, -1).cpu())
            self.axs[self.j][0].set_title("Original Image")

        # Plot attirubtions in first or second plot, depending on if an
        # image is given
        i = 0
        i = i + 1 if image is not None else i
        self.axs[self.j][i].imshow(
            binarized_attributions.cpu(),
            cmap=dark_red_cmap,
        )
        self.axs[self.j][i].set_title("Normalized atribution map")

        # histogram, bin_edges = np.histogram(binarized_attributions.cpu())

        # axs[i + 1].plot(bin_edges[0:-1], histogram)
        # axs[i + 1].set_title("Value histogram ")

        # Loop over different threshold methods
        methods = ["mean", "top 5%", "median", "fixed"]
        unique_values = len(binarized_attributions.flatten().unique())
        for i, method in enumerate(methods):
            # Set threshold
            if method == "mean":
                iou_threshold = binarized_attributions.mean()
            elif method == "top 5%":
                iou_threshold = binarized_attributions.flatten().unique()[
                    -int(unique_values * 0.05)
                ]
            elif method == "median":
                iou_threshold = binarized_attributions.flatten().unique().median()
            elif method == "fixed":
                iou_threshold = 0.5
            # Calculate IoU
            intersection_area = len(
                torch.where((binarized_attributions > iou_threshold) & (bb_mask == 1))[
                    0
                ]
            )
            union_area = (
                len(torch.where(binarized_attributions > iou_threshold)[0])
                + len(torch.where(bb_mask == 1)[0])
                - intersection_area
            )
            assert intersection_area >= 0
            assert union_area >= 0
            if union_area == 0:
                iou = 0.0
            else:
                iou = intersection_area / union_area

            # Plot binarized image with IoU in title
            i = i + 1 if image is not None else i
            self.axs[self.j][i + 1].imshow(
                torch.where(binarized_attributions > iou_threshold, 1, 0).cpu(),
                cmap=dark_red_cmap,
            )
            self.axs[self.j][i + 1].set_title(
                f"{method}, Threshold: {iou_threshold:.2f}, IoU: {iou:.2f}"
            )

        # Add boundingbox to all subplots
        for ax in self.axs[self.j]:
            ax.add_patch(
                patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fc="none",
                    ec="royalblue",
                    lw=2,
                )
            )

        # Increase count of images being visualized
        self.j += 1
        # If amount neede reached stop visualizing
        if self.j >= self.amount_img:
            # Save figure
            self.fig.tight_layout()
            self.fig.suptitle("Comparision of threshold method for IoU score:    ")
            plt.savefig("./methods_comparions.png")
            self.visualize_flag = False

    def update(self, attributions, bb_coordinates, image=None):
        """
        Function to update the IoU score class with IoU score of new image.
        Takes:
        attributions: attribution map of image
        bb_coordinates: bounding box coordinates list
        image: Original image for visualizing.
        """
        # Set all negative attributions to 0
        positive_attributions = attributions.clamp(min=0)
        # Get the boundingbox mask
        bb_mask = torch.zeros_like(positive_attributions, dtype=torch.long)
        for coords in bb_coordinates:
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1
        # Count the amount of pixels inside the boundingbox
        bb_size = len(torch.where(bb_mask == 1)[0])

        if self.min_box_size is not None and bb_size < self.min_box_size:
            return
        if self.max_box_size is not None and bb_size >= self.max_box_size:
            return

        # Normalize the positive attributions
        binarized_attributions = self.binarize(positive_attributions)

        # If first couple of images and visualize flag is true visualize different methods
        if self.visualize_flag:
            self.visualize(binarized_attributions, bb_coordinates, image)

        # Calculate amount of pixels inside boundingbox and higher then threshold
        intersection_area = len(
            torch.where((binarized_attributions > self.iou_threshold) & (bb_mask == 1))[
                0
            ]
        )

        # Calculate number of pixels higher then threshold plus inside boundingbox
        union_area = (
            len(torch.where(binarized_attributions > self.iou_threshold)[0])
            + len(torch.where(bb_mask == 1)[0])
            - intersection_area
        )
        assert intersection_area >= 0
        assert union_area >= 0

        # Calculate and store IoU score
        if union_area == 0:
            iou = 0.0
            self.fractions.append(torch.tensor(iou))
        else:
            iou = intersection_area / union_area
            self.defined_idxs.append(len(self.fractions))
            self.fractions.append(torch.tensor(iou))


"""
Source: https://github.com/stevenstalder/NN-Explainer 
"""


class MultiLabelMetrics(torchmetrics.Metric):
    def __init__(self, num_classes, threshold):
        super().__init__()

        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("true_positives", torch.tensor(0.0))
        self.add_state("false_positives", torch.tensor(0.0))
        self.add_state("true_negatives", torch.tensor(0.0))
        self.add_state("false_negatives", torch.tensor(0.0))

    def update(self, logits, labels):
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                for j in range(self.num_classes):
                    if labels[i][j] == 1.0:
                        if batch_sample_logits[j] >= self.threshold:
                            self.true_positives += 1.0
                        else:
                            self.false_negatives += 1.0
                    else:
                        if batch_sample_logits[j] >= self.threshold:
                            self.false_positives += 1.0
                        else:
                            self.true_negatives += 1.0

    def compute(self):
        self.accuracy = (self.true_positives + self.true_negatives) / (
            self.true_positives
            + self.true_negatives
            + self.false_positives
            + self.false_negatives
        )
        self.precision = self.true_positives / (
            self.true_positives + self.false_positives
        )
        self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        self.f_score = (2 * self.true_positives) / (
            2 * self.true_positives + self.false_positives + self.false_negatives
        )

        return {
            "Accuracy": self.accuracy.item(),
            "Precision": self.precision.item(),
            "Recall": self.recall.item(),
            "F-Score": self.f_score.item(),
            "True Positives": self.true_positives.item(),
            "True Negatives": self.true_negatives.item(),
            "False Positives": self.false_positives.item(),
            "False Negatives": self.false_negatives.item(),
        }

    def save(self, model, classifier_type, dataset):
        f = open(
            model + "_" + classifier_type + "_" + dataset + "_" + "test_metrics.txt",
            "w",
        )
        f.write("Accuracy: " + str(self.accuracy.item()) + "\n")
        f.write("Precision: " + str(self.precision.item()) + "\n")
        f.write("Recall: " + str(self.recall.item()) + "\n")
        f.write("F-Score: " + str(self.f_score.item()))
        f.close()
