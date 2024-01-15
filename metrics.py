import torch
import torchmetrics
import statistics
import torchmetrics.classification
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

    def binarize(self, attributions):
        # Also normalization
        attr_max = attributions.max()
        attr_min = attributions.min()
        if attr_max == 0:
            return attributions
        if torch.abs(attr_max - attr_min) < 1e-7:
            return attributions / attr_max
        return (attributions - attr_min) / (attr_max - attr_min)

    def visualize(self, binarized_attributions, bb_coordinates, image):
        bb_mask = torch.zeros_like(binarized_attributions, dtype=torch.long)
        for coords in bb_coordinates:
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1
        bb_size = len(torch.where(bb_mask == 1)[0])

        fig, axs = plt.subplots(1, 6, figsize=(20, 5))
        if image is not None:
            axs[0].imshow(torch.movedim(image[:3, :, :], 0, -1).cpu())
            axs[0].set_title("Original Image")

        i = 0
        i = i + 1 if image is not None else i
        axs[i].imshow(
            binarized_attributions.cpu(),
            cmap="Reds",
        )
        axs[i].set_title("Normalized atribution map")

        methods = ["mean", "median", "mode", "fixed"]
        for i, method in enumerate(methods):
            if method == "mean":
                iou_threshold = binarized_attributions.mean()
            elif method == "mode":
                iou_threshold = binarized_attributions.flatten().mode()[0]
            elif method == "mean":
                iou_threshold = binarized_attributions.median()
            elif method == "fixed":
                iou_threshold = 0.5
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

            i = i + 1 if image is not None else i
            axs[i + 1].imshow(
                torch.where(binarized_attributions > self.iou_threshold, 1, 0).cpu(),
                cmap="Reds",
            )
            axs[i + 1].set_title(
                f"{method}, Threshold: {self.iou_threshold:.2f}, IoU: {iou:.4f}"
            )

        for ax in axs:
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
        fig.tight_layout()
        fig.suptitle("Comparision of threshold method for IoU score:    ")
        plt.savefig("./methods_comparions.png")

    def update(self, attributions, bb_coordinates, image=None):
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

        binarized_attributions = self.binarize(positive_attributions)
        self.iou_threshold = binarized_attributions.median()
        if self.visualize_flag:
            self.visualize(binarized_attributions, bb_coordinates, image)
            self.visualize_flag = False

        intersection_area = len(
            torch.where((binarized_attributions > self.iou_threshold) & (bb_mask == 1))[
                0
            ]
        )
        union_area = (
            len(torch.where(binarized_attributions > self.iou_threshold)[0])
            + len(torch.where(bb_mask == 1)[0])
            - intersection_area
        )
        assert intersection_area >= 0
        assert union_area >= 0

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
