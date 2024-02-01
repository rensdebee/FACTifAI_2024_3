"""
Reproducibility Study of “Studying How to Efficiently and Effectively Guide Models with Explanations”

Description: This file is part of a project aiming to reproduce the study titled "Studying How to Efficiently and 
Effectively Guide Models with Explanations." The project focuses on verifying the results and methodologies 
proposed in the original study, and potentially extending or refining the study's findings.

Based on the code of orginal paper: https://github.com/sukrutrao/Model-Guidance

losses.py
"""

import torch

def get_localization_loss(loss_name):
    loss_map = {
        "Energy": EnergyPointingGameBBMultipleLoss,
        "L1": GradiaBBMultipleLoss,
        "RRR": RRRBBMultipleLoss,
        "PPCE": HAICSBBMultipleLoss
    }
    return loss_map[loss_name]()


class BBMultipleLoss:

    def __init__(self):
        super().__init__()

    def __call__(self, attributions, bb_coordinates):
        raise NotImplementedError

    def get_bb_mask(self, bb_coordinates, mask_shape):
        bb_mask = torch.zeros(mask_shape, dtype=torch.long)
        for coords in bb_coordinates:
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1
        return bb_mask


class EnergyPointingGameBBMultipleLoss:

    def __init__(self):
        super().__init__()
        self.only_positive = False
        self.binarize = False

    def __call__(self, attributions, bb_coordinates):
        pos_attributions = attributions.clamp(min=0)
        bb_mask = torch.zeros_like(pos_attributions, dtype=torch.long)
        for coords in bb_coordinates:
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1
        num = pos_attributions[torch.where(bb_mask == 1)].sum()
        den = pos_attributions.sum()
        if den < 1e-7:
            return 1-num
        return 1-num/den



class RRRBBMultipleLoss(BBMultipleLoss):

    def __init__(self):
        super().__init__()
        self.only_positive = False
        self.binarize = True

    def __call__(self, attributions, bb_coordinates):
        bb_mask = self.get_bb_mask(bb_coordinates, attributions.shape)
        irrelevant_attrs = attributions[torch.where(bb_mask == 0)]
        return torch.square(irrelevant_attrs).sum()


class GradiaBBMultipleLoss(BBMultipleLoss):

    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.only_positive = True
        self.binarize = True

    def __call__(self, attributions, bb_coordinates):
        bb_mask = self.get_bb_mask(bb_coordinates, attributions.shape).cuda()
        return self.l1_loss(attributions, bb_mask)


class HAICSBBMultipleLoss(BBMultipleLoss):

    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCELoss(reduction='mean')
        self.only_positive = True
        self.binarize = True

    def __call__(self, attributions, bb_coordinates):
        bb_mask = self.get_bb_mask(bb_coordinates, attributions.shape)
        attributions_in_box = attributions[torch.where(bb_mask == 1)]
        return self.bce_loss(attributions_in_box, torch.ones_like(attributions_in_box))