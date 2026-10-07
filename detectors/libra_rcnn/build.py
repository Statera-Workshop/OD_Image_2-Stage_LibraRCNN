"""
Libra R-CNN builder (ResNet-50-FPN), Image task.

Paper: Pang, Chen, Shi, Feng, Ouyang & Lin, "Libra R-CNN: Towards Balanced Learning for Object Detection"
       (CVPR 2019, arXiv 1904.02701).
Code : official OceanPang/Libra_R-CNN is an mmdetection fork; the same classes in mmdetection 3.x are used:
       configs/libra_rcnn/libra-faster-rcnn_r50_fpn_1x_coco.py, models/necks/bfp.py (BFP: refine_level 2,
       refine_type 'non_local'), models/layers NonLocal2d (embedded gaussian, reduction 1, use_scale False, zero-init
       output conv), task_modules/samplers (CombinedSampler: InstanceBalancedPosSampler + IoUBalancedNegSampler
       floor_thr -1, floor_fraction 0, num_bins 3; 512 RoIs, 25 % positives), losses/balanced_l1_loss.py
       (alpha 0.5, gamma 1.5, beta 1.0). Detector, RPN, RoI head: detectors/faster_rcnn (torchvision R50-FPN).

    IoU-balanced sampling     negatives drawn evenly from 3 IoU bins, positives evenly per GT instance
    Balanced feature pyramid  P2-P6 resized to P4, averaged, refined by a non-local block, added back to every level
    Balanced L1 loss          on the class-specific box deltas, / #samples

[Unified] as detectors/faster_rcnn; the common init also re-draws the non-local convs (incl. the zero-init output
conv), as for every other nn.Conv2d.
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou

from detectors.faster_rcnn.build import FasterRCNNDet


class NonLocal2d(nn.Module):
    """mmcv NonLocal2d, mode embedded_gaussian, reduction 1, use_scale False, zeros_init True"""

    def __init__(self, in_channels, reduction=1):
        super().__init__()
        self.inter_channels = max(in_channels // reduction, 1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.conv_out = nn.Conv2d(self.inter_channels, in_channels, 1)
        for m in (self.g, self.theta, self.phi):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x):
        n = x.size(0)
        g_x = self.g(x).view(n, self.inter_channels, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(n, self.inter_channels, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(n, self.inter_channels, -1)
        pairwise_weight = torch.matmul(theta_x, phi_x).softmax(dim=-1)
        y = torch.matmul(pairwise_weight, g_x).permute(0, 2, 1).reshape(n, self.inter_channels, *x.size()[2:])
        return x + self.conv_out(y)


class BFP(nn.Module):
    def __init__(self, in_channels=256, num_levels=5, refine_level=2):
        super().__init__()
        self.num_levels, self.refine_level = num_levels, refine_level
        self.refine = NonLocal2d(in_channels, reduction=1)

    def forward(self, inputs):
        gather_size = inputs[self.refine_level].size()[2:]
        feats = []
        for i in range(self.num_levels):
            if i < self.refine_level:
                feats.append(F.adaptive_max_pool2d(inputs[i], output_size=gather_size))
            else:
                feats.append(F.interpolate(inputs[i], size=gather_size, mode="nearest"))
        bsf = self.refine(sum(feats) / len(feats))
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode="nearest")
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])
        return outs


def balanced_l1_loss(pred, target, beta=1.0, alpha=0.5, gamma=1.5):
    diff = torch.abs(pred - target)
    b = np.e ** (gamma / alpha) - 1
    return torch.where(diff < beta, alpha / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
                       gamma * diff + gamma / b - alpha * beta)


def _random_choice(idx, num):
    return idx[torch.randperm(len(idx), device=idx.device)[:num]]


def instance_balanced_pos(pos_inds, gt_inds, num_expected):
    """mmdet InstanceBalancedPosSampler._sample_pos"""
    if len(pos_inds) <= num_expected:
        return pos_inds
    unique_gt_inds = gt_inds[pos_inds].unique()
    num_per_gt = int(round(num_expected / float(len(unique_gt_inds))) + 1)
    sampled = []
    for i in unique_gt_inds:
        inds = pos_inds[gt_inds[pos_inds] == i]
        sampled.append(_random_choice(inds, num_per_gt) if len(inds) > num_per_gt else inds)
    sampled = torch.cat(sampled)
    if len(sampled) < num_expected:
        extra = pos_inds[~torch.isin(pos_inds, sampled)]
        sampled = torch.cat([sampled, _random_choice(extra, num_expected - len(sampled))])
    elif len(sampled) > num_expected:
        sampled = _random_choice(sampled, num_expected)
    return sampled


def iou_balanced_neg(neg_inds, max_overlaps, num_expected, floor_thr=-1, num_bins=3):
    """mmdet IoUBalancedNegSampler._sample_neg with floor_thr -1, floor_fraction 0 (all negatives IoU-balanced)"""
    if len(neg_inds) <= num_expected:
        return neg_inds
    max_iou = max_overlaps.max()
    iou_interval = (max_iou - floor_thr) / num_bins
    per_num_expected = int(num_expected / num_bins)
    sampled = []
    ov = max_overlaps[neg_inds]
    for i in range(num_bins):
        start_iou, end_iou = floor_thr + i * iou_interval, floor_thr + (i + 1) * iou_interval
        tmp = neg_inds[(ov >= start_iou) & (ov < end_iou)]
        sampled.append(_random_choice(tmp, per_num_expected) if len(tmp) > per_num_expected else tmp)
    sampled = torch.cat(sampled)
    if len(sampled) < num_expected:
        extra = neg_inds[~torch.isin(neg_inds, sampled)]
        sampled = torch.cat([sampled, _random_choice(extra, num_expected - len(sampled))])
    return sampled


class LibraRCNN(FasterRCNNDet):
    def __init__(self, num_classes, num=512, pos_fraction=0.25):
        super().__init__(num_classes)
        self.bfp = BFP(256, num_levels=5, refine_level=2)
        self.num, self.pos_fraction = num, pos_fraction

    def neck(self, features):
        keys = list(features.keys())                                   # '0'-'3' (P2-P5), 'pool' (P6)
        outs = self.bfp([features[k] for k in keys])
        return type(features)(zip(keys, outs))

    @torch.no_grad()
    def sample(self, proposals, tgt):
        rh = self.det.roi_heads
        out_p, out_l, out_r = [], [], []
        for p, t in zip(proposals, tgt):
            gt_boxes, gt_labels = t["boxes"], t["labels"]
            p = torch.cat([p, gt_boxes])                                # add_gt_as_proposals
            if len(gt_boxes):
                ious = box_iou(gt_boxes, p)
                max_overlaps, gt_inds = ious.max(0)
                matched = rh.proposal_matcher(ious)                    # 0.5 / 0.5 -> -1 = bg
                pos_inds = torch.where(matched >= 0)[0]
                neg_inds = torch.where(matched < 0)[0]
            else:
                max_overlaps = p.new_zeros(len(p))
                gt_inds = torch.zeros(len(p), dtype=torch.long, device=p.device)
                pos_inds, neg_inds = p.new_zeros(0, dtype=torch.long), torch.arange(len(p), device=p.device)
            num_pos = int(self.num * self.pos_fraction)
            pos = instance_balanced_pos(pos_inds, gt_inds, num_pos)
            neg = iou_balanced_neg(neg_inds, max_overlaps, self.num - len(pos))
            keep = torch.cat([pos, neg])
            labels = torch.zeros(len(keep), dtype=torch.long, device=p.device)
            labels[:len(pos)] = gt_labels[gt_inds[pos]] if len(gt_boxes) else labels[:0]
            sp = p[keep]
            mg = gt_boxes[gt_inds[keep]] if len(gt_boxes) else torch.zeros_like(sp)
            out_p.append(sp)
            out_l.append(labels)
            out_r.append(rh.box_coder.encode([mg], [sp])[0])
        return out_p, out_l, out_r

    def box_loss(self, class_logits, box_regression, labels, regression_targets):
        labels, regression_targets = torch.cat(labels, 0), torch.cat(regression_targets, 0)
        classification_loss = F.cross_entropy(class_logits, labels)
        pos = torch.where(labels > 0)[0]
        N = class_logits.shape[0]
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
        box_loss = balanced_l1_loss(box_regression[pos, labels[pos]], regression_targets[pos]).sum() / labels.numel()
        return classification_loss, box_loss


def build_model(task, num_classes):
    return LibraRCNN(num_classes)
