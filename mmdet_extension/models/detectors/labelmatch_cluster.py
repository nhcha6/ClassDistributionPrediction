# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
LabelMatchCluster
"""
import numpy as np
import pickle

import torch

from mmcv.runner.dist_utils import get_dist_info
from mmdet.utils import get_root_logger
from mmdet.core.bbox.iou_calculators import iou2d_calculator
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.models.builder import DETECTORS

from mmdet_extension.models.detectors import SemiTwoStageDetector


@DETECTORS.register_module()
class LabelMatchCluster(SemiTwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 init_cfg=None,
                 # ema model
                 ema_config=None,
                 ema_ckpt=None,
                 classes=None,
                 # config
                 cfg=dict(),
                 # cluster image details
                 cluster_imgs=None
                 ):
        super().__init__(backbone=backbone, rpn_head=rpn_head, roi_head=roi_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, neck=neck, init_cfg=init_cfg,
                         ema_config=ema_config, ema_ckpt=ema_ckpt, classes=classes)
        self.debug = cfg.get('debug', False)
        self.num_classes = self.roi_head.bbox_head.num_classes
        self.cur_iter = 0

        # hyper-parameter: fixed
        self.tpt = cfg.get('tpt', 0.5)
        self.tps = cfg.get('tps', 1.0)
        self.momentum = cfg.get('momentum', 0.996)
        self.weight_u = cfg.get('weight_u', 2.0)

        # adat
        # this is where the cls_thr and cls_thr_ig variables for storing the adaptive threshold
        # are declared
        score_thr = cfg.get('score_thr', 0.9)  # if not use ACT, will use this hard thr
        self.cls_thr = [0.9 if self.debug else score_thr] * self.num_classes
        self.cls_thr_ig = [0.2 if self.debug else score_thr] * self.num_classes
        self.percent = cfg.get('percent', 0.2)
    
        # import the images associated with each cluster and convert to set
        with open(cluster_imgs, 'rb') as handle:
            self.cluster_imgs = pickle.load(handle)
        for cluster, imgs in self.cluster_imgs.items():
            self.cluster_imgs[cluster] = set(imgs)

        # mining
        self.use_mining = cfg.get('use_mining', True)
        self.reliable_thr = cfg.get('reliable_thr', 0.8)
        self.reliable_iou = cfg.get('reliable_iou', 0.8)

        # analysis
        self.image_num = 0
        self.pseudo_num = np.zeros(self.num_classes)
        self.pseudo_num_ig = np.zeros(self.num_classes)
        self.pseudo_num_tp = np.zeros(self.num_classes)
        self.pseudo_num_gt = np.zeros(self.num_classes)
        self.pseudo_num_tp_ig = np.zeros(self.num_classes)
        self.pseudo_num_mining = np.zeros(self.num_classes)

    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1,
    ):
        device = img.device
        _, _, h, w = img_unlabeled_1.shape
        self.image_num += len(img_metas_unlabeled)
        self.update_ema_model(self.momentum)
        self.cur_iter += 1
        self.analysis()  # record the information in the training
        # # ---------------------label data---------------------
        losses = self.forward_train(img, img_metas, gt_bboxes, gt_labels)
        losses = self.parse_loss(losses)
        # # -------------------unlabeled data-------------------
        bbox_transform, bbox_transform_1 = [], []
        for img_meta, img_meta_1 in zip(img_metas_unlabeled, img_metas_unlabeled_1):
            bbox_transform.append(img_meta.pop('bbox_transform'))
            bbox_transform_1.append(img_meta_1.pop('bbox_transform'))
        # create pseudo label
        bbox_results = self.inference_unlabeled(
            img_unlabeled, img_metas_unlabeled, rescale=True
        )
        gt_bboxes_pred, gt_labels_pred, gt_bboxes_ig_pred, gt_labels_ig_pred = \
            self.create_pseudo_results(
                img_unlabeled_1, bbox_results, bbox_transform_1, device,
                gt_bboxes_unlabeled, gt_labels_unlabeled, img_metas_unlabeled  # for analysis
            )
        if self.debug:
            self.visual_online(img_unlabeled_1, gt_bboxes_pred, gt_labels_pred,
                               boxes_ignore_list=gt_bboxes_ig_pred)
        # training on unlabeled data
        losses_unlabeled = self.training_unlabeled(
            img_unlabeled_1, img_metas_unlabeled_1, bbox_transform_1,
            img_unlabeled, img_metas_unlabeled, bbox_transform,
            gt_bboxes_pred, gt_labels_pred, gt_bboxes_ig_pred, gt_labels_ig_pred
        )
        losses_unlabeled = self.parse_loss(losses_unlabeled)
        for key, val in losses_unlabeled.items():
            if key.find('loss') == -1:
                continue
            else:
                losses_unlabeled[key] = self.weight_u * val
        losses.update({f'{key}_unlabeled': val for key, val in losses_unlabeled.items()})
        # extra info for analysis
        extra_info = {
            'pseudo_num': torch.Tensor([self.pseudo_num.sum() / self.image_num]).to(device),
            'pseudo_num_ig': torch.Tensor([self.pseudo_num_ig.sum() / self.image_num]).to(device),
            'pseudo_num_mining': torch.Tensor([self.pseudo_num_mining.sum() / self.image_num]).to(device),
            'pseudo_num(acc)': torch.Tensor([self.pseudo_num_tp.sum() / self.pseudo_num.sum()]).to(device),
            'pseudo_num ig(acc)': torch.Tensor([self.pseudo_num_tp_ig.sum() / (self.pseudo_num_ig.sum() + 1e-10)]).to(
                device),
        }
        losses.update(extra_info)
        return losses

    # # ---------------------------------------------------------------------------------
    # # training on unlabeled data
    # # ---------------------------------------------------------------------------------
    def training_unlabeled(self, img, img_metas, bbox_transform,
                           img_t, img_metas_t, bbox_transform_t,
                           gt_bboxes, gt_labels, gt_bboxes_ig, gt_labels_ig):
        losses = dict()
        x = self.extract_feat(img)
        # rpn loss
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        gt_bboxes_cmb = [torch.cat([a, b]) for a, b in zip(gt_bboxes, gt_bboxes_ig)]
        rpn_losses, proposal_list = self.rpn_head.forward_train(
            x, img_metas, gt_bboxes_cmb, gt_labels=None, proposal_cfg=proposal_cfg)
        losses.update(rpn_losses)
        # roi loss
        sampling_results = self.roi_head.forward_train_step1(
            x, img_metas, proposal_list, gt_bboxes, gt_labels, gt_bboxes_ig, gt_labels_ig
        )
        # teacher model to get pred
        ig_boxes = [torch.cat([ig, res.ig_bboxes])
                    for ig, res in zip(gt_bboxes_ig, sampling_results)]
        ig_len = [len(ig) for ig in gt_bboxes_ig]
        for i in range(len(img_metas)):
            ig_boxes[i] = self.rescale_bboxes(ig_boxes[i], img_metas[i], bbox_transform_t[i])
        ignore_boxes_t = [b[:l] for l, b in zip(ig_len, ig_boxes)]
        ig_boxes = [b[l:] for l, b in zip(ig_len, ig_boxes)]
        with torch.no_grad():
            ema_model = self.ema_model.module
            x_t = ema_model.extract_feat(img_t)
            det_bboxes_t, det_labels_t = ema_model.roi_head.simple_test_bboxes_base(
                x_t, img_metas_t, ig_boxes)
            cls_scores_t = [torch.softmax(l / self.tpt, dim=-1) for l in det_labels_t]
            det_labels_t = [torch.softmax(l, dim=-1) for l in det_labels_t]
        # mining
        for n, res in enumerate(sampling_results):
            for i in range(max(res.ig_assigned_gt_inds) + 1 if len(res.ig_assigned_gt_inds) > 0 else 0):
                flag = res.ig_assigned_gt_inds == i
                if flag.sum() < 1:
                    continue
                cls_cur = gt_labels_ig[n][i]
                if self.use_mining:
                    mean_iou = iou2d_calculator.bbox_overlaps(ignore_boxes_t[n][i:i + 1],
                                                              det_bboxes_t[n][flag]).mean()
                    mean_score = det_labels_t[n][flag][:, cls_cur].mean()
                    if mean_iou >= self.reliable_iou and mean_score >= self.reliable_thr:
                        res.ig_reg_weight[flag] = 1.0
                        self.pseudo_num_mining[cls_cur] += 1
        roi_losses, cls_scores = self.roi_head.forward_train_step2(
            x, sampling_results, gt_bboxes, gt_labels)
        losses.update(roi_losses)
        # proposal based learning
        weight = torch.cat([1 - res.ig_reg_weight for res in sampling_results])
        cls_scores_t = torch.cat(cls_scores_t)
        cls_scores = torch.softmax(cls_scores / self.tps, dim=-1)
        if len(cls_scores) > 0:
            avg_factor = len(img_metas) * self.roi_head.bbox_sampler.num
            losses_cls_ig = (-cls_scores_t * torch.log(cls_scores)).sum(-1)
            losses_cls_ig = (losses_cls_ig * weight).sum() / avg_factor
        else:
            losses_cls_ig = cls_scores.sum()  # 0
        losses.update({'losses_cls_ig': losses_cls_ig})
        return losses

    # # ---------------------------------------------------------------------------------
    # # create pseudo labels
    # # ---------------------------------------------------------------------------------

    # this function uses the thresholds calculated in labelmatch_hooks.py to implement
    # the labelmatch mean teacher framework. This function will need to be updated if we are
    # calculating a specific threshold for different clusters.
    def create_pseudo_results(self, img, bbox_results, box_transform, device,
                              gt_bboxes=None, gt_labels=None, img_metas=None):
        # print(len(bbox_results))
        # print(bbox_results)
        # print(len(img_metas))
        # print(img_metas)


        """using dynamic score to create pseudo results"""
        gt_bboxes_pred, gt_labels_pred = [], []
        gt_bboxes_ig_pred, gt_labels_ig_pred = [], []
        _, _, h, w = img.shape
        use_gt = gt_bboxes is not None
        
        for b, result in enumerate(bbox_results):
            # calculate which cluster the current image belongs to
            img_name = img_metas[b]['filename'].split('/')[-1]
            for cluster, imgs in self.cluster_imgs.items():
                if img_name in imgs:
                    current_cluster = cluster
                    break
            # print(current_cluster)
            # extract the relevant thresholds
            cls_thr = self.cls_thr[current_cluster]
            cls_thr_ig = self.cls_thr_ig[current_cluster]

            # use adaptive thresholds to select the pseudo-labels for self-training
            bboxes, labels = [], []
            bboxes_ig, labels_ig = [], []
            if use_gt:
                gt_bbox, gt_label = gt_bboxes[b].cpu().numpy(), gt_labels[b].cpu().numpy()
                scale_factor = img_metas[b]['scale_factor']
                gt_bbox_scale = gt_bbox / scale_factor
            for cls, r in enumerate(result):
                label = cls * np.ones_like(r[:, 0], dtype=np.uint8)
                flag_pos = r[:, -1] >= cls_thr[cls]
                flag_ig = (r[:, -1] >= cls_thr_ig[cls]) & (~flag_pos)
                bboxes.append(r[flag_pos][:, :4])
                bboxes_ig.append(r[flag_ig][:, :4])
                labels.append(label[flag_pos])
                labels_ig.append(label[flag_ig])
                if use_gt and (gt_label == cls).sum() > 0 and len(bboxes[-1]) > 0:
                    overlap = bbox_overlaps(bboxes[-1], gt_bbox_scale[gt_label == cls])
                    iou = overlap.max(-1)
                    self.pseudo_num_tp[cls] += (iou > 0.5).sum()
                if use_gt and (gt_label == cls).sum() > 0 and len(bboxes_ig[-1]) > 0:
                    overlap = bbox_overlaps(bboxes_ig[-1], gt_bbox_scale[gt_label == cls])
                    iou = overlap.max(-1)
                    self.pseudo_num_tp_ig[cls] += (iou > 0.5).sum()
                self.pseudo_num_gt[cls] += (gt_label == cls).sum()
                self.pseudo_num[cls] += len(bboxes[-1])
                self.pseudo_num_ig[cls] += len(bboxes_ig[-1])
            bboxes = np.concatenate(bboxes)
            bboxes_ig = np.concatenate(bboxes_ig)
            bboxes_concat = np.r_[bboxes, bboxes_ig]
            labels = np.concatenate(labels)
            labels_ig = np.concatenate(labels_ig)
            for bf in box_transform[b]:
                bboxes_concat, labels = bf(bboxes_concat, labels)
            bboxes, bboxes_ig = bboxes_concat[:len(bboxes)], bboxes_concat[len(bboxes):]
            gt_bboxes_pred.append(torch.from_numpy(bboxes).float().to(device))
            gt_labels_pred.append(torch.from_numpy(labels).long().to(device))
            gt_bboxes_ig_pred.append(torch.from_numpy(bboxes_ig).float().to(device))
            gt_labels_ig_pred.append(torch.from_numpy(labels_ig).long().to(device))
        return gt_bboxes_pred, gt_labels_pred, gt_bboxes_ig_pred, gt_labels_ig_pred

    # # -----------------------------analysis function------------------------------
    def analysis(self):
        if self.cur_iter % 500 == 0 and get_dist_info()[0] == 0:
            logger = get_root_logger()
            info = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                             in zip(self.CLASSES, self.pseudo_num, self.pseudo_num_tp)])
            info_ig = ' '.join([f'{b / (a + 1e-10):.2f}({a}-{cls})' for cls, a, b
                                in zip(self.CLASSES, self.pseudo_num_ig, self.pseudo_num_tp_ig)])
            info_gt = ' '.join([f'{a}' for a in self.pseudo_num_gt])
            logger.info(f'pseudo pos: {info}')
            logger.info(f'pseudo ig: {info_ig}')
            logger.info(f'pseudo gt: {info_gt}')
            if self.use_mining:
                info_mining = ' '.join([f'{a}' for a in self.pseudo_num_mining])
                logger.info(f'pseudo mining: {info_mining}')
