# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
hooks for LabelMatchCluster
"""
import shutil
import os.path as osp
import numpy as np
import pickle

import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm

import mmcv
from mmcv.runner import HOOKS, Hook, get_dist_info
from mmdet.utils import get_root_logger
from mmdet.core.evaluation import EvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.apis import inference_detector, init_detector, show_result_pyplot


@HOOKS.register_module()
class LabelMatchClusterHook(Hook):
    def __init__(self, cfg):
        rank, world_size = get_dist_info()
        distributed = world_size > 1
        samples_per_gpu = cfg.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.pipeline = replace_ImageToTensor(cfg.data.pipeline)
        
        # random select 10000 image as reference image (in order to save the inference time)
        dataset = build_dataset(cfg.data, dict(test_mode=True))
        dataloader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        self.CLASSES = dataset.CLASSES
        
        # this function calculates the class distribution of the example dataset
        boxes_per_image_gt, cls_ratio_gt = self.get_data_info(cfg.label_file)
        file = cfg.data['ann_file']
        eval_cfg = cfg.get('evaluation', {})
        
        # alternatively, we can set a manual prior to be what ever we like. This must be input when
        # the labelmatchhook is declared in the training config file.
        cluster_prior_fp = cfg.get('cluster_prior', None)
        cluster_imgs_fp = cfg.get('cluster_imgs', None)

        # target thr
        target_thr = cfg.get('target_thr', 0.6)

        # import saved cluster priors and imgs
        with open(cluster_prior_fp, 'rb') as handle:
            cluster_prior = pickle.load(handle)
        with open(cluster_imgs_fp, 'rb') as handle:
            cluster_imgs = pickle.load(handle)
        
        # calculate the expected number of positives for each cluster
        potential_positives = {}
        for cluster, manual_prior in cluster_prior.items():
            if manual_prior:  # manual setting the boxes_per_image and cls_ratio
                boxes_per_image_gt = manual_prior.get('boxes_per_image', boxes_per_image_gt)
                cls_ratio_gt = manual_prior.get('cls_ratio', cls_ratio_gt)

                # print the class distribution
                logger = get_root_logger()
                logger.info(f'boxes per image ({cluster}): {boxes_per_image_gt}')
                logger.info(f'class ratio ({cluster}): {cls_ratio_gt}')

                # calculate the number of positives we expect to see in the sampled dataset
                potential_positive = [boxes_per_image_gt * ratio for ratio in cls_ratio_gt]
                potential_positives[cluster] = potential_positive
        min_thr = cfg.get('min_thr', 0.05)  # min cls score threshold for ignore

        # eval_hook object is declared to control what runs each time the hook is called
        # this eval_hook controls the recalculation of the class thresholds every n iterations
        if distributed:
            self.eval_hook = LabelMatchClusterDistEvalHook(
                file, dataloader, potential_positives, boxes_per_image_gt, target_thr, cls_ratio_gt, cluster_imgs, min_thr, **eval_cfg)
        else:
            self.eval_hook = LabelMatchClusterEvalHook(
                file, dataloader, potential_positives, boxes_per_image_gt, target_thr, cls_ratio_gt, cluster_imgs, min_thr, **eval_cfg)

    # this function calculates the class distribution of the example dataset
    def get_data_info(self, json_file):
        """get information from labeled data"""
        print(json_file)
        info = mmcv.load(json_file)
        id2cls = {}
        total_image = len(info['images'])
        for value in info['categories']:
            id2cls[value['id']] = self.CLASSES.index(value['name'])
        cls_num = [0] * len(self.CLASSES)
        for value in info['annotations']:
            cls_num[id2cls[value['category_id']]] += 1
        cls_num = [max(c, 1) for c in cls_num]  # for some cls not select, we set it 1 rather than 0
        total_boxes = sum(cls_num)
        cls_ratio_gt = np.array([c / total_boxes for c in cls_num])
        boxes_per_image_gt = total_boxes / total_image
        logger = get_root_logger()
        info = ' '.join([f'({v:.4f}-{self.CLASSES[i]})' for i, v in enumerate(cls_ratio_gt)])
        logger.info(f'boxes per image (actual data): {boxes_per_image_gt}')
        logger.info(f'class ratio (actual data): {info}')
        return boxes_per_image_gt, cls_ratio_gt

    # the hook functions simply point to those in self.eval_hook
    def before_train_epoch(self, runner):
        self.eval_hook.before_train_epoch(runner)

    def after_train_epoch(self, runner):
        self.eval_hook.after_train_epoch(runner)

    def after_train_iter(self, runner):
        self.eval_hook.after_train_iter(runner)

    def before_train_iter(self, runner):
        self.eval_hook.before_train_epoch(runner)


class LabelMatchClusterEvalHook(EvalHook):
    def __init__(self,
                 file,
                 dataloader,
                 potential_positive,
                 boxes_per_image_gt,
                 target_thr,
                 cls_ratio_gt,
                 cluster_imgs,
                 min_thr,
                 **eval_kwargs
                 ):
        super().__init__(dataloader, **eval_kwargs)
        self.file = file
        self.dst_root = None
        self.initial_epoch_flag = True

        self.cluster_imgs = cluster_imgs
        self.potential_positive = potential_positive
        self.boxes_per_image_gt = boxes_per_image_gt
        self.target_thr = target_thr
        self.cls_ratio_gt = cls_ratio_gt
        self.min_thr = min_thr
        self.dataloader = dataloader

        self.CLASSES = self.dataloader.dataset.CLASSES

    def before_train_epoch(self, runner):
        if not self.initial_epoch_flag:
            return
        if self.dst_root is None:
            self.dst_root = runner.work_dir
        interval_temp = self.interval
        self.interval = 1
        if self.by_epoch:
            self.after_train_epoch(runner)
        else:
            self.after_train_iter(runner)
        self.initial_epoch_flag = False
        self.interval = interval_temp
        runner.model.module.boxes_per_image_gt = self.boxes_per_image_gt
        runner.model.module.cls_ratio_gt = self.cls_ratio_gt

    # self.update_cls_thr(runner) is where the new thresholds are calculated
    def after_train_epoch(self, runner):
        if not self.by_epoch or not self.evaluation_flag(runner):
            return
        # self.update_cls_thr(runner)
        self.update_regularised_thr(runner)

    def after_train_iter(self, runner):
        if self.by_epoch or not self.every_n_iters(runner, self.interval):
            return
        # self.update_cls_thr(runner)
        self.update_regularised_thr(runner)

    def do_evaluation(self, runner):
        from mmdet_extension.apis.test import single_gpu_test
        self.dataloader.dataset.shuffle_data_info()  # random shuffle
        results = single_gpu_test(runner.model.module.ema_model, self.dataloader)
        return results

    # calls self.eval_score_thresh to calculate the desired threhold for each class
    def update_cls_thr(self, runner):
        # get the percentage of pseudo-labels that are considered reliable
        percent = runner.model.module.percent  # percent as positive
        # get the the inference of the teacher on the dataset
        results = self.do_evaluation(runner)
        
        # visualise result
        # for i in range(len(results)):
        #     show_result_pyplot(runner.model, self.dataloader.dataset.img_prefix + self.dataloader.dataset.data_infos[i]['filename'], results[i], score_thr=0.9, wait_time=0)
        
        # collect set of images from the current batch
        test_data_set = set()
        for i in range(len(results)):
            im_info = self.dataloader.dataset.data_infos[i]
            test_data_set.add(im_info['filename'])

        # update the cluster_imgs to only include those in the sampled dataset
        cluster_imgs = {}
        for cluster, imgs in self.cluster_imgs.items():
            new_imgs = []
            for im in imgs:
                if 'JPEGImages/' + im in test_data_set:
                    new_imgs.append(im)
            cluster_imgs[cluster] = new_imgs
        

        # generate dictionary containing results and images from each cluster
        cluster_results = {}
        cluster_imgs = {}
        for i in range(len(results)):
            im_name = self.dataloader.dataset.data_infos[i]['filename'].split('/')[-1]
            for cluster, imgs in self.cluster_imgs.items():
                if im_name in imgs:
                    if cluster in cluster_results.keys():
                        cluster_results[cluster].append(results[i])
                        cluster_imgs[cluster].append(im_name)
                    else:
                        cluster_results[cluster] = [results[i]]
                        cluster_imgs[cluster] = [im_name]
        

        # iterate through self.potential_positives for each cluster
        cls_thr_cluster = {}
        cls_thr_ig_cluster = {}
        logger = get_root_logger()
        for cluster, pp in self.potential_positive.items():
            logger.info(f'current cluster: {cluster}')

            # print the iteration number
            # scale=0.5+float(runner._iter)/32000.0
            # logger.info(f'current scale: {scale}')
            scale = 1

            # update pp with cluster size, which was not available up to this point
            pp = [scale*x*len(cluster_imgs[cluster]) for x in pp]
           
            # calculate the classification thresholds (one for reliable pseudo-labels and one for unreliable)
            cls_thr, cls_thr_ig = self.eval_score_thr(cluster_results[cluster], percent, pp)
            cls_thr_cluster[cluster] = cls_thr
            cls_thr_ig_cluster[cluster] = cls_thr_ig
        
        # print(cls_thr_cluster)
        # print(cls_thr_ig_cluster)
        
        # set the cls_thr and cls_thr_ig for use in label match training
        runner.model.module.cls_thr = cls_thr_cluster
        runner.model.module.cls_thr_ig = cls_thr_ig_cluster 

    # calls self.eval_score_thresh to calculate the desired threhold for each class
    def update_regularised_thr(self, runner):
        # get the percentage of pseudo-labels that are considered reliable
        percent = runner.model.module.percent  # percent as positive
        # get the the inference of the teacher on the dataset
        results = self.do_evaluation(runner)
        
        # visualise result
        # for i in range(len(results)):
        #     show_result_pyplot(runner.model, self.dataloader.dataset.img_prefix + self.dataloader.dataset.data_infos[i]['filename'], results[i], score_thr=0.9, wait_time=0)
        
        # collect set of images from the current batch
        test_data_set = set()
        for i in range(len(results)):
            im_info = self.dataloader.dataset.data_infos[i]
            test_data_set.add(im_info['filename'])

        # update the cluster_imgs to only include those in the sampled dataset
        cluster_imgs = {}
        for cluster, imgs in self.cluster_imgs.items():
            new_imgs = []
            for im in imgs:
                if 'JPEGImages/' + im in test_data_set:
                    new_imgs.append(im)
            cluster_imgs[cluster] = new_imgs
        

        # generate dictionary containing results and images from each cluster
        cluster_results = {}
        cluster_imgs = {}
        for i in range(len(results)):
            im_name = self.dataloader.dataset.data_infos[i]['filename'].split('/')[-1]
            for cluster, imgs in self.cluster_imgs.items():
                if im_name in imgs:
                    if cluster in cluster_results.keys():
                        cluster_results[cluster].append(results[i])
                        cluster_imgs[cluster].append(im_name)
                    else:
                        cluster_results[cluster] = [results[i]]
                        cluster_imgs[cluster] = [im_name]

        # iterate through self.potential_positives for each cluster
        cls_thr_cluster = {}
        cls_thr_ig_cluster = {}
        logger = get_root_logger()
        for cluster, pp in self.potential_positive.items():
            logger.info(f'current cluster: {cluster}')
            logger.info(f'objects per im: {self.boxes_per_image_gt}')
            cls_ratio_gt = [x/self.boxes_per_image_gt for x in pp]
            logger.info(f'class ratio: {self.cls_ratio_gt}')
            logger.info(f'number of images: {len(cluster_imgs[cluster])}')

            # calculate the confidence score list
            score_list = [[] for _ in self.CLASSES]
            for result in cluster_results[cluster]:
                for cls, r in enumerate(result):
                    score_list[cls].append(r[:, -1])
            score_list = [np.concatenate(c) for c in score_list]
            score_list = [np.zeros(1) if len(c) == 0 else np.sort(c)[::-1] for c in score_list]

            # calculate the percent and boxes per image that yield the desired average threshold
            percent_optimal, boxes_per_image_optimal = self.eval_regularised_thr(self.boxes_per_image_gt, cls_ratio_gt, score_list, len(cluster_imgs[cluster]))
            logger.info(f'optimal percent: {percent_optimal}')
            logger.info(f'optimal boxes per image: {boxes_per_image_optimal}')

            # calculate the optimal potential positive
            pp_optimal = [x*len(cluster_imgs[cluster])*boxes_per_image_optimal for x in self.cls_ratio_gt]

            # update pp with cluster size, which was not available up to this point
            pp = [x*len(cluster_imgs[cluster]) for x in pp]
            
            # calculate the classification thresholds (one for reliable pseudo-labels and one for unreliable)
            # this method uses the default percent and the pp as per the prior number of objects per image
            # cls_thr, cls_thr_ig = self.eval_score_thr(cluster_results[cluster], percent, pp)
            # alternatively, we use what we have calculated
            cls_thr, cls_thr_ig = self.eval_score_thr(cluster_results[cluster], percent_optimal, pp_optimal)
            
            cls_thr_cluster[cluster] = cls_thr
            cls_thr_ig_cluster[cluster] = cls_thr_ig
        
        # print(cls_thr_cluster)
        # print(cls_thr_ig_cluster)
        
        # set the cls_thr and cls_thr_ig for use in label match training
        runner.model.module.cls_thr = cls_thr_cluster
        runner.model.module.cls_thr_ig = cls_thr_ig_cluster

    # this function is what calculates the threshold for each class
    # first, the confidence scores for each class are placed in a list and sorted
    # then, the threshold that produces the expected number of positives is calculated and store in cls_thr_ig
    # the most confident x% of these thresholds are then extracted and placed in cls_thr - these are termed reliable pseudo-labels
    def eval_score_thr(self, results, percent, potential_positive):
        score_list = [[] for _ in self.CLASSES]
        for result in results:
            for cls, r in enumerate(result):
                score_list[cls].append(r[:, -1])
        score_list = [np.concatenate(c) for c in score_list]
        score_list = [np.zeros(1) if len(c) == 0 else np.sort(c)[::-1] for c in score_list]
        cls_thr = [0] * len(self.CLASSES)
        cls_thr_ig = [0] * len(self.CLASSES)
        for i, score in enumerate(score_list):
            cls_thr[i] = max(0.05, score[min(int(potential_positive[i] * percent), len(score) - 1)])
            # NOTE: original use 0.05, for UDA, we change to 0.001
            cls_thr_ig[i] = max(self.min_thr, score[min(int(potential_positive[i]), len(score) - 1)])
        logger = get_root_logger()
        logger.info(f'current percent: {percent}')
        info = ' '.join([f'({v:.2f}-{self.CLASSES[i]})' for i, v in enumerate(cls_thr)])
        logger.info(f'update score thr (positive): {info}')
        info = ' '.join([f'({v:.2f}-{self.CLASSES[i]})' for i, v in enumerate(cls_thr_ig)])
        logger.info(f'update score thr (ignore): {info}')
        return cls_thr, cls_thr_ig
    
    def eval_regularised_thr(self, boxes_per_image_gt, cls_ratio_gt, score_list, num_images):
        mean_thr = []
        boxes_per_img_list = []
        for step in range(1, 100):
            boxes_per_img = step*boxes_per_image_gt/50
            potential_positive = [x*boxes_per_img*num_images for x in cls_ratio_gt]

            # iterate through each class and print the results
            selected_thr = []
            num_labels = []
            average_thr = []
            for cls, scores in enumerate(score_list):
                # calculate the index set by the class distribution prior
                try:
                    i = int(potential_positive[cls])
                    score_list[cls][i]
                except IndexError:
                    i = -1

                # if the threshold is less than 0.05, we need to recalculate it
                # when calculating average threshold, we do not want to consider this
                # if score_list[cls][i]<0.05:
                #     closest_thr = self.closest_value(score_list[cls], 0.05)
                #     i = list(score_list[cls]).index(closest_thr)

                # what if the desired threshold is below the minimum?
                selected_thr.append(score_list[cls][i])
                if i==0:
                    average_thr.append(0)
                else:
                    average_thr.append(np.mean(score_list[cls][0:i]))
                num_labels.append(i)

            # if using the thresholds
            # mean_thr.append(self.class_weighted_mean(selected_thr, num_labels))
            # if using the average confidence scores
            mean_thr.append(self.class_weighted_mean(average_thr, num_labels))
            boxes_per_img_list.append(boxes_per_img)

        # calculate the noptimal number of boxes per image, where mean_thr = 0.3
        # closest_thr = self.closest_value(mean_thr, 0.3)
        # if using the average confidence score, we instead want mean_thr = 0.6
        print(self.target_thr)
        closest_thr = self.closest_value(mean_thr, self.target_thr)
        i = mean_thr.index(closest_thr)
        boxes_per_img_optimal = boxes_per_img_list[i]
        potential_positive = [x*boxes_per_img_optimal*num_images for x in cls_ratio_gt]
        # iterate through each class and print the results
        mean_thr = []
        percents = [x/20 for x in range(1, 21)]
        for percent in percents:
            percent_thr = []
            num_labels = []
            for cls, score in enumerate(score_list):
                # calculate the index set by the class distribution prior
                try:
                    i = int(potential_positive[cls]*percent)
                    score_list[cls][i]
                except IndexError:
                    i = -1
                # if the threshold is less than 0.05, we need to recalculate it
                if score_list[cls][i]<0.05:
                    closest_thr = self.closest_value(score_list[cls], 0.05)
                    i = list(score_list[cls]).index(closest_thr)
                
                percent_thr.append(score_list[cls][i])
                num_labels.append(i)
            
            mean_thr.append(self.class_weighted_mean(percent_thr, num_labels))
        
        # calculate the optimal percent value, where mean_thr = 0.7
        closest_thr = self.closest_value(mean_thr, 0.7)
        i = mean_thr.index(closest_thr)
        percent_optimal = percents[i]

        return percent_optimal, boxes_per_img_optimal
            
    def class_weighted_mean(self, values, class_dist=False):
        try:
            weighted_values = [values[i]*class_dist[i] for i in range(len(values))]
            return sum(weighted_values)/sum(class_dist)
        except:
            return np.mean(values)
    
    def closest_value(self, input_list, input_value):
        arr = np.asarray(input_list)
        i = (np.abs(arr - input_value)).argmin()
        return arr[i]

# if making changes to LabelMatchEvalHook, must also change the distributed version
class LabelMatchClusterDistEvalHook(LabelMatchClusterEvalHook):
    def __init__(self,
                 file,
                 dataloader,
                 potential_positive,
                 boxes_per_image_gt,
                 target_thr,
                 cls_ratio_gt,
                 cluster_imgs,
                 min_thr,
                 tmpdir=None,
                 gpu_collect=False,
                 broadcast_bn_buffer=True,
                 **eval_kwargs
                 ):
        super().__init__(file, dataloader, potential_positive, boxes_per_image_gt, target_thr, cls_ratio_gt, cluster_imgs, min_thr, **eval_kwargs)
        self.broadcast_bn_buffer = broadcast_bn_buffer
        self.tmpdir = tmpdir
        self.gpu_collect = gpu_collect

    def _broadcast_bn_buffer(self, model):
        if self.broadcast_bn_buffer:
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

    def do_evaluation(self, runner):
        if self.broadcast_bn_buffer:
            self._broadcast_bn_buffer(runner.model.module.ema_model)
        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')
        from mmdet_extension.apis.test import multi_gpu_test
        self.dataloader.dataset.shuffle_data_info()
        results = multi_gpu_test(runner.model.module.ema_model, self.dataloader,
                                 tmpdir=tmpdir, gpu_collect=self.gpu_collect)
        return results

    def update_regularised_thr(self, runner):
        percent = runner.model.module.percent  # percent as positive
        results = self.do_evaluation(runner)
        tmpdir = './tmp_file'
        tmpfile = osp.join(tmpdir, 'tmp.pkl')

        if runner.rank == 0:
            # generate dictionary containing results and images from each cluster
            cluster_results = {}
            cluster_imgs = {}
            for i in range(len(results)):
                im_name = self.dataloader.dataset.data_infos[i]['filename'].split('/')[-1]
                for cluster, imgs in self.cluster_imgs.items():
                    if im_name in imgs:
                        if cluster in cluster_results.keys():
                            cluster_results[cluster].append(results[i])
                            cluster_imgs[cluster].append(im_name)
                        else:
                            cluster_results[cluster] = [results[i]]
                            cluster_imgs[cluster] = [im_name]

            # iterate through self.potential_positives for each cluster
            cls_thr_cluster = {}
            cls_thr_ig_cluster = {}
            logger = get_root_logger()
            for cluster, pp in self.potential_positive.items():
                logger.info(f'current cluster: {cluster}')
                logger.info(f'objects per im: {self.boxes_per_image_gt}')
                cls_ratio_gt = [x/self.boxes_per_image_gt for x in pp]
                logger.info(f'class ratio: {self.cls_ratio_gt}')
                logger.info(f'number of images: {len(cluster_imgs[cluster])}')

                # calculate the confidence score list
                score_list = [[] for _ in self.CLASSES]
                for result in cluster_results[cluster]:
                    for cls, r in enumerate(result):
                        score_list[cls].append(r[:, -1])
                score_list = [np.concatenate(c) for c in score_list]
                score_list = [np.zeros(1) if len(c) == 0 else np.sort(c)[::-1] for c in score_list]

                # calculate the percent and boxes per image that yield the desired average threshold
                percent_optimal, boxes_per_image_optimal = self.eval_regularised_thr(self.boxes_per_image_gt, cls_ratio_gt, score_list, len(cluster_imgs[cluster]))
                logger.info(f'optimal percent: {percent_optimal}')
                logger.info(f'optimal boxes per image: {boxes_per_image_optimal}')

                # calculate the optimal potential positive
                pp_optimal = [x*len(cluster_imgs[cluster])*boxes_per_image_optimal for x in self.cls_ratio_gt]

                # update pp with cluster size, which was not available up to this point
                pp = [x*len(cluster_imgs[cluster]) for x in pp]
                
                # calculate the classification thresholds (one for reliable pseudo-labels and one for unreliable)
                # this method uses the default percent and the pp as per the prior number of objects per image
                # cls_thr, cls_thr_ig = self.eval_score_thr([cluster], percent, pp)
                # alternatively, we use what we have calculated
                cls_thr, cls_thr_ig = self.eval_score_thr(cluster_results[cluster], percent, pp_optimal)

                cls_thr_cluster[cluster] = cls_thr
                cls_thr_ig_cluster[cluster] = cls_thr_ig

            mmcv.mkdir_or_exist(tmpdir)
            mmcv.dump((cls_thr_cluster, cls_thr_ig_cluster), tmpfile)

        dist.barrier()
        cls_thr_cluster, cls_thr_ig_cluster = mmcv.load(tmpfile)
        dist.barrier()
        if runner.rank == 0:
            shutil.rmtree(tmpdir)
        runner.model.module.cls_thr = cls_thr_cluster
        runner.model.module.cls_thr_ig = cls_thr_ig_cluster

    def update_cls_thr(self, runner):
        percent = runner.model.module.percent  # percent as positive
        results = self.do_evaluation(runner)
        tmpdir = './tmp_file'
        tmpfile = osp.join(tmpdir, 'tmp.pkl')

        if runner.rank == 0:
            # generate dictionary containing results and images from each cluster
            cluster_results = {}
            cluster_imgs = {}
            for i in range(len(results)):
                im_name = self.dataloader.dataset.data_infos[i]['filename'].split('/')[-1]
                for cluster, imgs in self.cluster_imgs.items():
                    if im_name in imgs:
                        if cluster in cluster_results.keys():
                            cluster_results[cluster].append(results[i])
                            cluster_imgs[cluster].append(im_name)
                        else:
                            cluster_results[cluster] = [results[i]]
                            cluster_imgs[cluster] = [im_name]

            # iterate through self.potential_positives for each cluster
            cls_thr_cluster = {}
            cls_thr_ig_cluster = {}
            logger = get_root_logger()
            for cluster, pp in self.potential_positive.items():
                logger.info(f'current cluster: {cluster}')
                # update pp with cluster size, which was not available up to this point
                pp = [x*len(cluster_imgs[cluster]) for x in pp]
                # calculate the classification thresholds (one for reliable pseudo-labels and one for unreliable)
                cls_thr, cls_thr_ig = self.eval_score_thr(cluster_results[cluster], percent, pp)
                cls_thr_cluster[cluster] = cls_thr
                cls_thr_ig_cluster[cluster] = cls_thr_ig

            mmcv.mkdir_or_exist(tmpdir)
            mmcv.dump((cls_thr_cluster, cls_thr_ig_cluster), tmpfile)

        dist.barrier()
        cls_thr_cluster, cls_thr_ig_cluster = mmcv.load(tmpfile)
        dist.barrier()
        if runner.rank == 0:
            shutil.rmtree(tmpdir)
        runner.model.module.cls_thr = cls_thr_cluster
        runner.model.module.cls_thr_ig = cls_thr_ig_cluster