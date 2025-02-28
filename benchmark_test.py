# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os
import random
import queue
import threading
from collections.abc import MutableMapping, Sequence

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# Models
from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from minvis import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    add_minvis_config,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)


def to_cuda(packed_data):
    if isinstance(packed_data, bytes):
        return packed_data

    if isinstance(packed_data, torch.Tensor):
        packed_data = packed_data.to(device="cuda", non_blocking=True)
    elif isinstance(packed_data, (int, float, str, bool, complex)):
        packed_data = packed_data
    elif isinstance(packed_data, MutableMapping):
        for key, value in packed_data.items():
            packed_data[key] = to_cuda(value)
    elif isinstance(packed_data, Sequence):
        try:
            for i, value in enumerate(packed_data):
                packed_data[i] = to_cuda(value)
        except TypeError:
            pass
    return packed_data


class CUDADataLoader:

    def __init__(self, dataloader):
        self.dataloader = dataloader

        self.stream = torch.cuda.Stream() # create a new cuda stream in each process
        # setting a queue for storing prefetched data
        self.queue = queue.Queue(16)
        # 
        self.iter = dataloader.__iter__()
        # starting a new thread to prefetch data
        def data_to_cuda_then_queue():
            while True:
                try:
                    self.preload()
                except StopIteration:
                    break
            # NOTE: end flag for the queue
            self.queue.put(None)
        self.cuda_thread = threading.Thread(target=data_to_cuda_then_queue, args=())
        self.cuda_thread.daemon = True

        # NOTE: preload several batch of data
        (self.preload() for _ in range(8))
        self.cuda_thread.start()

    def preload(self):
        batch = next(self.iter)
        if batch is None:
            return None
        torch.cuda.current_stream().wait_stream(self.stream)  # wait tensor to put on GPU
        with torch.cuda.stream(self.stream):
            batch = to_cuda(batch)
            # batch = batch.to(device="cuda", non_blocking=True)
        self.queue.put(batch)

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        next_item = self.queue.get()
        # NOTE: __iter__ will be stopped when __next__ raises StopIteration 
        if next_item is None:
            raise StopIteration
        return next_item

    def __del__(self):
        # NOTE: clean up the thread
        try:
            self.cuda_thread.join(timeout=10)
        finally:
            if self.cuda_thread.is_alive():
                self.cuda_thread._stop()
        # NOTE: clean up the stream
        self.stream.synchronize()
        # NOTE: clean up the queue
        self.queue.queue.clear()


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        return YTVISEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]
        mapper = YTVISDatasetMapper(cfg, is_train=True)

        dataset_dict = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        train_loader = build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dict)

        return CUDADataLoader(train_loader)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        dataset_name = cfg.DATASETS.TEST[0]
        # if dataset_name in ["ytvis2019_val", "ytvis2019_test", "ytvis2021_val", "ytvis2021_test", "ovis_val", "lvvis_val", "lvvis_test"]:
        mapper = YTVISDatasetMapper(cfg, is_train=False)
        test_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)

        return CUDADataLoader(test_loader)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with torch.amp.autocast("cuda"):
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if 'OUTPUT_DIR' not in args.opts:
        work_dir_prefix = os.path.dirname(args.config_file).replace('configs/', '')
        work_dir_suffix = os.path.splitext(os.path.basename(args.config_file))[0]
        cfg.OUTPUT_DIR = f'work_dirs/{work_dir_prefix}/{work_dir_suffix}'
        if args.eval_only:
            cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, 'eval')

    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(name="mask2former")
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="openvis")
    return cfg


def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def main(args):
    cfg = setup(args)
    # set_seed()

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    model = model.eval()

    data_loader = Trainer.build_test_loader(cfg, cfg.DATASETS.TEST[0])
    # data_mapper = YTVISDatasetMapper(cfg, is_train=False)

    for idx, inputs in enumerate(data_loader):
        with torch.amp.autocast("cuda"), torch.no_grad():
            # directly inference
            outputs = model(inputs)
        
        if idx == 5:
            break

    import time
    avg_time = []
    for idx, inputs in enumerate(data_loader):
        with torch.amp.autocast("cuda"), torch.no_grad():
            # online inference
            images = sum([x["image"] for x in inputs], [])

            video_outputs = [None]
            for image in images:
                start = time.time()
                video_output = model.online_inference(image[None], video_outputs[-1])
                video_outputs.append(video_output)
                end = time.time()
                avg_time.append(end - start)
                if len(avg_time) > 5:
                    print("frame time:", sum(avg_time[5:]) / len(avg_time[5:]))
            video_outputs = video_outputs[1:]

        if idx == 10:
            break

# def main(args):
#     cfg = setup(args)

#     model = Trainer.build_model(cfg)
#     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)

#     model = model.eval()

#     pseudo_input = [
#         {
#             "image": [torch.rand(3, 720, 1280).cuda()] * 40,
#             "height": 720,
#             "width":  1280,
#         },
#         # {
#         #     "image": [torch.rand(3, 727, 1236).cuda()] * 40,
#         #     "height": 727,
#         #     "width":  1236,
#         # }
#     ]

#     # AMP context
#     for _ in range(10):
#         with torch.amp.autocast("cuda"), torch.no_grad():
#             outputs = model(pseudo_input)

#     # with torch.amp.autocast("cuda"), torch.no_grad():
#     #     for i in range(4):
#     #         outputs = model.online_inference(torch.randn(1, 5, 3, 720, 1280).cuda())

#     exit(0)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args, ),
    )
