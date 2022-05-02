
"""
#  @Time     : 2021/1/28  21:02
#  @Author   : Yufang
#  滑窗推理，从上往下
"""


import argparse
import warnings
import tempfile

import copy
import os.path as osp
import os
import shutil
import torch.distributed as dist
import cv2
import math
import pdb
import torch
import numpy as np
from tqdm import tqdm
import json
import time
import pickle
import mmcv
from mmcv.parallel import collate, scatter
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn

from mmdet.datasets.pipelines import Compose
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
# from mmdet.ops.nms.nms_wrapper import nms
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.models import build_detector

from mmcv.ops.nms import nms

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection network inference')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--patch_size',
                        nargs='+', type=int, help='patch_size')
    parser.add_argument('--strides',
                        nargs='+', type=int, help='strides')
    parser.add_argument(
        '--iou_thr',
        type=float,
        default=0.5,
        help='nms iou threshold (default: 0.5)')

    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function (deprecate), '
             'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options

    return args


def load_annotation(ann_file):
    with open(ann_file, 'r') as f:
        ann = json.load(f)

        roi_x = ann['x']
        roi_y = ann['y']
        width = ann['w']
        height = ann['h']

        pos_list = ann['pos_list']
        pos_nums = len(pos_list)

    gt_bboxes = np.zeros((pos_nums, 4), dtype=np.float32)

    for idx, pos in enumerate(pos_list):
        x1 = max(pos['x'] - roi_x, 0)
        y1 = max(pos['y'] - roi_y, 0)
        x2 = min(x1 + pos['w'], width)
        y2 = min(y1 + pos['h'], height)

        gt_bboxes[idx, :] = [x1, y1, x2, y2]

    return gt_bboxes


def calc_split_num(image_shape, patch_size, strides):
    # strides = [cfg.patch_size[0]//2, cfg.patch_size[1]//2]
    # x_num = (image_shape[0] - cfg.patch_size[0]) // strides[0] + 2
    # y_num = (image_shape[1] - cfg.patch_size[1]) // strides[1] + 2
    x_num = math.ceil((image_shape[0] - patch_size[0]) / strides[0]) + 1
    y_num = math.ceil((image_shape[1] - patch_size[1]) / strides[1]) + 1
    return x_num, y_num


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone'):
            if cfg.model.neck.rfp_backbone.get('pretrained'):
                cfg.model.neck.rfp_backbone.pretrained = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_slice_test(model, data_loader, cfg, args.strides, args.patch_size, args.iou_thr)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_slice_test(model, data_loader, cfg, args.strides, args.patch_size, args.iou_thr, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))



def single_gpu_slice_test(model, data_loader, cfg, strides, patch_size, iou_thr):

    # # build the data pipeline
    test_pipeline = cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    device = next(model.parameters()).device  # model device

    # image_names = os.listdir(args.image_dir)
    dataset = data_loader.dataset
    print("Begin to predict mask: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    outputs = []
    prog_bar = mmcv.ProgressBar(len(dataset))
    # for image_name in image_names:
    for _, data in enumerate(data_loader):
        image_name = data['img_metas'][0].data[0][0]['ori_filename']
        image_path = os.path.join(cfg.data.test['img_prefix'], image_name)
        img = mmcv.imread(image_path)

        height, width, _ = img.shape
        image_shape = (width, height)
        x_num, y_num = calc_split_num(image_shape, patch_size, strides)

        # print(image_name)
        results = []
        for i in range(x_num):
            for j in range(y_num):
                x = strides[0] * i if i < x_num - 1 else image_shape[0] - patch_size[0]
                y = strides[1] * j if j < y_num - 1 else image_shape[1] - patch_size[1]

                crop_img = img[y:y + patch_size[1], x:x + patch_size[0], :].copy()
                data_crop = dict(filename=image_name,
                            ori_filename=image_name,
                            img=crop_img,
                            img_shape=crop_img.shape,
                            ori_shape=img.shape)
                data_crop = test_pipeline(data_crop)
                data_crop = scatter(collate([data_crop], samples_per_gpu=1), [device])[0]
                # forward the model
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data_crop)[0]

                for _i in range(len(result)):
                    result[_i][:, 0] += x
                    result[_i][:, 1] += y
                    result[_i][:, 2] += x
                    result[_i][:, 3] += y

                if not results:
                    results.extend(result)
                else:
                    for _i in range(len(results)):
                        results[_i] = np.concatenate((results[_i],result[_i]), axis=0)

        # out_file = os.path.join('../submit/', image_name)
        # vis_img = show_result_pyplot(model,img,
        #                       results,
        #                       # model.CLASSES,
        #                       score_thr=0.05,
        #                       wait_time=0)
        #                       # show=False,
        #                       # out_file=None)

        # ann_file = os.path.join(cfg.data.test['ann_file'], image_name.replace('png', 'json'))
        # gt_bboxes = load_annotation(ann_file)
        # for gt_bbox in gt_bboxes:
        #     xmin, ymin, xmax, ymax = gt_bbox
        #     cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        # cv2.imwrite(out_file, vis_img)

        # nms
        for _i in range(len(results)):
            results[_i], _ = nms(copy.deepcopy(results[_i][:,:4]), copy.deepcopy(results[_i][:,4]), iou_threshold=iou_thr)
        outputs.append(results)
        # batch_size = len(results)
        for _ in range(1):
            prog_bar.update()

        # import pdb;pdb.set_trace()
        # result_all = result
        # result_all = np.concatenate(result_all, axis=0)
        # nms_result, _ = nms(result_all, 0.5, device_id=args.device)
    return outputs


def multi_gpu_slice_test(model, data_loader, cfg, strides, patch_size, iou_thr, tmpdir=None, gpu_collect=False):

    # # build the data pipeline
    test_pipeline = cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # device = next(model.parameters()).device  # model device
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    outputs = []
    for _, data in enumerate(data_loader):

        image_name = data['img_metas'][0].data[0][0]['ori_filename']
        image_path = os.path.join(cfg.data.test['img_prefix'], image_name)
        img = mmcv.imread(image_path)

        height, width, _ = img.shape
        image_shape = (width, height)
        x_num, y_num = calc_split_num(image_shape, patch_size, strides)

        # print(image_name)
        results = []
        for i in range(x_num):
            for j in range(y_num):
                x = strides[0] * i if i < x_num - 1 else image_shape[0] - patch_size[0]
                y = strides[1] * j if j < y_num - 1 else image_shape[1] - patch_size[1]

                crop_img = img[y:y + patch_size[1], x:x + patch_size[0], :].copy()
                data_crop = dict(filename=image_name,
                            ori_filename=image_name,
                            img=crop_img,
                            img_shape=crop_img.shape,
                            ori_shape=img.shape)
                data_crop = test_pipeline(data_crop)
                data_crop = scatter(collate([data_crop], samples_per_gpu=1), [next(model.parameters()).device])[0]
                # forward the model
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data_crop)[0]

                for _i in range(len(result)):
                    result[_i][:, 0] += x
                    result[_i][:, 1] += y
                    result[_i][:, 2] += x
                    result[_i][:, 3] += y

                if not results:
                    results.extend(result)
                else:
                    for _i in range(len(results)):
                        results[_i] = np.concatenate((results[_i], result[_i]), axis=0)

        # nms
        for _i in range(len(results)):
            results[_i], _ = nms(copy.deepcopy(results[_i][:,:4]), copy.deepcopy(results[_i][:,4]), iou_threshold=iou_thr)

        outputs.append(results)

        if rank == 0:
            batch_size = 1#len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        outputs = collect_results_gpu(outputs, len(dataset))
    else:
        outputs = collect_results_cpu(outputs, len(dataset), tmpdir)
    return outputs


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


if __name__ == '__main__':
    main()