# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import decord
import mmcv
import numpy as np
import os
import re
import os.path as osp
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from tqdm import tqdm

from PIL import Image
from numpy import asarray

import pyskl  # noqa: F401
from pyskl.smp import mrlines

try:
    import mmdet  # noqa: F401
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose  # noqa: F401
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')


pyskl_root = osp.dirname(pyskl.__path__[0])
default_det_config = f'{pyskl_root}/demo/faster_rcnn_r50_fpn_1x_coco-person.py'
default_det_ckpt = (
    'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco-person/'
    'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth')
default_pose_config = f'{pyskl_root}/demo/hrnet_w32_coco_256x192.py'
default_pose_ckpt = (
    'https://download.openmmlab.com/mmpose/top_down/hrnet/'
    'hrnet_w32_coco_256x192-c78dce93_20200708.pth')


def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]

def extract_frame_jpg(video_path):
    frames = next(os.walk(video_path))[2]

    image = Image.open(video_path + "/" + frames[0])
    img_shape = asarray(image).shape
    # data = np.zeros((len(frames),img_shape[0],img_shape[1],img_shape[2]))
    data = []
    for index,frame_name in enumerate(frames):
        image = Image.open(video_path + "/" + frame_name)
        #data[index,:,:,:] = asarray(image)
        data.append(asarray(image))
    return data


def detection_inference(model, frames):
    results = []
    for index, frame in enumerate(frames):
        #test a single image and return detection result (frame can be str/ndarray)
        result = inference_detector(model, frame)
        results.append(result)
    return results


def pose_inference(anno_in, model, frames, det_results, compress=False):
    anno = cp.deepcopy(anno_in)
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    anno['total_frames'] = total_frames
    anno['num_person_raw'] = num_person

    if compress:
        kp, frame_inds = [], []
        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Align input format
            d = [dict(bbox=x) for x in list(d)]
            pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
            for j, item in enumerate(pose):
                kp.append(item['keypoints'])
                frame_inds.append(i)
        anno['keypoint'] = np.stack(kp).astype(np.float16)
        anno['frame_inds'] = np.array(frame_inds, dtype=np.int16)
    else:
        kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)
        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Align input format
            d = [dict(bbox=x) for x in list(d)]
            pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
            for j, item in enumerate(pose):
                kp[j, i] = item['keypoints']
        anno['keypoint'] = kp[..., :2].astype(np.float16)
        anno['keypoint_score'] = kp[..., 2].astype(np.float16)
    return anno


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    # * Both mmdet and mmpose should be installed from source
    # parser.add_argument('--mmdet-root', type=str, default=default_mmdet_root)
    # parser.add_argument('--mmpose-root', type=str, default=default_mmpose_root)
    parser.add_argument('--det-config', type=str, default=default_det_config)
    parser.add_argument('--det-ckpt', type=str, default=default_det_ckpt)
    parser.add_argument('--pose-config', type=str, default=default_pose_config)
    parser.add_argument('--pose-ckpt', type=str, default=default_pose_ckpt)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.35)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--video-list', type=str, help='the list of source videos')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='tmp')
    parser.add_argument('--local_rank', type=int, default=0)
    # * When non-dist is set, will only use 1 GPU
    parser.add_argument('--non-dist', action='store_true', help='whether to use distributed skeleton extraction')
    parser.add_argument('--compress', action='store_true', help='whether to do K400-style compression')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args

def find_players(anno,det_results,video_folder):
    video = anno["filename"].split("/")[-2]
    with open(video_folder+video+'/annotations.txt') as f:
        lines = f.readlines()
    for j in range(len(lines)):
        line = lines[j]
        if(anno["frame_dir"]+".jpg" == line.split(" ")[0]):
            #remove non players from the frame
            people = det_results[20]
            player = np.zeros(len(people))
            for index in range(len(people)):
                person = people[index]
                x = (person[0]+person[2])/2
                y = (person[1]+person[3])/2
                bounding_boxes = re.findall(r'\b\d+\b', line)
                for l in range(0,int((len(bounding_boxes)-1)/4)):
                    bounding_box_x = [int(bounding_boxes[1+4*l]),int(bounding_boxes[1+4*l])+int(bounding_boxes[3+4*l])]
                    bounding_box_y = [int(bounding_boxes[2+4*l]),int(bounding_boxes[2+4*l])+int(bounding_boxes[4+4*l])]
                    if((x>bounding_box_x[0]) and (x<bounding_box_x[1])):
                        if((y>bounding_box_y[0]) and (y<bounding_box_y[1])):
                            player[index] = 1
            true_players = [true_players for true_players, x in enumerate(player) if x] 
            return true_players
    
    print("Annotations not found!")
    return np.zeros(len(det_results[20]))


def main():
    # we read the args that are called with the function
    # --video-list: input video list with each line: "xxx.mp4 label" ('label' is an integer (category index)) (the label is not mandatory)
    # --out: name of the output file
    print("test1")
    args = parse_args()
    #verification that the output format is .pkl
    assert args.out.endswith('.pkl')

    #we remove the empty lines at the end of the video list arg
    lines = mrlines(args.video_list)
    lines = [x.split() for x in lines]
    print("test2")
    # * We set 'frame_dir' as the base name (w/o. suffix) of each video
    assert len(lines[0]) in [1, 2]
    if len(lines[0]) == 1:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0]) for x in lines]
    else:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0], label=int(x[1])) for x in lines]
    print("test3")
    #allocate the task between the different GPUs. Each GPU will take care of 1 part ("my_part")
    if args.non_dist:
        my_part = annos
        os.makedirs(args.tmpdir, exist_ok=True)
    else:
        init_dist('pytorch', backend='nccl')
        rank, world_size = get_dist_info()
        if rank == 0:
            os.makedirs(args.tmpdir, exist_ok=True)
        dist.barrier()
        my_part = annos[rank::world_size]
    print("test4")
    # get model from checkpoint
    # det_model = init_detector(args.det_config, args.det_ckpt, 'cuda')
    det_model = init_detector(args.det_config, args.det_ckpt, 'cuda') #cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, ve, ort, mlc, xla, lazy, vulkan, meta, hpu
    assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    #pose_model = init_pose_model(args.pose_config, args.pose_ckpt, 'cuda')
    pose_model = init_pose_model(args.pose_config, args.pose_ckpt, 'cuda') #cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, ve, ort, mlc, xla, lazy, vulkan, meta, hpu
    print("test5")
    results = []
    #loop for each video
    for anno in tqdm(my_part):
    # for i in tqdm(range(1,10)): #898
    #     print(i)
    #     anno = my_part[i]
        video_path_args = anno["filename"].split("/")[1:-2]
        video_folder = "/"
        for i in (video_path_args):
            video_folder = video_folder + i + "/"

        frames = extract_frame_jpg(anno['filename'])

        # make prediction for each frames of the video 
        det_results = detection_inference(det_model, frames)
        #print("finished_detection_inference")

        # * Get detection results for human
        det_results = [x[0] for x in det_results]
        for j, res in enumerate(det_results):
            # * filter boxes with small scores
            res = res[res[:, 4] >= args.det_score_thr]
            # * filter boxes with small areas
            box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
            assert np.all(box_areas >= 0)
            res = res[box_areas >= args.det_area_thr]

            det_results[j] = res
   

        shape = frames[0].shape[:2]
        anno['img_shape'] = shape
        
        anno["video"] = anno["filename"].split("/")[-2]
        anno = pose_inference(anno, pose_model, frames, det_results, compress=args.compress)
        anno.pop('filename')
        results.append(anno)

    if args.non_dist:
        mmcv.dump(results, args.out)
    else:
        mmcv.dump(results, osp.join(args.tmpdir, f'part_{rank}.pkl'))
        dist.barrier()

        if rank == 0:
            parts = [mmcv.load(osp.join(args.tmpdir, f'part_{i}.pkl')) for i in range(world_size)]
            rem = len(annos) % world_size
            if rem:
                for i in range(rem, world_size):
                    parts[i].append(None)

            ordered_results = []
            for res in zip(*parts):
                ordered_results.extend(list(res))
            ordered_results = ordered_results[:len(annos)]
            mmcv.dump(ordered_results, args.out)


if __name__ == '__main__':
    main()