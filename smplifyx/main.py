# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import re
import sys
import os

import os.path as osp

import time

import cv2
import yaml
import torch

import smplx

from utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False

import pickle
import json

def main(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)
    # Store the arguments for the current experiment
    conf_fn = osp.join(output_folder, 'conf.yaml')
    with open(conf_fn, 'w') as conf_file:
        yaml.dump(args, conf_file)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float64
    else:
        print('Unknown float type {}, exiting!'.format(float_dtype))
        sys.exit(-1)

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        print('CUDA is not available, exiting!')
        sys.exit(-1)

    keypoints_folder = args['keypoints_folder']
    # if os.path.isdir(keypoints_folder):
    #     file_list = os.listdir(keypoints_folder)
    #     file_list = sorted(file_list, key=lambda file_name: int(re.search(r'_(\d+)_keypoints', file_name).group(1)))
    #     import numpy as np
    #     keypoints_all = np.zeros((len(file_list), 118,  3))
    #     for i, file_name in enumerate(file_list):
    #         with open(os.path.join(keypoints_folder, file_name), 'rb') as f:
    #             keypoints_dict = json.load(f)
    #             keypoints_person = keypoints_dict['people'][0]
    #             keypoints = []
    #             keypoints_body = keypoints_person['pose_keypoints_2d']
    #             keypoints_face = keypoints_person['face_keypoints_2d'][51:204]
    #             keypoints_lhand = keypoints_person['hand_left_keypoints_2d']
    #             keypoints_rhand = keypoints_person['hand_right_keypoints_2d']
    #             keypoints.extend(keypoints_body)
    #             keypoints.extend(keypoints_lhand)
    #             keypoints.extend(keypoints_rhand)
    #             keypoints.extend(keypoints_face)
    #             keypoints = np.array(keypoints).reshape(-1, 3)
    #             keypoints_all[i] = keypoints
    # else:
    #     with open(keypoints_folder, 'rb') as f:
    #         keypoints_all = pickle.load(f)

    dataset_obj = create_dataset(**args)

    start = time.time()

    input_gender = args.pop('gender', 'neutral')
    gender_lbl_type = args.pop('gender_lbl_type', 'none')
    max_persons = args.pop('max_persons', -1)

    float_dtype = args.get('float_dtype', 'float32')
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    joint_mapper = JointMapper(dataset_obj.get_model2data())

    model_params = dict(model_path=args.get('model_folder'),
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args.get('use_vposer'),
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=False,
                        dtype=dtype,
                        **args)

    male_model = smplx.create(gender='male', **model_params)
    # SMPL-H has no gender-neutral model
    if args.get('model_type') != 'smplh':
        neutral_model = smplx.create(gender='neutral', **model_params)
    female_model = smplx.create(gender='female', **model_params)

    # Create the camera object
    focal_length = args.get('focal_length')
    camera = create_camera(focal_length_x=focal_length,
                           focal_length_y=focal_length,
                           dtype=dtype,
                           **args)

    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False

    use_hands = args.get('use_hands', True)
    use_face = args.get('use_face', True)

    body_pose_prior = create_prior(
        prior_type=args.get('body_prior_type'),
        dtype=dtype,
        **args)

    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args.get('jaw_prior_type'),
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args.get('expr_prior_type', 'l2'),
            dtype=dtype, **args)

    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(
            prior_type=args.get('left_hand_prior_type'),
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)

        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(
            prior_type=args.get('right_hand_prior_type'),
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)

    shape_prior = create_prior(
        prior_type=args.get('shape_prior_type', 'l2'),
        dtype=dtype, **args)

    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')

        camera = camera.to(device=device)
        female_model = female_model.to(device=device)
        male_model = male_model.to(device=device)
        if args.get('model_type') != 'smplh':
            neutral_model = neutral_model.to(device=device)
        body_pose_prior = body_pose_prior.to(device=device)
        angle_prior = angle_prior.to(device=device)
        shape_prior = shape_prior.to(device=device)
        if use_face:
            expr_prior = expr_prior.to(device=device)
            jaw_prior = jaw_prior.to(device=device)
        if use_hands:
            left_hand_prior = left_hand_prior.to(device=device)
            right_hand_prior = right_hand_prior.to(device=device)
    else:
        device = torch.device('cpu')

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device,
                                                       dtype=dtype)
    # Add a fake batch dimension for broadcasting
    batch_size = args.get('batch_size', 1)
    if batch_size == 1:
        joint_weights.unsqueeze_(dim=0)

    for idx, data in enumerate(dataset_obj):

        video_cap = data["video_cap"]
        keypoints_path_list = data["keypoints_path_list"]
        # img = data['img']
        fn = data["video_name"]

        import numpy as np
        keypoints = np.zeros((len(keypoints_path_list), 118, 3))
        for i, file_name in enumerate(keypoints_path_list):
            with open(os.path.join(keypoints_folder, file_name), 'rb') as f:
                keypoints_dict = json.load(f)
                keypoints_person = keypoints_dict['people'][0]
                keypoints_per_frame = []
                keypoints_body = keypoints_person['pose_keypoints_2d']
                keypoints_face = keypoints_person['face_keypoints_2d'][51:204]
                keypoints_lhand = keypoints_person['hand_left_keypoints_2d']
                keypoints_rhand = keypoints_person['hand_right_keypoints_2d']
                keypoints_per_frame .extend(keypoints_body)
                keypoints_per_frame .extend(keypoints_lhand)
                keypoints_per_frame .extend(keypoints_rhand)
                keypoints_per_frame .extend(keypoints_face)
                keypoints_per_frame  = np.array(keypoints_per_frame).reshape(-1, 3)
                keypoints[i] = keypoints_per_frame

        # torch.save(keypoints, "/home/hhm/pose_process/keypoints_smplx.pt")
        if len(keypoints.shape) == 2:
            keypoints = np.expand_dims(keypoints, axis=0)
            # keypoints.unsqueeze_(dim=0)
        print('Processing: {}'.format(fn))

        curr_result_folder = osp.join(result_folder, fn).replace(".mp4", "")
        if not osp.exists(curr_result_folder):
            os.makedirs(curr_result_folder)
        curr_mesh_folder = osp.join(mesh_folder, fn)
        if not osp.exists(curr_mesh_folder):
            os.makedirs(curr_mesh_folder)
        result = {}
        result["video_info"] = []
        # betas_fix = torch.zeros([batch_size, 10])
        # camera_transl = torch.zeros([batch_size, 3])
        # camera_orient = torch.zeros([batch_size, 3])
        # joints_smooth = torch.zeros([batch_size,21,3])
        betas_fix = torch.zeros([batch_size, 10])
        camera_transl = torch.zeros([3])
        camera_orient = torch.zeros([3])
        joints_smooth = torch.zeros([batch_size,21,3])
        if use_cuda and torch.cuda.is_available():
            camera_transl=camera_transl.to(device=device)
            camera_orient=camera_orient.to(device=device)
            betas_fix=betas_fix.to(device=device)
            joints_smooth=joints_smooth.to(device=device)

        pose_dict_list = []
        for frame_idx in range(int(len(keypoints_path_list) / batch_size)+1):
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, img = video_cap.read()
            if batch_size * frame_idx >= keypoints.shape[0]:
                continue
            curr_result_fn = osp.join(curr_result_folder,
                                      '{:04d}.pkl'.format(frame_idx))
            curr_mesh_fn = osp.join(curr_mesh_folder,
                                    '{:03d}.obj'.format(frame_idx))

            curr_img_folder = osp.join(output_folder, 'images', fn,
                                       '{:03d}'.format(frame_idx))
            if not osp.exists(curr_img_folder):
                os.makedirs(curr_img_folder)

            if gender_lbl_type != 'none':
                if gender_lbl_type == 'pd' and 'gender_pd' in data:
                    gender = data['gender_pd'][frame_idx]
                if gender_lbl_type == 'gt' and 'gender_gt' in data:
                    gender = data['gender_gt'][frame_idx]
            else:
                gender = input_gender

            if gender == 'neutral':
                body_model = neutral_model
            elif gender == 'female':
                body_model = female_model
            elif gender == 'male':
                body_model = male_model

            out_img_fn = osp.join(curr_img_folder, 'output.png')

            ret, joints_smooth_re, camera_transl_re, camera_orient_re, betas_fix_re = fit_single_frame(img, frame_idx,keypoints[frame_idx*batch_size:frame_idx*batch_size+batch_size],
                             body_model=body_model,
                             camera=camera,
                             joint_weights=joint_weights,
                             dtype=dtype,
                             output_folder=output_folder,
                             result_folder=curr_result_folder,
                             out_img_fn=out_img_fn,
                             result_fn=curr_result_fn,
                             mesh_fn=curr_mesh_fn,
                             shape_prior=shape_prior,
                             expr_prior=expr_prior,
                             body_pose_prior=body_pose_prior,
                             left_hand_prior=left_hand_prior,
                             right_hand_prior=right_hand_prior,
                             jaw_prior=jaw_prior,
                             angle_prior=angle_prior,
                             betas_fix=betas_fix,
                             camera_transl=camera_transl,
                             camera_orient=camera_orient,
                             joints_smooth=joints_smooth,
                             **args)
            print("%d finished"% (frame_idx))
            joints_smooth = joints_smooth_re.clone()
            camera_transl = camera_transl_re.clone()
            camera_orient = camera_orient_re.clone()
            betas_fix = betas_fix_re.clone()
            result["video_info"].extend(ret)
            # result["video_info"].append(ret)
        # save the file as the render format
        with open(curr_result_fn, 'wb') as result_file:
            video_info = result["video_info"]
            frame_number = len(video_info)
            betas = np.zeros((frame_number, 10), dtype=np.float32)
            global_oris = np.zeros((frame_number, 3), dtype=np.float32)
            body_rots = np.zeros((frame_number, 63), dtype=np.float32)
            left_hand_rots = np.zeros((frame_number, 45), dtype=np.float32)
            right_hand_rots = np.zeros((frame_number, 45), dtype=np.float32)
            for i, frame in enumerate(video_info):
                betas[i] = frame["betas"]
                global_oris[i] = frame["global_orient"]
                body_rots[i] = frame["body_pose_rot"]
                left_hand_rots[i] = frame["left_hand_pose_rot"]
                right_hand_rots[i] = frame["right_hand_pose_rot"]
            pose_dict = {}
            pose_dict["poses"] = []
            for i in range(frame_number):
                action_dict = {}
                action_dict["smplx_body_pose"] = body_rots[i]
                action_dict["smplx_lhand_pose"] = left_hand_rots[i]
                action_dict["smplx_rhand_pose"] = right_hand_rots[i]
                action_dict["smplx_root_pose"] = global_oris[i]
                action_dict["smplx_jaw_pose"] = np.zeros(3)
                pose_dict["poses"].append(action_dict)
            pose_dict_list.append(pose_dict)
        with open(curr_result_fn, 'wb') as result_file:
            pickle.dump(pose_dict_list, result_file, protocol=2)
        video_cap.release()
            # pickle.dump(result, result_file, protocol=2)


    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    args = parse_config()
    main(**args)
