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

import sys
import os

import os.path as osp

import time
import yaml
import torch

import smplx
from click.core import batch

from utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False

import pickle

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

    video_ids_folder = args["data_subset"]
    with open(video_ids_folder, "rb") as f:
        video_ids = pickle.load(f)

    keypoints_folder = args['keypoints_folder']
    with open(keypoints_folder, 'rb') as f:
        keypoints_all = pickle.load(f)

    img_folder = args.pop('img_folder', 'images')
    dataset_obj = create_dataset(img_folder=img_folder, video_id=video_ids[0], **args)

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

        img = data['img']
        fn = data['fn']

        import numpy as np
        keypoints_coco = np.array(keypoints_all[video_ids[idx]]["keypoints"])
        keypoints_coco = torch.tensor(keypoints_coco)

        mapping = np.array([0, 6, 6, 8, 10, 5, 7, 9, 12, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3, 17, 18, 19, 21, 21, 22],
                           dtype=np.int32)
        keypoints_openpose = torch.zeros([keypoints_coco.shape[0], 118, 3])
        for k, indice in enumerate(mapping):
            keypoints_openpose[:, k, :] = keypoints_coco[:, indice, :]
        keypoints_openpose[:, 1, 0] = (keypoints_coco[:, 5, 0] + keypoints_coco[:, 6, 0]) / 2
        keypoints_openpose[:, 8, 0] = (keypoints_coco[:, 12, 0] + keypoints_coco[:, 11, 0]) / 2
        keypoints_openpose[:, 25:67, :] = keypoints_coco[:, 91:, :]
        keypoints_openpose[:, 67:, :] = keypoints_coco[:, 40:91, :]
        keypoints_openpose = keypoints_openpose.to(device=device)
        # if dataset_to_fitting == 'Phoenix-2014T':
        #     keypoints_openpose[:, :, 0] *= 1.24

        len_shoulder = keypoints_openpose[:, 5, 0] - keypoints_openpose[:, 2, 0]
        len_waist = len_shoulder / 1.7
        keypoints_openpose[:, 8, 0] = keypoints_openpose[:, 1, 0]
        keypoints_openpose[:, 8, 1] = keypoints_openpose[:, 1, 1] + 1.5 * len_shoulder
        keypoints_openpose[:, 9, 0] = keypoints_openpose[:, 8, 0] - 0.5 * len_waist
        keypoints_openpose[:, 9, 1] = keypoints_openpose[:, 8, 1]
        keypoints_openpose[:, 12, 0] = keypoints_openpose[:, 8, 0] + 0.5 * len_waist
        keypoints_openpose[:, 12, 1] = keypoints_openpose[:, 8, 1]
        keypoints_openpose[:, 10, 0] = keypoints_openpose[:, 9, 0]
        keypoints_openpose[:, 10, 1] = keypoints_openpose[:, 9, 1] + 2. * len_waist
        keypoints_openpose[:, 11, 0] = keypoints_openpose[:, 9, 0]
        keypoints_openpose[:, 11, 1] = keypoints_openpose[:, 9, 1] + 4. * len_waist
        keypoints_openpose[:, 13, 0] = keypoints_openpose[:, 12, 0]
        keypoints_openpose[:, 13, 1] = keypoints_openpose[:, 12, 1] + 2. * len_waist
        keypoints_openpose[:, 14, 0] = keypoints_openpose[:, 12, 0]
        keypoints_openpose[:, 14, 1] = keypoints_openpose[:, 12, 1] + 4. * len_waist
        keypoints_openpose[:, 8:15, 2] = 0.65

        keypoints = keypoints_openpose
        # torch.save(keypoints, "/home/hhm/pose_process/keypoints_smplx.pt")
        if len(keypoints.shape) == 2:
            keypoints.unsqueeze_(dim=0)
        print('Processing: {}'.format(data['img_path']))

        curr_result_folder = osp.join(result_folder, fn)
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


        for person_id in range(int(keypoints.shape[0] / batch_size)+1):
            # if person_id >= max_persons and max_persons > 0:
            #     continue
            if batch_size * person_id >= keypoints_openpose.shape[0]:
                continue
            curr_result_fn = osp.join(curr_result_folder,
                                      '{:03d}.pkl'.format(person_id))
            curr_mesh_fn = osp.join(curr_mesh_folder,
                                    '{:03d}.obj'.format(person_id))

            curr_img_folder = osp.join(output_folder, 'images', fn,
                                       '{:03d}'.format(person_id))
            if not osp.exists(curr_img_folder):
                os.makedirs(curr_img_folder)

            if gender_lbl_type != 'none':
                if gender_lbl_type == 'pd' and 'gender_pd' in data:
                    gender = data['gender_pd'][person_id]
                if gender_lbl_type == 'gt' and 'gender_gt' in data:
                    gender = data['gender_gt'][person_id]
            else:
                gender = input_gender

            if gender == 'neutral':
                body_model = neutral_model
            elif gender == 'female':
                body_model = female_model
            elif gender == 'male':
                body_model = male_model

            out_img_fn = osp.join(curr_img_folder, 'output.png')

            ret, joints_smooth_re, camera_transl_re, camera_orient_re, betas_fix_re = fit_single_frame(img, person_id,keypoints[person_id*batch_size:person_id*batch_size+batch_size],
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
            print("%d finished"% (person_id))
            joints_smooth = joints_smooth_re.clone()
            camera_transl = camera_transl_re.clone()
            camera_orient = camera_orient_re.clone()
            betas_fix = betas_fix_re.clone()
            result["video_info"].extend(ret)
            # result["video_info"].append(ret)
        with open(curr_result_fn, 'wb') as result_file:
            pickle.dump(result, result_file, protocol=2)


    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the data took: {}'.format(time_msg))


if __name__ == "__main__":
    args = parse_config()
    main(**args)
