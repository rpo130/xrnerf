import os
import torch
import numpy as np
import imageio 
import json
from .load_blender import pose_spherical
import random
from scipy.spatial.transform import Rotation as R

def gen_split(imgs, testskip):
    i_train = []
    i_val = [] 
    i_test = []

    for i in range(len(imgs)):
        b = random.randint(0,9)
        if b >=0 and b<9:
            i_train.append(i)
        else:
            i_test.append(i)

    if len(i_train) == 0:
        i_train.append(random.randint(0, len(imgs)-1))  
    if len(i_test) == 0:
        i_test.append(random.randint(0, len(imgs)-1))

    if testskip == 0:
        skip = 1
    else:
        skip = testskip

    i_test = i_test[::skip]
    i_val = i_test

    i_train = sorted(i_train)
    i_val = sorted(i_val)
    i_test = sorted(i_test)
    return [i_train, i_val, i_test]

def gen_split_seq(imgs, test_skip=8):
    i_train = []
    i_val = [] 
    i_test = []

    i_test = np.arange(len(imgs))[::test_skip]
    i_val = i_test
    i_train = np.array([i for i in np.arange(len(imgs)) if
                    (i not in i_test and i not in i_val)])

    return [i_train, i_val, i_test]


def load_avt_data(basedir, test_skip=1, need_fix_pose=True):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    all_imgs = []
    all_poses = []
    imgs = []
    poses = []
        
    for frame in meta['frames'][::1]:
        file_path = frame['file_path']
        if '.png' not in file_path:
            file_path = file_path + '.png'
        fname = os.path.join(basedir, file_path)
        img = imageio.imread(fname)
        imgs.append(img)
        T_cam_to_world = np.array(frame['transform_matrix'])
        if need_fix_pose:
            #fix poses
            T_cam_face_to_world = T_cam_to_world

            T_img_to_cam_face = np.eye(4) 
            T_img_to_cam_face[:3, :3] = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
            T_cam_to_world = T_cam_face_to_world @ T_img_to_cam_face

        poses.append(T_cam_to_world)
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    poses = np.array(poses).astype(np.float32)
    all_imgs.append(imgs)
    all_poses.append(poses)
        
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]

    fx = meta['fx']
    fy = meta['fy']
    cx = meta['cx']
    cy = meta['cy']

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]])

    focal = fx

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    i_split = gen_split_seq(imgs, test_skip)

    return imgs, poses, render_poses, [H, W, focal], i_split, K
