import torch
import cv2
import csv
import torchvision.transforms.functional as tf
from skimage.io import imread
from skimage.transform import estimate_transform, warp
from torchvision.transforms import Normalize
import numpy as np
import pickle
from pathlib import Path
import os.path
import random
from PIL import Image
from ..utils.load_mats import load_lm3d
import torchvision.transforms as transforms
from ..utils.preprocess import *

def get_transform(grayscale=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


def get_affine_mat(preprocess='shift_scale_rot_flip', size=(224, 224)):
    shift_x, shift_y, scale, rot_angle, flip = 0., 0., 1., 0., False
    w, h = size
    scale_delta = 0.1
    rot_angle = 10

    if 'shift' in preprocess:
        shift_pixs = int(10)
        shift_x = random.randint(-shift_pixs, shift_pixs)
        shift_y = random.randint(-shift_pixs, shift_pixs)
    if 'scale' in preprocess:
        scale = 1 + scale_delta * (2 * random.random() - 1)
    if 'rot' in preprocess:
        rot_angle = rot_angle * (2 * random.random() - 1)
        rot_rad = -rot_angle * np.pi / 180
    if 'flip' in preprocess:
        flip = random.random() > 0.5

    shift_to_origin = np.array([1, 0, -w // 2, 0, 1, -h // 2, 0, 0, 1]).reshape([3, 3])
    flip_mat = np.array([-1 if flip else 1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape([3, 3])
    shift_mat = np.array([1, 0, shift_x, 0, 1, shift_y, 0, 0, 1]).reshape([3, 3])
    rot_mat = np.array([np.cos(rot_rad), np.sin(rot_rad), 0, -np.sin(rot_rad), np.cos(rot_rad), 0, 0, 0, 1]).reshape(
        [3, 3])
    scale_mat = np.array([scale, 0, 0, 0, scale, 0, 0, 0, 1]).reshape([3, 3])
    shift_to_center = np.array([1, 0, w // 2, 0, 1, h // 2, 0, 0, 1]).reshape([3, 3])

    affine = shift_to_center @ scale_mat @ rot_mat @ shift_mat @ flip_mat @ shift_to_origin
    affine_inv = np.linalg.inv(affine)
    return affine, affine_inv, flip


def apply_img_affine(img, affine_inv, method=Image.BICUBIC):
    return img.transform(img.size, Image.AFFINE, data=affine_inv.flatten()[:6], resample=Image.BICUBIC)


def apply_lm_affine(landmark, affine, flip, size):
    _, h = size
    lm = landmark.copy()
    lm[:, 1] = h - 1 - lm[:, 1]
    lm = np.concatenate((lm, np.ones([lm.shape[0], 1])), -1)
    lm = lm @ np.transpose(affine)
    lm[:, :2] = lm[:, :2] / lm[:, 2:]
    lm = lm[:, :2]
    lm[:, 1] = h - 1 - lm[:, 1]
    if flip:
        lm_ = lm.copy()
        lm_[:17] = lm[16::-1]
        lm_[17:22] = lm[26:21:-1]
        lm_[22:27] = lm[21:16:-1]
        lm_[31:36] = lm[35:30:-1]
        lm_[36:40] = lm[45:41:-1]
        lm_[40:42] = lm[47:45:-1]
        lm_[42:46] = lm[39:35:-1]
        lm_[46:48] = lm[41:39:-1]
        lm_[48:55] = lm[54:47:-1]
        lm_[55:60] = lm[59:54:-1]
        lm_[60:65] = lm[64:59:-1]
        lm_[65:68] = lm[67:64:-1]
        lm = lm_
    return lm


def parse_label(label):
    return torch.tensor(np.array(label).astype(np.float32))


left_eye = [37, 38, 40, 41]
right_eye = [43, 44, 46, 47]
nose = [30]
left_mouth = [48]
right_mouth = [54]

class CelebDataset(torch.utils.data.Dataset):

    def  __init__(self, device, train, height, width, scale,trans_scale = 0, landmark_file=False, test_mode=False,
                 occloss_mode=False, is_use_aug=True, bfm_folder='BFM'):
        super(CelebDataset, self)
        self.test_mode = test_mode
        self.train = train
        self.is_occ_mode = occloss_mode
        # Set is_use_aug as True if data augmentation is required during training.
        self.use_aug = is_use_aug
        self.trans_scale = trans_scale
        self.root = 'D:/OE/szakdolgozat/celeba/celeba/img_align_celeba/img_align_celeba/' #'image_root/Data/Dataset/'

        # Get landmark files
        if self.train:
            landmark_filename = 'D:/OE/szakdolgozat/celeba/celeba/train_landmarks.csv' # self.root + '../train_landmarks.csv'
        else:
            landmark_filename = 'D:/OE/szakdolgozat/celeba/celeba/val_landmarks.csv'# self.root + '../val_landmarks.csv'
        if self.test_mode:
            self.train = False
            if landmark_file:
                landmark_filename = landmark_file
            else:
                landmark_filename = 'D:/OE/szakdolgozat/celeba/celeba/test_landmarks.csv'# self.root + '../test_landmarks.csv'
        if 'NoWDataset' in self.root:
            landmark_filename = self.root + 'nowchallenge_landmarks.csv'
        print('Dataset:' + landmark_filename)
        self.landmark_list = list(csv.reader(open(landmark_filename), delimiter=','))
        self.num = len(self.landmark_list)

        self.device = device
        self.scale = scale
        self.width = width
        self.height = height
        self.lm3d_std = load_lm3d(bfm_folder)

    def __len__(self):
        return self.num

    def shuffle(self):
        random.shuffle(self.landmark_list)

    def __getitem__(self, index):

        filename = self.root + self.landmark_list[index][0].replace('//', '/')
        landmark_cpu = [int(x.strip('.0')) for x in self.landmark_list[index][1:]]

        if os.path.exists(filename):
            image = cv2.imread(filename)
            image = cv2.resize(image, (224, 224))
        else:
            print('Image does not exist: ' + filename)

        image = imread(filename) / 255.
        if len(image.shape) < 3:
            image = np.tile(image[:, :, None], 3)


        raw_img = Image.open(filename).convert('RGB')
        _, H = raw_img.size
        raw_lm = np.reshape(np.asarray(landmark_cpu), (-1, 2)).astype(np.float32)
        raw_lm = np.stack([raw_lm[:, 0], H - raw_lm[:, 1]], 1)

        try:
            _, img, lm, msk = align_img(raw_img, raw_lm, self.lm3d_std, mask=None)
        except:
            index += 1
            self.__getitem__(index + 1)

        # Data augmentation is only for training
        aug_flag = self.use_aug and self.train
        if aug_flag:
            img, lm, gt_mask = self._augmentation(img, lm, msk)

        _, H = img.size
        transform = get_transform()

        tform = self.crop(image, lm)

        cropped_image = warp(image, tform.inverse, output_shape=(self.width, self.height))
        cropped_kpt = np.dot(tform.params,np.hstack([raw_lm, np.ones([raw_lm.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

        # normalized kpt
        cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.width * 2 - 1 #image_size

        img_tensor = torch.from_numpy(cropped_image.transpose(2,0,1)).type(dtype = torch.float32) #224,224,3

        lm = np.stack([lm[:, 0], H - lm[:, 1]], 1)
        lm_tensor = torch.from_numpy(cropped_kpt).type(dtype = torch.float32) #224,224,3

        if self.test_mode:
            # Get GT masks for testing
            skin_vis_mask_path = filename.replace('.bmp', '.jpg').replace('.png', '.jpg').replace('.jpg', '_visible_skin_mask.png')
            if os.path.exists(skin_vis_mask_path):
                raw_gtmask = Image.open(skin_vis_mask_path)
                _, gt_mask, _, _ = align_img(raw_gtmask, raw_lm, self.lm3d_std, mask=None)
                gt_mask_tensor = transform(gt_mask)[:1, ...].to(self.device)
            else:
                gt_mask_tensor = False

            data_dict = {
                'image': img_tensor,
                'landmark': lm_tensor,
                'mask': gt_mask_tensor
            }
            return data_dict

        data_dict = {
            'image': img_tensor,
            'landmark': lm_tensor,
            'mask': img_tensor #TODO
        }
        return data_dict

    def _augmentation(self, img, lm, msk=None):
        affine, affine_inv, flip = get_affine_mat()
        img = apply_img_affine(img, affine_inv)
        lm = apply_lm_affine(lm, affine, flip, img.size)
        if msk is not None:
            msk = apply_img_affine(msk, affine_inv, method=Image.BILINEAR)
        return img, lm, msk

    def crop(self, image, kpt):
        left = np.min(kpt[:, 0]);
        right = np.max(kpt[:, 0]);
        top = np.min(kpt[:, 1]);
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]

        size = int(old_size * scale)

        # crop image
        # src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
        DST_PTS = np.array([[0, 0], [0, self.width - 1], [self.height - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform
