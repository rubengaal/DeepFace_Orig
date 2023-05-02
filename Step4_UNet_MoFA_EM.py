#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Tue Feb 23 13:12:25 2021

UNet + MoFA
@author: li0005
"""
import os
import cv2
import numpy
import torch
from datetime import datetime
import math
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import util.util as focus_util
import decalib.utils.util as deca_util
import csv
import util.load_object as lob
import renderer.rendering as ren
import encoder.encoder as enc
import torch.nn.functional as F
from loguru import logger
import numpy as np
import time
import UNet.UNet as unet
import argparse
from datetime import date
import util.advanced_losses as adlosses
from decalib.datasets import build_datasets
from decalib.utils.config import parse_args
from decalib.deca import DECA
from decalib.utils import lossfunc
from decalib.utils.config import cfg
import decalib.utils.config as config
from models import networks
from decalib.trainer import Trainer
from face_toolbox_keras.models.parser import face_parser
import util.load_dataset as load_dataset
import pickle

# hyper-parameters
par = argparse.ArgumentParser(description='MoFA')

par.add_argument('--learning_rate', default=0.1, type=float, help='The learning rate')
par.add_argument('--epochs', default=1, type=int, help='Total epochs')
par.add_argument('--batch_size', default=4, type=int, help='Batch sizes')
par.add_argument('--gpu', default=0, type=int, help='The GPU ID')
par.add_argument('--pretrained_model', default=00, type=int, help='Pretrained model')
par.add_argument('--img_path', type=str, default='image_root/Data/Dataset', help='Root of the training samples')
par.add_argument('--cfg', type=str, default='configs/release_version/deca_coarse.yml')

args = par.parse_args()
cfg = parse_args()
if cfg.cfg_file is not None:
    exp_name = cfg.cfg_file.split('/')[-1].split('.')[0]
    cfg.exp_name = exp_name

GPU_no = args.gpu

dist_weight = {'neighbour': 15, 'dist': 3, 'area': 0.5, 'preserve': 0.25, 'binary': 10}
ct = args.pretrained_model  # load trained mofa model

deca = DECA(cfg)
trainer = Trainer(model=deca, config=cfg)


output_name = 'Deca_UNet'

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

begin_learning_rate = args.learning_rate
decay_step_size = 5000
decay_rate_gamma = 0.99
decay_rate_unet = 0.95
learning_rate_begin = begin_learning_rate * (decay_rate_gamma ** ((300000) // decay_step_size)) * 0.8
mofa_lr_begin = learning_rate_begin * (decay_rate_gamma ** (ct // decay_step_size))
unet_lr_begin = learning_rate_begin * 0.06 * (decay_rate_unet ** (ct // decay_step_size))

ct_begin = ct

today = date.today()
current_path = os.getcwd()
# model_path = current_path + '/basel_3DMM/model2017-1_bfm_nomouth.h5'

image_path = (args.img_path + '/').replace('//', '/')
output_path = current_path + '/DECA_UNet_Save/robustness/' + output_name + '/'
if not os.path.exists(output_path):
    os.makedirs(output_path)

loss_log_path_train = output_path + today.strftime("%b-%d-%Y") + "loss_train.csv"
loss_log_path_test = output_path + today.strftime("%b-%d-%Y") + "loss_test.csv"

weight_log_path_train = output_path + today.strftime("%b-%d-%Y") + "weight_train.pkl"
with open(weight_log_path_train, 'wb') as f:
    pickle.dump(dist_weight, f)

'''------------------
  Load Data & Models
------------------'''
# parameters
batch = args.batch_size
width = 224
height = 224

epoch = args.epochs
test_batch_num = 2

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

render_net = ren.Renderer(32)
net_recog = networks.define_net_recog(net_recog='r50', pretrained_path='models/ms1mv3_arcface_r50_fp16/backbone.pth')
net_recog = net_recog.to(device)

# 3dmm data
'''----------------
Prepare Network and Optimizer
----------------'''

current_path = os.getcwd()
model_path = current_path + '/basel_3DMM/model2017-1_bfm_nomouth.h5'

obj = lob.Object3DMM(model_path, device, is_crop=True)
A = torch.Tensor([[9.06 * 224 / 2, 0, (width - 1) / 2.0, 0, 9.06 * 224 / 2, (height - 1) / 2.0, 0, 0, 1]]).view(-1, 3,
                                                                                                                3).to(
    device)  # intrinsic camera mat
T_ini = torch.Tensor([0, 0, 1000]).to(device)  # camera translation(direction of conversion will be set by flg later)
sh_ini = torch.zeros(3, 9, device=device)  # offset of spherical harmonics coefficient
sh_ini[:, 0] = 0.7 * 2 * math.pi
sh_ini = sh_ini.reshape(-1)

# renderer and encoder and UNet

'''-------------
Network Forward
-------------'''


#################################################################
def proc_mofaunet(batch, images, landmarks, render_mode, train_net=False, occlusion_mode=False, valid_mask=None,
                  image_org=None, is_cutmix_mode=False):
    # valid_mask: 1 indicating unoccluded part of faces, vice versa
    '''
    images: network_input
    landmarks: landmark ground truth
    render_mode: renderer mode
    occlusion mode: use occlusion robust loss or not
    landmark_vmask: landmark valid mask
    valid_mask: use the valid region
    image_org: if is supervised mode, image_org 
    '''

    '''-------------------------------------
    U-Net input: Raster [RGB] + ORG [RGB]
    ----------------------------------------'''

    codedict = deca.encode(images)  # code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
    opdict, visdict = deca.decode(codedict)  # tensor

    lm68 = opdict['landmarks2d']
    raster_mask = opdict['rasterized_masks']
    raster_mask = raster_mask.permute(0, 3, 2, 1)
    raster_image = visdict['rendered_images']
    image_concatenated = torch.cat((raster_image, images), axis=1)
    unet_est_mask = unet_for_mask(image_concatenated)
    valid_loss_mask = raster_mask * unet_est_mask  # TODO: kell ez az unsqueeze? raster_mask.unsqueeze(1) * unet_est_mask

    masked_rec_loss = torch.mean(torch.sum(torch.norm(valid_loss_mask * (images - raster_image), 2, 1)) / torch.clamp(
        torch.sum(raster_mask.unsqueeze(1) * unet_est_mask), min=1))

    bg_unet_loss = torch.mean(torch.sum(raster_mask.unsqueeze(1) * (1 - unet_est_mask), axis=[2, 3]) / torch.clamp(
        torch.sum(raster_mask.unsqueeze(1), axis=[2, 3]), min=1))  # area loss
    mask_binary_loss = torch.zeros([1])
    loss_mofa = torch.zeros([1])
    if train_net == 'unet':
        mask_binary_loss = (0.5 - torch.mean(torch.norm(valid_loss_mask - 0.5, 2, 1)))
        loss_unet = mask_binary_loss * dist_weight['binary'] + bg_unet_loss * dist_weight['area']

    if train_net == False:
        mask_binary_loss = (0.5 - torch.mean(torch.norm(valid_loss_mask - 0.5, 2, 1)))
        loss_test = mask_binary_loss.to(device) * dist_weight['binary'] + masked_rec_loss.to(
            'cpu') * 0.5 + bg_unet_loss.to(device) * dist_weight[
                        'area'] + loss_mofa.to(device)

    I_target_masked = images * valid_loss_mask
    reducedlm = torch.mean(landmarks, -1)
    id_target_masked = net_recog(I_target_masked, landmarks, is_shallow=True)  # landmarks.transpose(1, 2)
    id_target = net_recog(images, landmarks, is_shallow=True)  # landmarks.transpose(1, 2)
    id_reconstruct_masked = net_recog(raster_image * valid_loss_mask, pred_lm=lm68, is_shallow=True)
    I_IM_Per_loss = torch.mean(1 - cos(id_target, id_target_masked))
    IRM_IM_Per_loss = torch.mean(1 - cos(id_reconstruct_masked, id_target_masked))
    if train_net == 'unet':
        loss_unet += I_IM_Per_loss * dist_weight['preserve'] + IRM_IM_Per_loss * dist_weight['dist']
    if train_net == False:
        loss_test += I_IM_Per_loss.cuda() * dist_weight['preserve'] + IRM_IM_Per_loss.cuda() * dist_weight['dist']

    # force it to be binary mask
    loss_mask_neighbor = torch.zeros([1])
    if train_net == 'unet':
        loss_mask_neighbor = adlosses.neighbor_unet_loss(images, valid_loss_mask, raster_image)
        loss_unet += loss_mask_neighbor * dist_weight['neighbour']
        loss = loss_unet
        loss_name='loss_unet'
    if train_net == False:
        loss = loss_test
        loss_name = 'loss_test'
    losses_return = torch.FloatTensor(
        [loss.item(), masked_rec_loss.item(), bg_unet_loss.item(), \
         loss_mask_neighbor.item(), mask_binary_loss.item(),I_IM_Per_loss.item(),IRM_IM_Per_loss.item()] )

    losses_dict = {
        'loss': loss.item(),
        'masked_rec_loss': masked_rec_loss.item(),
        'bg_unet_loss': bg_unet_loss.item(),
        'loss_mask_neighbor': loss_mask_neighbor.item(),
        'mask_binary_loss': mask_binary_loss.item(),
        'I_IM_Per_loss': I_IM_Per_loss.item(),
        'IRM_IM_Per_loss': IRM_IM_Per_loss.item(),
    }

    if train_net == 'unet':
        return loss_unet, losses_return, raster_image, raster_mask, unet_est_mask, valid_loss_mask, losses_dict
    if train_net == False:
        return loss_test, losses_return, raster_image, raster_mask, unet_est_mask, valid_loss_mask, losses_dict


#################################################################

'''-----------------------------------------
load pretrained model and continue training
-----------------------------------------'''

unet_model_path = current_path + '\\MoFA_UNet_Save\\Pretrain_UNet' + '\\00180000_unet.tar'
unet_for_mask = torch.load(unet_model_path, map_location='cuda:{}'.format(focus_util.device_ids[GPU_no]))

print('Loading pre-trained unet: \n' + unet_model_path)

'''----------
Set Optimizer
----------'''
optimizer_unet = optim.Adadelta(unet_for_mask.parameters(), lr=unet_lr_begin)
scheduler_unet = torch.optim.lr_scheduler.StepLR(optimizer_unet, step_size=decay_step_size, gamma=decay_rate_unet)

print('Training ...')
trainer.prepare_data()
iters_every_epoch = int(len(trainer.train_dataset) / trainer.batch_size)
start_epoch = trainer.global_step // iters_every_epoch
for epoch in range(start_epoch, trainer.cfg.train.max_epochs):
    # for step, batch in enumerate(tqdm(self.train_dataloader, desc=f"Epoch: {epoch}/{self.cfg.train.max_epochs}")):
    for step in tqdm(range(iters_every_epoch), desc=f"Epoch[{epoch + 1}/{trainer.cfg.train.max_epochs}]"):
        if epoch * iters_every_epoch + step < trainer.global_step:
            continue
        try:
            batch = next(trainer.train_iter)
        except:
            trainer.train_iter = iter(trainer.train_dataloader)
            batch = next(trainer.train_iter)
        '''-------------------------
        Model Training
        --------------------------'''
        if trainer.global_step % 6000 < 5000:
            unet_for_mask.eval()
            codedict = deca.encode(batch['image'].cuda())
            opdict, visdict = deca.decode(codedict)  # tensor
            raster_image = visdict['rendered_images']
            image_concatenated = torch.cat((raster_image, batch['image'].cuda()), axis=1)
            unet_est_mask = unet_for_mask(image_concatenated)
            batch['mask'] = unet_est_mask
            losses, opdict = trainer.training_step(batch, step)
            all_loss = losses['all_loss']


            all_loss.backward()
            trainer.opt.step()

            loss_info = f"ExpName: DECA- \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
            for k, v in losses.items():
                loss_info = loss_info + f'{k}: {v:.4f}, '
                trainer.writer.add_scalar('deca_train_loss/' + k, v, global_step=trainer.global_step)
            logger.info(loss_info)
        else:
            unet_for_mask.train()
            deca.eval()
            images = batch['image']
            landmarks = batch['landmark']
            images = images.cuda()
            landmarks = landmarks.cuda()
            loss_unet, losses_return_unet, _, _, _, _, losses_dict = proc_mofaunet(batch, images, landmarks, True,
                                                                      'unet')  # TODO swap data and batch
            loss_unet.backward()
            optimizer_unet.step()

            loss_info = f"ExpName: Unet \nEpoch: {epoch}, Iter: {step}/{iters_every_epoch}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"

            for k, v in losses_dict.items():
                loss_info = loss_info + f'{k}: {v:.4f}, '
                trainer.writer.add_scalar('unet_train_loss/' + k, v, global_step=trainer.global_step)
            logger.info(loss_info)

        '''-------------------------
        Save images for observation
        --------------------------'''
        if trainer.global_step % 5000 == 0:
            visind = list(range(1))
            shape_images = trainer.deca.render.render_shape(opdict['verts'][visind], opdict['trans_verts'][visind])
            visdict = {
                'inputs': opdict['images'][visind],
                'landmarks2d_gt': deca_util.tensor_vis_landmarks(opdict['images'][visind], opdict['lmk'][visind],
                                                            isScale=True),
                'landmarks2d': deca_util.tensor_vis_landmarks(opdict['images'][visind], opdict['landmarks2d'][visind],
                                                         isScale=True),
                'shape_images': shape_images,
            }
            if 'predicted_images' in opdict.keys():
                visdict['predicted_images'] = opdict['predicted_images'][visind]
            if 'predicted_detail_images' in opdict.keys():
                visdict['predicted_detail_images'] = opdict['predicted_detail_images'][visind]
            savepath = os.path.join(trainer.cfg.output_dir, trainer.cfg.train.vis_dir, f'{trainer.global_step:06}.jpg')
            grid_image = deca_util.visualize_grid(visdict, savepath, return_gird=True)
            trainer.writer.add_image('deca_train_images', (grid_image / 255.).astype(np.float32).transpose(2, 0, 1),
                                     trainer.global_step)
        '''-------------------------
        Save Model
        --------------------------'''
        if trainer.global_step > 0 and trainer.global_step % trainer.cfg.train.checkpoint_steps == 0:
            model_dict = trainer.deca.model_dict()
            model_dict['opt'] = trainer.opt.state_dict()
            model_dict['global_step'] = trainer.global_step
            model_dict['batch_size'] = trainer.batch_size
            torch.save(model_dict, os.path.join(trainer.cfg.output_dir, 'model' + '.tar'))
            torch.save(unet_for_mask, os.path.join(trainer.cfg.output_dir, 'model_unet' + '.tar'))

            logger.info("SAVED MODEL")

            if trainer.global_step % trainer.cfg.train.checkpoint_steps * 10 == 0:
                os.makedirs(os.path.join(trainer.cfg.output_dir, 'models'), exist_ok=True)
                torch.save(model_dict, os.path.join(trainer.cfg.output_dir, 'models', f'{trainer.global_step:08}.tar'))
                torch.save(unet_for_mask,
                           os.path.join(trainer.cfg.output_dir, 'models', f'{trainer.global_step:08}_unet.tar'))
        '''-------------------------
        Validate Model
        --------------------------'''
        if trainer.global_step % 5000 == 0: #5000
            trainer.validation_step()
            val_batch = trainer.current_val_batch
            val_images = val_batch['image'].cuda()
            val_landmarks = val_batch['landmark'].cuda()
            with torch.no_grad():
                loss_, losses_return_, _, _, _, _, losses_dict = proc_mofaunet(val_batch, val_images, val_landmarks, True, False)
            str = 'test loss:{}'.format(trainer.global_step)
            for loss_temp in losses_return_:
                str += ' {:05f}'.format(loss_temp)
            logger.info(str)

            for k, v in losses_dict.items():
                loss_info = loss_info + f'{k}: {v:.4f}, '
                trainer.writer.add_scalar('unet_val_loss/' + k, v, global_step=trainer.global_step)
            logger.info(loss_info)

        if trainer.global_step % 5000 == 0:
            trainer.evaluate()

        trainer.global_step += 1

        trainer.opt.zero_grad(set_to_none=True)
        trainer.opt.step()

        optimizer_unet.zero_grad()
        optimizer_unet.step()

        logger.info(trainer.global_step)
        if trainer.global_step > trainer.cfg.train.max_steps:
            break
