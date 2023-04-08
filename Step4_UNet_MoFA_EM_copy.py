import os
import cv2
import numpy
import torch
import math
import torch.optim as optim
from tqdm import tqdm
import util.util as util
import csv
import util.load_dataset as load_dataset
import util.load_object as lob
import renderer.rendering as ren
import encoder.encoder as enc
import torch.nn.functional as F
import numpy as np
import time
import UNet.UNet as unet
import argparse
from datetime import date
import util.advanced_losses as adlosses
from decalib.deca import DECA
from decalib.utils import lossfunc
from decalib.utils.config import cfg
import decalib.utils.config as config
from models import networks
from decalib.trainer import Trainer
from face_toolbox_keras.models.parser import face_parser
import pickle

# hyper-parameters
par = argparse.ArgumentParser(description='MoFA')

par.add_argument('--learning_rate', default=0.1, type=float, help='The learning rate')
par.add_argument('--epochs', default=1, type=int, help='Total epochs')
par.add_argument('--batch_size', default=4, type=int, help='Batch sizes')
par.add_argument('--gpu', default=0, type=int, help='The GPU ID')
par.add_argument('--pretrained_model', default=00, type=int, help='Pretrained model')
par.add_argument('--img_path', type=str, default='image_root/Data/Dataset', help='Root of the training samples')

args = par.parse_args()

GPU_no = args.gpu

dist_weight = {'neighbour': 15, 'dist': 3, 'area': 0.5, 'preserve': 0.25, 'binary': 10}

ct = args.pretrained_model  # load trained mofa model
deca = DECA(cfg)
output_name = 'Deca_UNet'

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

trainer = Trainer(model=deca, config=cfg)

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
#model_path = current_path + '/basel_3DMM/model2017-1_bfm_nomouth.h5'

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
  Prepare Log Files
------------------'''
if ct != 0:
    try:
        fid_train = open(loss_log_path_train, 'a')
        fid_test = open(loss_log_path_test, 'a')
    except:
        fid_train = open(loss_log_path_train, 'w')
        fid_test = open(loss_log_path_test, 'w')
else:
    fid_train = open(loss_log_path_train, 'w')
    fid_test = open(loss_log_path_test, 'w')
writer_train = csv.writer(fid_train, lineterminator="\r\n")
writer_test = csv.writer(fid_test, lineterminator="\r\n")

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

'''-------------
  Load Dataset
-------------'''

testset = load_dataset.CelebDataset(device, image_path, False, height, width, 1)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch, shuffle=False, num_workers=0)

# 3dmm data
'''----------------
Prepare Network and Optimizer
----------------'''

# renderer and encoder and UNet

'''----------------------------------
 Fixed Testing Images for Observation
----------------------------------'''
test_input_images = []
test_landmarks = []
test_landmark_masks = []
for i_test, data_test in enumerate(testloader, 0):
    if i_test >= test_batch_num:
        break
    images, landmarks = data_test
   # landmarks.permute()
    test_input_images += [images]
    test_landmarks += [landmarks]
util.write_tiled_image(torch.cat(test_input_images, dim=0), output_path + 'test_gt.png', 10)

def generateMasks(images):
    masks = []
    for img in tqdm(images):
        #img = F.interpolate(img, size=(3,224,224))

        #img = F.interpolate(img, size=[3,224,224])
        im = img.cpu().detach().numpy()[..., ::-1]
        im = im.transpose(1, 2, 0)
        fp = face_parser.FaceParser()
        parsing_map = fp.parse_face(im, bounding_box=None, with_detection=False)[0]
        img_mask = np.array(parsing_map)

        img_mask[img_mask == 1] = 1
        img_mask[img_mask == 2] = 1
        img_mask[img_mask == 3] = 1
        img_mask[img_mask == 4] = 1
        img_mask[img_mask == 5] = 1
        img_mask[img_mask == 10] = 1
        img_mask[img_mask == 11] = 1
        img_mask[img_mask == 12] = 1
        img_mask[img_mask == 13] = 1

        img_mask[img_mask == 6] = 0
        img_mask[img_mask == 7] = 0
        img_mask[img_mask == 8] = 0
        img_mask[img_mask == 9] = 0
        img_mask[img_mask == 14] = 0
        img_mask[img_mask == 15] = 0
        img_mask[img_mask == 16] = 0
        img_mask[img_mask == 17] = 0
        img_mask[img_mask == 18] = 0

        masks.append(img_mask)
    return torch.ShortTensor(numpy.array(masks))


'''-------------
Network Forward
-------------'''


#################################################################
def proc_mofaunet(batch, batchn, images, landmarks, render_mode, train_net=False, occlusion_mode=False, valid_mask=None,
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
    raster_mask = generateMasks(images).cuda() #TODO load it
    raster_image = visdict['rendered_images']
    image_concatenated = torch.cat((raster_image, images), axis=1)
    unet_est_mask = unet_for_mask(image_concatenated)
    valid_loss_mask = raster_mask.unsqueeze(1) * unet_est_mask

    masked_rec_loss = torch.mean(torch.sum(torch.norm(valid_loss_mask * (images - raster_image), 2, 1)) / torch.clamp(
        torch.sum(raster_mask.unsqueeze(1) * unet_est_mask), min=1))

    bg_unet_loss = torch.mean(torch.sum(raster_mask.unsqueeze(1) * (1 - unet_est_mask), axis=[2, 3]) / torch.clamp(
        torch.sum(raster_mask.unsqueeze(1), axis=[2, 3]), min=1))  # area loss
    mask_binary_loss = torch.zeros([1])
    loss_mofa = torch.zeros([1])
    if train_net == 'unet':
        mask_binary_loss = (0.5 - torch.mean(torch.norm(valid_loss_mask - 0.5, 2, 1)))
        loss_unet = mask_binary_loss * dist_weight['binary'] + bg_unet_loss * dist_weight['area']
    if train_net == 'mofa':
        deca_batch = {}
        deca_batch['image'] = images
        deca_batch['landmark'] = landmarks
        deca_batch['mask'] = unet_est_mask

        #d_losses, paramdict = trainer.training_step(deca_batch, batchn, training_type='detail')
        c_losses, paramdict = trainer.training_step(deca_batch, batchn, training_type='coarse')

        loss_mofa = c_losses['all_loss'] #+ d_losses['all_loss']

    if train_net == False:
        mask_binary_loss = (0.5 - torch.mean(torch.norm(valid_loss_mask - 0.5, 2, 1)))
        loss_test = mask_binary_loss.to('cpu') * dist_weight['binary'] + masked_rec_loss.to('cpu') * 0.5 + bg_unet_loss.to('cpu') * dist_weight[
            'area'] + loss_mofa.to('cpu')

    #I_target_masked = images * valid_loss_mask
    #id_target_masked = net_recog(I_target_masked, landmarks.transpose(1, 2), is_shallow=True)
    #id_target = net_recog(images, landmarks.transpose(1, 2), is_shallow=True)
    #id_reconstruct_masked = net_recog(raster_image * valid_loss_mask, pred_lm=lm68.transpose(1, 2), is_shallow=True)
    #I_IM_Per_loss = torch.mean(1 - cos(id_target, id_target_masked))
    #IRM_IM_Per_loss = torch.mean(1 - cos(id_reconstruct_masked, id_target_masked))
    #if train_net == 'unet':
    #    loss_unet += I_IM_Per_loss * dist_weight['preserve'] + IRM_IM_Per_loss * dist_weight['dist']
    #if train_net == False:
    #    loss_test += I_IM_Per_loss * dist_weight['preserve'] + IRM_IM_Per_loss * dist_weight['dist']

    #TODO maybe calculate loss for DECA

    # force it to be binary mask
    loss_mask_neighbor = torch.zeros([1])
    if train_net == 'unet':
        loss_mask_neighbor = adlosses.neighbor_unet_loss(images, valid_loss_mask, raster_image)
        loss_unet += loss_mask_neighbor * dist_weight['neighbour']
        loss = loss_unet
    if train_net == 'mofa':
        loss = loss_mofa
    if train_net == False:
        loss = loss_test
    losses_return = torch.FloatTensor(
        [loss.item(), masked_rec_loss.item(), bg_unet_loss.item(), \
         loss_mask_neighbor.item(),
         mask_binary_loss.item(), loss_mofa])

    if train_net == 'unet':
        return loss_unet, losses_return, raster_image, raster_mask, unet_est_mask, valid_loss_mask
    if train_net == 'mofa':
        return loss_mofa, losses_return, raster_image, raster_mask, unet_est_mask, valid_loss_mask
    if train_net == False:
        return loss_test, losses_return, raster_image, raster_mask, unet_est_mask, valid_loss_mask


#################################################################

'''-----------------------------------------
load pretrained model and continue training
-----------------------------------------'''
if ct != 0:

    trained_model_path = output_path + 'model_{:06d}.tar'.format(ct)
    trainer.load_checkpoint(ct);
    unet_model_path = output_path + 'unet_{:06d}.model'.format(ct)
    unet_for_mask = torch.load(unet_model_path, map_location='cuda:{}'.format(util.device_ids[GPU_no]))


else:
    trained_model_path = current_path + '/decalib/pretrain/' + 'deca_modelgit.tar'
    # enc_net = torch.load(trained_model_path, map_location='cuda:{}'.format(util.device_ids[GPU_no]))
    unet_model_path = current_path + '\\MoFA_UNet_Save\\Pretrain_UNet' + '\\unet_mask_000070.model'
    unet_for_mask = torch.load(unet_model_path, map_location='cuda:{}'.format(util.device_ids[GPU_no]))

print('Loading pre-trained unet: \n' + trained_model_path + '\n' + unet_model_path)

'''----------
Set Optimizer
----------'''
# optimizer_mofa = optim.Adadelta(enc_net.parameters(), lr=mofa_lr_begin)
optimizer_unet = optim.Adadelta(unet_for_mask.parameters(), lr=unet_lr_begin)

scheduler_unet = torch.optim.lr_scheduler.StepLR(optimizer_unet, step_size=decay_step_size, gamma=decay_rate_unet)

opt = torch.optim.Adam(
    list(deca.E_detail.parameters()) + \
    list(deca.D_detail.parameters()),
    lr=1e-4,
    amsgrad=False)

scheduler_deca = torch.optim.lr_scheduler.StepLR(opt, step_size=decay_step_size, gamma=0.99)

print('Training ...')
start = time.time()
mean_losses_mofa = torch.zeros([6])
mean_losses_unet = torch.zeros([6])

trainset = load_dataset.CelebDataset(device, image_path, True, height, width, 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


for ep in range(0, epoch):

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True, num_workers=0)

    for i, data in enumerate(trainloader, 0):

        '''-------------------------
        Save images for observation
        --------------------------'''
        a = count_parameters(deca)
        b = count_parameters(unet_for_mask)
        if (ct - ct_begin) % 3 == 0: #1000
            deca.eval()
            unet_for_mask.eval()
            test_raster_images = []
            valid_loss_mask_temp = []

            for images, landmarks in zip(test_input_images, test_landmarks):
                with torch.no_grad():
                    _, _, raster_image, raster_mask, fg_mask, valid_loss_mask = proc_mofaunet(images, i,images, landmarks, True,
                                                                                              False)

                    test_raster_images += [
                        images * (1 - raster_mask.unsqueeze(1)) + raster_image * raster_mask.unsqueeze(1)]
                    valid_loss_mask_temp += [valid_loss_mask]

            util.write_tiled_image(torch.cat(test_raster_images, dim=0), output_path + 'test_image_{}.png'.format(ct),
                                   10)
            util.write_tiled_image(torch.cat(valid_loss_mask_temp, dim=0),
                                   output_path + 'valid_loss_mask_{}.png'.format(ct), 10)

            '''-------------------------
             Save Model every 5000 iters
            --------------------------'''
            if (ct - ct_begin) % 6 == 0 and ct > ct_begin: #10000
                # torch.save(enc_net, output_path + 'enc_net_{:06d}.model'.format(ct))
                model_dict = trainer.deca.model_dict()
                model_dict['opt'] = opt.state_dict()
                model_dict['global_step'] = ct
                model_dict['batch_size'] = batch
                torch.save(model_dict, os.path.join(output_path, 'deca' + '.tar'))
                torch.save(unet_for_mask, output_path + 'unet_{:06d}.model'.format(ct))

            # validating
            '''-------------------------
            Validate Model every 1000 iters
            --------------------------'''
            if (ct - ct_begin) % 6 == 0 and ct > ct_begin: #5000
                print('Training mode:' + output_name)
                c_test = 0
                mean_test_losses = torch.zeros([6])

                for i_test, data_test in enumerate(testloader, 0):
                    image, landmark = data_test
                    c_test += 1
                    with torch.no_grad():
                        loss_, losses_return_, _, _, _, _ = proc_mofaunet(data, i,image, landmark, True, False)
                        mean_test_losses += losses_return_
                mean_test_losses = mean_test_losses / c_test
                str = 'test loss:{}'.format(ct)
                for loss_temp in losses_return_:
                    str += ' {:05f}'.format(loss_temp)
                print(str)
                writer_test.writerow(str)

            fid_train.close()
            fid_train = open(loss_log_path_train, 'a')
            writer_train = csv.writer(fid_train, lineterminator="\r\n")

            fid_test.close()
            fid_test = open(loss_log_path_test, 'a')
            writer_test = csv.writer(fid_test, lineterminator="\r\n")

        '''-------------------------
        Model Training
        --------------------------'''

        images, landmarks = data
        torch.permute(landmarks, (0,2,1))

        if images.shape[0] != batch:
            continue
        if ct % 10 > 5: # 30000 > 5000
            deca.train()
            unet_for_mask.eval()
            loss_mofa, losses_return_mofa, _, _, _, _ = proc_mofaunet(data, i, images, landmarks, True, 'mofa')
            loss_mofa.backward()
            opt.step()

            mean_losses_mofa += losses_return_mofa
            # optimizer_mofa.zero_grad()
        else:
            unet_for_mask.train()
            deca.eval()
            loss_unet, losses_return_unet, _, _, _, _ = proc_mofaunet(data, i,images, landmarks, True, 'unet')
            loss_unet.backward()
            optimizer_unet.step()

            # optimizer_unet.zero_grad()
            mean_losses_unet += losses_return_unet
        ct += 1
        scheduler_unet.step()
        scheduler_deca.step()
        optimizer_unet.zero_grad()
        opt.zero_grad()

        '''-------------------------
        Show Training Loss
        --------------------------'''

        if (ct - ct_begin) % 10 == 0 and ct > ct_begin: #100
            end = time.time()
            mean_losses_unet = mean_losses_unet / 10
            mean_losses_mofa = mean_losses_mofa / 10
            str = 'deca loss:{}'.format(ct)
            for loss_temp in mean_losses_mofa:
                str += ' {:05f}'.format(loss_temp)
            str += '\nunet loss:{}'.format(ct)
            for loss_temp in mean_losses_unet:
                str += ' {:05f}'.format(loss_temp)
            str += ' time: {}'.format(end - start)
            print(str)
            writer_train.writerow(str)
            start = time.time()
            mean_losses_unet = torch.zeros([6])
            mean_losses_mofa = torch.zeros([6])

model_dict = trainer.deca.model_dict()
model_dict['opt'] = opt.state_dict()
model_dict['global_step'] = ct
model_dict['batch_size'] = batch
torch.save(model_dict, os.path.join(output_path, 'deca' + '.tar'))
