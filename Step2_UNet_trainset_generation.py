import numpy
import torch
import os
import math

from scipy.io import savemat
from tqdm import tqdm

import util.load_dataset as load_dataset
import util.util as focusutil
import util.load_object as lob
import renderer.rendering as ren
import encoder.encoder as enc
import cv2
import numpy as np
import argparse
import torch.nn.functional as F

from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

from decalib import deca
from decalib.datasets import datasets
from decalib.deca import DECA
from decalib.utils.config import cfg as cfg
from decalib.models.encoders import ResnetEncoder
from face_toolbox_keras.models.parser import face_parser
from face_toolbox_keras.models.detector import face_detector
import face_alignment

par = argparse.ArgumentParser(description='Generate training set for UNet')
par.add_argument('--pretrained_MoFA',default = '/pretrain/deca_model.tar',type=str,help='Path of the pre-trained model')
par.add_argument('--gpu',default=0,type=int,help='The GPU ID')
par.add_argument('--img_path',type=str,help='Root of the training samples')

args = par.parse_args()
GPU_no = args.gpu
trained_model_path = args.pretrained_MoFA
output_name = 'UNet_trainset'

image_path = (args.img_path + '/' ).replace('//','/')
current_path = os.getcwd()
#model_path = current_path+'/basel_3DMM/model2017-1_bfm_nomouth.h5'
save_path = current_path+'/image_root/Data/'+output_name + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

savefolder = current_path+'/image_root/Data/Dataset2/'+output_name + '/'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
    
device = torch.device("cuda:{}".format(focusutil.device_ids[GPU_no ]) if torch.cuda.is_available() else "cpu")


#parameters
batch = 2
width = height = 224


testdata = datasets.TestData('image_root/Data/Dataset2', iscrop=True, face_detector='fan',sample_step=10)
deca = DECA(config=deca_cfg, device=device)
print("Model loaded...")

for i in tqdm(range(len(testdata))):
     name = testdata[i]['imagename']
     images = testdata[i]['image'].to(device)[None, ...]
     with torch.no_grad():
         codedict = deca.encode(images)
         opdict, visdict = deca.decode(codedict)  # tensor
         tform = testdata[i]['tform'][None, ...]
         tform = torch.inverse(tform).transpose(1, 2).to(device)
         original_image = testdata[i]['original_image'][None, ...].to(device)
         _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)
         orig_visdict['inputs'] = original_image
     # -- save results
         depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
         visdict['depth_images'] = depth_image
         cv2.imwrite(os.path.join(savefolder, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
         np.savetxt(os.path.join(savefolder,  name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
         np.savetxt(os.path.join(savefolder,  name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
         #deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
         opdict = util.dict_tensor2npy(opdict)
         savemat(os.path.join(savefolder, name + '.mat'), opdict)
         cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
         cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
         for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images',
                          'landmarks2d']:
             if vis_name not in visdict.keys():
                 continue
             image = util.tensor2image(visdict[vis_name][0])
             cv2.imwrite(os.path.join(savefolder, name + '_' + vis_name + '.jpg'),
                         util.tensor2image(visdict[vis_name][0]))

             image = util.tensor2image(orig_visdict[vis_name][0])
             cv2.imwrite(os.path.join(savefolder, 'orig_' + name + '_' + vis_name + '.jpg'),
                         util.tensor2image(orig_visdict[vis_name][0]))

root = 'image_root/Data/Dataset2/'
dir_list = os.listdir(root)
for img in tqdm(dir_list):
    im = cv2.imread(root + img)[..., ::-1]
    im = cv2.resize(im, (224, 224))
    fp = face_parser.FaceParser()
    parsing_map = fp.parse_face(im, bounding_box=None, with_detection=False)[0]
    img_mask = numpy.array(parsing_map)
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

    cv2.imwrite('image_root/Data/output/' + img.replace('.jpg','_mask.jpg'),img_mask*255)

    np.save('image_root/Data/output/' + img.strip('.jpg'), parsing_map)
print(f'-- please check the results in {savefolder}')


