import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import denormalizeLAFs, LAFs2ell
from Utils import line_prepender

from architectures import AffNetFast, OriNetFast
from skimage.filters import unsharp_mask
import math
from HardNet import HardNet

def init_hardnet(model_weights, use_cuda=True):
    descriptor = HardNet()
    hncheckpoint = torch.load(model_weights, map_location=torch.device('cpu'))
    descriptor.load_state_dict(hncheckpoint['state_dict'])
    descriptor.eval()
    if use_cuda:
        descriptor = descriptor.cuda()
    return descriptor

def init_affnet(weightd_fname, patch_size=32, use_cuda=True):
    AffNetPix = AffNetFast(PS=patch_size)

    checkpoint = torch.load(weightd_fname, map_location=torch.device('cpu'))
    AffNetPix.load_state_dict(checkpoint['state_dict'])

    AffNetPix.eval()
    if use_cuda:
        AffNetPix.cuda()
    return AffNetPix

def init_orinet(o_fname, patch_size=32, use_cuda=True):
    ONet = OriNetFast(PS=patch_size)
    checkpoint = torch.load(o_fname, map_location=torch.device('cpu'))
    ONet.load_state_dict(checkpoint['state_dict'])
    ONet.eval()
    if use_cuda:
        ONet.cuda()
    return ONet

def LAF2ell(LAF):
    LAF = np.array(LAF)
    u, s, _ = np.linalg.svd(LAF[:2, :2])
    angle = math.atan2(u[1, 0], u[0, 0])
    return np.array([LAF[0, 2], LAF[1, 2], s[0], s[1], angle])

def extract_hesaff_patches(img_detect, 
                           num_features=400, 
                           nlevels=10, 
#                            mrSize=12, 
                           mrSize=12,
#                            border=5, 
                           border=5,
                           num_Baum_iters=26, 
                           patch_size=48, 
#                            init_sigma=1.6, 
                           init_sigma = 200,
                           unsharp_radius=1, 
                           unsharp_amount=25, 
                           patch_scale=1,
                           RespNet=None, 
                           OriNet=None, 
                           AffNet=None, 
                           use_cuda=True):
    
    HA = ScaleSpaceAffinePatchExtractor(nlevels=int(nlevels),
                                        mrSize=mrSize,
                                        num_features=int(num_features),
                                        border=border,
                                        num_Baum_iters=num_Baum_iters,
                                        patch_size=patch_size,
                                        init_sigma=init_sigma,
                                        AffNet=AffNet,
                                        OriNet=OriNet,
                                        RespNet=RespNet,
                                       patch_scale=patch_scale)

    var_image_detect = torch.autograd.Variable(torch.from_numpy(np.array(img_detect).astype(np.float32)))
    var_image_detect = var_image_detect.view(1, 1, var_image_detect.size(0),var_image_detect.size(1))
    if use_cuda:
        HA = HA.cuda()
        var_image_detect = var_image_detect.cuda()
    
    LAFs, resp  = HA(var_image_detect)
    
    patches = HA.extract_patches_from_pyr(LAFs, patch_size)
    if use_cuda:
        patches = patches.cpu()
    patches = patches[:, 0, ...].detach().numpy()
    # print(unsharp_radius)
    if (unsharp_radius is not None) and (unsharp_amount is not None):
        for i in range(patches.shape[0]):
            patches[i,...] = 255*unsharp_mask(patches[i,...]/255, radius=unsharp_radius, amount=unsharp_amount)
    LAFs = LAFs.cpu().detach().numpy()
    ells = [None] * LAFs.shape[0]
    for i in range(len(ells)):
        ells[i] = LAF2ell(LAFs[i, ...])
    return patches, ells
    