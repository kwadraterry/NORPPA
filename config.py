import os
import sys
from pathlib import Path
import cv2
import numpy as np

file_folder = Path(__file__).resolve().parent
sys.path.append(str(file_folder))
sys.path.append(str(file_folder / "reidentification/hesaff_pytorch"))


from HessianAffinePatches import init_affnet, init_orinet, init_hardnet
# from segmentation.detectron_segment import create_predictor
from pattern_extraction.extract_pattern import create_unet
from segmentation.seem.seem_segment import init_seem
from torchvision.datasets.utils import download_url
# from sql import create_connection
import torch
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def init_file(path, url, allow_download=True):
    if Path(path).exists():
        return path
    elif allow_download:
        download_url(url, Path(path).parent, Path(path).name)
        return path
    else:
        raise Exception("The file {path} is not found!")
    

# base_dir = Path(__file__).resolve().parent

filedir = Path(__file__).resolve().parent

def config(use_cuda=torch.cuda.is_available(), allow_download=True, base_dir=filedir):   

    config =  {}     
    # mount_path = "/ekaterina/work/data/"
    
    # path_db = mount_path + "DB.db"
    # config["conn"] = create_connection()
    # config["detectron_predictor"] = create_predictor(init_file(base_dir/"models/R-101-FPN_150ims.pth",  
    #                                 "https://github.com/kwadraterry/NORPPA/raw/models/models/R-101-FPN_150ims.pth", 
    #                                 allow_download=allow_download),
    #                                 not  use_cuda )
    config["unet"] = create_unet(init_file(base_dir/"models/unet_seals_512.hdf5",  
                                            "https://github.com/kwadraterry/NORPPA/raw/models/models/unet_seals_512.hdf5", 
                                            allow_download=allow_download))
    config["net"] = init_hardnet(init_file(base_dir/"models/HardNet++.pth", 
                                            "https://github.com/kwadraterry/NORPPA/raw/models/models/HardNet++.pth", 
                                            allow_download=allow_download), 
                                use_cuda=use_cuda)
    affnet = init_affnet(init_file(base_dir/"models/AffNet.pth", 
                                            "https://github.com/kwadraterry/NORPPA/raw/models/models/AffNet.pth", 
                                            allow_download=allow_download), 
                                use_cuda=use_cuda)
    orinet = init_orinet(init_file(base_dir/"models/OriNet.pth",  
                                            "https://github.com/kwadraterry/NORPPA/raw/models/models/OriNet.pth", 
                                            allow_download=allow_download), 
                                use_cuda=use_cuda)
    codebooks_path = init_file(base_dir/'codebooks/codebooks.pickle',
                               "https://github.com/kwadraterry/NORPPA/raw/models/codebooks/codebooks.pickle", 
                               allow_download=allow_download)
    config["codebooks_path"] = Path(base_dir/"codebooks/norppa.pickle")
    config["codebooks"] = None
    config["hesaff_args"] = {'init_sigma': 1.3213713243956968, 
                            'mrSize': 9.348280997446642, 
                            'nlevels': 10, 
                            'num_features': 480, 
                            'unsharp_amount': 6.80631647207343, 
                            'unsharp_radius': None,
                            'use_cuda' :use_cuda}
    config["hesaff_args"]["AffNet"] = affnet
    config["hesaff_args"]["OriNet"] = orinet

    config["hesaff_args"]["patch_size"] = 32


    config["use_hesaff"] = True
    

    config["pca"] = 64
    config["use_kpca"] = False

    config["n_clusters"] = 1400
    config["features_shape"] = 64
    config["topk"] = 10

    config["kernel"] = "rbf"
    config["use_cuda"] = use_cuda
    config["dataset_dir"] = base_dir/'data'
    config["sequence_dataset_dir"] = '/ekaterina/work/data/many_dataset/original_small'

    config["batch_size"] = 256
    
    config["geometric"] = {
        "method": cv2.RANSAC,
        "max_iters": 5000,
        "max_reproj_err": 0.2,
        "estimator": lambda d, mask: d ** np.sum(mask)
    }
    seem_model, seem_transform = init_seem(conf_files=str(Path(filedir/"segmentation/seem/configs/seem/seem_focall_lang.yaml")), model_path=str(Path(base_dir/"models")), use_cuda=use_cuda)
    config["seem"] = {
        "model": seem_model,
        "transform": seem_transform
    }
    

    return config