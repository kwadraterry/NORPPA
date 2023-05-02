import os 
# import sys
# sys.path.append('/ekaterina/work/src/NORPPA/repository/NORPPA')
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from config_whaleshark import config
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import zipfile
import tensorflow as tf
import wget
import pickle
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from torchvision.datasets.utils import download_url
from datasets import SimpleDataset, DatasetSlice

from tools import apply_pipeline, crop_step, curry, apply_pipeline_dataset, get_save_step, apply_sequential, compose, compose_sequential
from tonemapping.tonemapping import tonemap, tonemap_step
from segmentation.segmentation import segment
from pattern_extraction.extract_pattern import extract_pattern
from reidentification.identify import encode_single, encode_pipeline, encode_dataset, identify, identify_single
from reidentification.visualisation import visualise_match

cfg = config()

train_path = "/ekaterina/work/data/whaleshark_norppa_tonemapped/train"
test_path = "/ekaterina/work/data/whaleshark_norppa/test"

dataset_train = Path(train_path)
dataset_test = Path(test_path)

train_dataset = SimpleDataset(dataset_train)
codebooks_path = '/ekaterina/work/src/NORPPA/repository/NORPPA/codebooks/whaleshark_tonemapped.pickle'


codebooks, encoded_dataset = encode_dataset(train_dataset, cfg, compute_codebooks=True)

try:
    with open("tonemapped_whaleshark.pickle", 'wb') as f_file:
        pickle.dump(encoded_dataset, f_file, protocol=4)
except Exception as e:
    print(e)
    
try:
    with open(codebooks_path, "wb") as codebooks_file:
        pickle.dump(codebooks, codebooks_file, protocol=4)
except Exception as e:
    print(e)

