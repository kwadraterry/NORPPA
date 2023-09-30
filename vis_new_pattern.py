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

from norppa_tools import *
from tonemapping.tonemapping import tonemap, tonemap_step
from segmentation.segmentation import segment
from pattern_extraction.extract_pattern import extract_pattern
from reidentification.identify import *
from reidentification.visualisation import visualise_match
from reidentification.find_matches import find_matches



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

from norppa_tools import apply_pipeline, crop_step, curry, apply_pipeline_dataset, get_save_step, apply_sequential, compose, compose_sequential
from tonemapping.tonemapping import tonemap, tonemap_step
from segmentation.segmentation import segment
from pattern_extraction.extract_pattern import extract_pattern
from reidentification.identify import encode_single, encode_pipeline, encode_dataset, identify, identify_single
from reidentification.visualisation import visualise_match

cfg = config()


cfg["codebooks"] = load_pickle("whaleshark_norppa_tonemapped_pattern_maxim_oldgmm_codebooks_scale1.pickle")
print("loaded codebooks")
encoded_train_dataset = load_pickle("whaleshark_norppa_tonemapped_pattern_maxim_oldgmm_encoded_scale1.pickle")
print("loaded encodings")

test_pipeline1 = [
                 curry(identify, encoded_train_dataset, cfg["topk"], leave_one_out=True),
                 curry(print_topk_accuracy, label="Before geometric verification:"),
                ]
test_pipeline2 = [
                 curry_sequential(find_matches, cfg),
                 curry_sequential(apply_geometric, cfg["geometric"]),
                 curry(print_topk_accuracy, label="After geometric verification:"),
#                  curry_sequential(visualise_match, cfg["topk"])
                ]

matches1 = apply_pipeline_dataset(encoded_train_dataset, test_pipeline1)

print("found matches 1")
save_pickle(matches1, "temp/files/new_pattern.matches1.pickle")


matches2 = apply_pipeline_dataset(matches1, test_pipeline2)

print("found matches 1")
save_pickle(matches2, "temp/files/new_pattern.matches2.pickle")
