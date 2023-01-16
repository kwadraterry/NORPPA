from math import ceil, floor
import numpy as np
from PIL import ImageFile
import tensorflow as tf

from math import ceil, floor
from pattern_extraction.model import *
from pattern_extraction.utils import *
from pathlib import Path

import skimage.transform as trans

file_folder = Path(__file__).resolve().parent
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_unet(model_path):
    return unet(pretrained_weights=str(model_path))

def extract_pattern(input, model):
    confidence=0.6
    image, img_label = input
    if image is None:
        return [input]

    initial_size = image.size

    
    scale = 512/max(initial_size)
    new_size = [floor(x*scale) for x in initial_size]
    image = image.convert('L')
    image = np.array(image)
    image = trans.resize(image, new_size[::-1])
    scaled_size = image.shape
    image = np.pad(image, ((0, 512-scaled_size[0]),(0, 512-scaled_size[1])), 'constant', constant_values=(0))

    image = tf.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis=-1)
    result = model.predict(image)

    result = result[0, :scaled_size[0], :scaled_size[1], 0]

    result = postprocess(result, confidence, initial_size, True)
    if result is not None:
        result, ratio = result
        if type(img_label) is dict:
            img_label["resize_ratio"] = ratio
        if sum(result.convert("L").getextrema()) in (0, 2):
            result = None
    
    return [(result, img_label)]