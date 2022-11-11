from pathlib import Path
import os
import pycocotools
from PIL import Image
import numpy as np
from detectron2.structures import BoxMode

def create_dataset_json(full_dir, segmented_dir, keyword="", suffix=".result.png"):
    full_dir = Path(full_dir)
    segmented_dir = Path(segmented_dir)
    result = []
    counter = 0
    for root, _, files in os.walk(full_dir):
        root = Path(root)
        for name in files:
            full_img = root / name
            if keyword in str(full_img):
                segmented_path = segmented_dir / (str(full_img.relative_to(full_dir))+suffix)
                if not segmented_path.is_file():
                    continue
                mask_img = Image.open(segmented_path).convert("L")
                width, height = mask_img.size
                bbox = mask_img.getbbox()
                mask_img = (np.asarray(mask_img) > 0).astype(np.uint8)
                # print(segmented_path)
                # print(np.sum(mask_img))
                # print(mask_img.shape[0] * mask_img.shape[1])
                mask = pycocotools.mask.encode(np.asarray(mask_img, order="F"))
                counter += 1
                result.append({'file_name': str(full_img),
                    'image_id': counter,
                    'height': height,
                    'width': width,
                    'annotations': [
                        {'iscrowd': 0,
                        'segmentation': mask, 
                        'bbox': bbox,
                        'bbox_mode': BoxMode.XYXY_ABS,
                        'category_id': 0 }]})
    return result


        

