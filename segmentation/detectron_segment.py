from argparse import ArgumentParser
from PIL import Image
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import numpy as np
import cv2
import rawpy
from pathlib import Path

def is_raw_image(filename):
    return filename.lower().endswith(('cr2', 'pef', 'nef'))

def convert_raw_img(path):
    if is_raw_image(path):
        with rawpy.imread(path) as raw:
            rgb = raw.postprocess()
        return rgb
    return cv2.imread(path)

def pil2cv2(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    

def apply_masks(im, masks):
    final_mask = np.tile(np.expand_dims(np.sum(masks.detach().cpu().numpy(), axis=0), axis=2), (1, 1, 3))
    seg_im = im * final_mask
    return seg_im

def apply_mask(im, mask):
    final_mask = np.tile(np.expand_dims(mask.detach().cpu().numpy(), axis=2), (1, 1, 3))
    seg_im = im * final_mask
    return seg_im

def create_predictor(model_path, use_cpu=False):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    cfg.MODEL.DEVICE = "cpu" if use_cpu else "cuda" # uncomment if gpu is unavailable
    return DefaultPredictor(cfg)

def detectron_segment(predictor, source_image, instance_segmentation):

    # im = convert_raw_img(source_image)

    im = pil2cv2(source_image)
    # im = Image.open(source_image)
    # im = np.asarray(im)[:, :, ::-1]

    outputs = predictor(im)
    fields = outputs["instances"].get_fields()
    b_boxes = fields["pred_boxes"].tensor.tolist() # bounding boxes
    masks = fields["pred_masks"] # masks
    num_instances = len(b_boxes) # number of detected instances in the current image
    # print(num_instances)
    result_images = []
    if instance_segmentation:
        for i in range(0, num_instances):
            bb_i = list(map(int, b_boxes[i]))
            crop_im = im[bb_i[1]:bb_i[3], bb_i[0]:bb_i[2]]
            cur_mask = masks[i][bb_i[1]:bb_i[3], bb_i[0]:bb_i[2]]
            bin_seg_im = apply_mask(crop_im, cur_mask)
            bin_seg_im = Image.fromarray(np.uint8(bin_seg_im[:, :, ::-1]))
            # cv2.imwrite(result_path.format(i), bin_seg_im)
            if sum(bin_seg_im.convert("L").getextrema()) in (0, 2):
                bin_seg_im = None
            result_images.append(bin_seg_im)
    else:
        seg_im = apply_masks(im, masks)
        seg_im = Image.fromarray(np.uint8(seg_im[:, :, ::-1]))
        # seg_im.save(result_path)
        # cv2.imwrite(result_path, seg_im)
        if sum(seg_im.convert("L").getextrema()) in (0, 2):
            seg_im = None
        result_images.append(seg_im)


    return result_images, num_instances


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--source",
                        dest="src",
                        required=True,
                        help="directory containing source image")
    parser.add_argument("-d", "--destination",
                        dest="dest",
                        required=True,
                        help="destination directory for the cropped dataset")
    parser.add_argument("-c", "--crop",
                        nargs='?',
                        dest="crop",
                        const=True,
                        default=False,
                        help="should the image be croped")

    args = parser.parse_args()

    detectron_segment(args.src, args.dest, args.crop)

if __name__ == "__main__":
    main()
