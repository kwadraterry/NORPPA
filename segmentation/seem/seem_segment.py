import os
from PIL import Image
import sys
from pathlib import Path

file_folder = Path(__file__).resolve().parent
sys.path.append(str(file_folder))

import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from segmentation.seem.xdecoder.language.loss import vl_similarity

from segmentation.seem.xdecoder.BaseModel import BaseModel
from segmentation.seem.xdecoder import build_model
from segmentation.seem.utils.distributed import init_distributed
from segmentation.seem.utils.arguments import load_opt_from_config_files
from segmentation.seem.utils.constants import COCO_PANOPTIC_CLASSES
import cv2


from collections import namedtuple


def init_seem(conf_files="segmentation/seem/configs/seem/seem_focall_lang.yaml", model_path="."):

    Arguments = namedtuple('Arguments', 'conf_files')


    opt = load_opt_from_config_files(conf_files)
    opt = init_distributed(opt)


    # META DATA
    cur_model = 'None'
    if 'focalt' in conf_files:
        pretrained_pth = os.path.join(model_path, "seem_focalt_v2.pt")
        if not os.path.exists(pretrained_pth):
            os.system("wget -P {} {}".format(model_path, "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v2.pt"))
        cur_model = 'Focal-T'
    elif 'focal' in conf_files:
        pretrained_pth = os.path.join(model_path, "seem_focall_v1.pt")
        if not os.path.exists(pretrained_pth):
            os.system("wget -P {} {}".format(model_path, "https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
        cur_model = 'Focal-L'

    '''
    build model
    '''
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=True)


    t = []
    t.append(transforms.Resize(512, interpolation=Image.BICUBIC))
    transform = transforms.Compose(t)

    return (model, transform)

def interactive_infer_image(model, transform, image, reftxt):
    image_ori = transform(image)
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda()

    # stroke_inimg = None
    # stroke_refimg = None

    data = {"image": images, "height": height, "width": width}
    
    # inistalize task
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False

    model.model.task_switch['grounding'] = True
    data['text'] = [reftxt]

    batch_inputs = [data]
    results,image_size,extra = model.model.evaluate_demo(batch_inputs)

    pred_masks = results['pred_masks'][0]
    v_emb = results['pred_captions'][0]
    t_emb = extra['grounding_class']

    t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
    v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

    temperature = model.model.sem_seg_head.predictor.lang_encoder.logit_scale
    out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
    
    matched_id = out_prob.max(0)[1]
    pred_masks_pos = pred_masks[matched_id,:,:]
    pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]

    
    # interpolate mask to ori size
    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:], mode='bilinear')[0,:,:data['height'],:data['width']] > 0.0).float().cpu().numpy()

    torch.cuda.empty_cache()
    # return Image.fromarray(res), stroke_inimg, stroke_refimg
    return pred_masks_pos




@torch.no_grad()
def inference(image, text, model, transform):
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        res = interactive_infer_image(model, transform, image, text)
        print(res.shape)
        res = cv2.resize(res[0], dsize=image.size, interpolation=cv2.INTER_NEAREST)
        return np.asarray(image) * np.repeat(res[:,:,None], 3, axis=2)



def seem_segment(img, model, transform, object="seal"):
    # img = Image.open("/ekaterina/work/data/2_sides_viewpoint/1_left/49b80e88-0f90-4e25-b2ab-65ea549e43e6")

    res = inference(img, object, model, transform)

    # res_img = Image.fromarray((res[0, :, :] * 255).astype(np.uint8)).resize(img.size).convert("RGB")
    # Image.blend(img, res_img, 0.5)
    result_images = []
    result = Image.fromarray(res.astype(np.uint8))
    if sum(result.convert("L").getextrema()) in (0, 2):
        result = None
    result_images.append(result)

    return result_images
