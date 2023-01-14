import numpy as np 
import skimage.transform as trans
from skimage.morphology import skeletonize
from PIL import Image
from math import ceil, floor



def crop(img_path, flag_multi_class=False):
    img = np.asarray(Image.open(img_path).convert('L'))
    size_y, size_x = img.shape

    where = np.where(img!=0)
    y1, x1 = np.amin(where, axis=1)
    y2, x2 = np.amax(where, axis=1)

    x = x2 - x1
    y = y2 - y1

    z = ceil((x + y) / 2)
    z = max(x,y)
    x1 = max(x2 - floor(x/2) - floor(z/2), 0)
    x2 =  x2 - floor(x/2) + ceil(z/2)
    y1 =  max(y2 - floor(y/2) - floor(z/2), 0)
    y2 =  y2 - floor(y/2) + ceil(z/2)


    source_cropped = img[y1:y2, x1:x2]
    xx, yy = source_cropped.shape


    source_cropped = np.pad(source_cropped, ((0, max(z-xx,0)),(0, max(z-yy,0))), 'constant', constant_values=(0))

    s = source_cropped.shape
    source_cropped = trans.resize(source_cropped, [512,512])
    img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
    img = np.reshape(img,(1,)+img.shape)
    return (source_cropped, img_path, s, [max(z-xx,0),max(z-yy,0)], [y1, max(size_y-y2,0), x1, max(size_x-x2,0)])



def smart_resize(img, size, return_ratio=False):
    ratio = size / max(img.size)
    result = img.resize(tuple(int(i * ratio) for i in img.size), Image.ANTIALIAS)
    if return_ratio:
        return (result, ratio)
    else:
        return result

def thickness_resize(img, thickness=2, return_ratio=False):
    if img.mode != 'L':
        img = img.convert('L')
    img0 = np.array(img) > 0
    area = np.sum(img0)
    if area==0:
        return smart_resize(img, 300, return_ratio)
    img1 = skeletonize(img0)
    length = np.sum(img1)
    
    thickness_current = area / length
    ratio = thickness / thickness_current
    resized_img = img.resize(tuple(int(i * ratio) for i in img.size), Image.ANTIALIAS)
    if return_ratio:
        return (resized_img, ratio)
    else:
        return resized_img


def postprocess(img, confidence, initial_size, return_ratio=False):
    img = trans.resize(img, initial_size[::-1])
    img = ((img>confidence)*255).astype('uint8')
    if (np.sum(img) < 0.01* np.prod(initial_size)):
        return None
    img = Image.fromarray(img)
    return thickness_resize(img, return_ratio=return_ratio)

