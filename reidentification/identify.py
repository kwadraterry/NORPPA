from skimage.measure import label
from sklearn.decomposition import KernelPCA
from skimage.morphology import convex_hull_image, skeletonize
from cyvlfeat.fisher import fisher
from PIL import Image
import math

import torch
from torchvision import transforms

import pickle
from reidentification.encoding_utils import *
from reidentification.geometric import *
import numpy as np
import torchvision.datasets as dset
import gc
from datasets import DatasetSlice
import pickle

import cv2
import itertools as it

from HessianAffinePatches import extract_hesaff_patches

from extract_patches.core import extract_patches as keypoints_to_patches

import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *

torch.autograd.set_grad_enabled(False)
from tqdm.autonotebook import tqdm

def get_patch_num(width, patch_size, step):
    return math.ceil((width - patch_size + 1)/step)

def crop_image(img, target, size):
    start_x, start_y = target

    end_x, end_y = start_x + size, start_y + size
    cropped = img.crop((start_x, start_y, end_x, end_y))

    return cropped


def thickness_resize(img, thickness=2):
    img0 = np.array(img) > 0
    area = np.sum(img0)
    img1 = skeletonize(img0)
    length = np.sum(img1)
    thickness_current = area / length
    ratio = thickness / thickness_current
    return (img.resize(tuple(int(i * ratio) for i in img.size), Image.ANTIALIAS), ratio)


def check_filled_area(min_val=0.15):
    def func(patch):
        return np.mean(patch > 0) > min_val
    return func


def check_largest_CC(min_val=100):
    def func(patch):
        labels = label(patch)
        if labels.max() == 0:
            return False
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return sum(largestCC) >= min_val
    return func


def check_convex_hull_area(min_val=0.5):
    def func(patch):
        if np.max(patch) == 0:
            return False
        chull = convex_hull_image(patch.copy())
        return np.mean(chull > 0) > min_val
    return func


def check_bounding_box_area(min_val=0.8):
    def func(patch):
        if np.max(patch) == 0:
            return False
        where = np.nonzero(patch)
        y1, x1 = np.amin(where, axis=1)
        y2, x2 = np.amax(where, axis=1)
        ratio = ((x2 - x1) * (y2 - y1)) / (patch.shape[-1] * patch.shape[-2])
        return ratio > min_val
    return func


def always_true(patch):
    return True


def test_patch(patch, filters):
    return np.all([f(patch) for f in filters])

def get_step_range(size, patch_size, step):
    fin = size-patch_size+1
    return [0] if fin < 1 else range(0, fin, step)


def extract_dense_patches(img, patch_size=48, step=24, final_size=48,
                           filter_funcs=[check_filled_area(),
                                         check_convex_hull_area()]):

    size = img.size
    result = []
    pos = []
    for x in get_step_range(size[0], patch_size, step):
        for y in get_step_range(size[1], patch_size, step):
            patch = crop_image(img, (x, y), patch_size)
            if patch_size != final_size:
                patch = patch.resize((final_size, final_size), Image.NONE)
            patch = np.asarray(patch)
            if test_patch(patch, filter_funcs):
                result.append(patch)
                pos.append((x, y))
    if len(result) == 0:
        return None
    result = np.stack(result, axis=0) 
    return result, pos


def apply_net(patch, net):
    with torch.no_grad():
        return net.forward(patch).detach().cpu().numpy()

def apply_batch_net(patches, net, batch_size=256):
    indices = np.append(np.arange(start=0, stop=patches.shape[0], step=batch_size), patches.shape[0])
    ind_pairs = list(zip(indices[:-1], indices[1:]))
    return np.concatenate([apply_net(patches[start:end,...], net) for (start, end) in ind_pairs])

def cvkeypoint_to_ell(keypoint):
    return [keypoint.pt[0], keypoint.pt[1], keypoint.size/2, keypoint.size/2, keypoint.angle * math.pi/180]

def extract_sift_patches(image, patch_size=32, sigma=1.6, nfeatures=480, nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=5, scale=2):
    image = np.array(image)
    sift = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold, sigma=sigma)
    keypoints = sift.detect(image)
    
    if scale != 1:
        for i in range(len(keypoints)):
            keypoints[i].size *= scale
    
    patches = np.array(keypoints_to_patches(keypoints, image, patch_size, sigma, 'cv2'))
    ells = np.array([np.array(cvkeypoint_to_ell(kp)) for kp in keypoints])
    
    return patches, ells


def patch_extraction(image, config):
    if config["use_hesaff"] or config.get("patch_extraction", "hesaff") == "hesaff":
        return extract_hesaff_patches(image, **config["hesaff_args"])
    elif config.get("patch_extraction", "hesaff") == "sift":
        return extract_sift_patches(image, **(config.get("sift_args", {})))
    else:
        return extract_dense_patches(image, **config["dense_args"])

def fisher_single(patch_features, cfg):
    codebooks = load_codebooks(cfg)
    encoding_params = codebooks["gmm"]
    encoded = fisher(patch_features, *encoding_params, improved=True)
    return encoded

def fisher_multiple(db_features, db_ids, cfg):
    codebooks = load_codebooks(cfg)
    encoding_params = codebooks["gmm"]
    _, indices = np.unique(db_ids, return_inverse=True)
    encoded, _ = encode_all_images(db_features, indices, encoding_params)
    return encoded

def stable_unique(a):
    indexes = np.unique(a, return_index=True)[1]
    return [a[index] for index in sorted(indexes)]

def group_by(dataset, group_label, patches):
    groups = stable_unique([labels[group_label] for (img, labels) in dataset])
    mapping = {group:i for (i,group) in enumerate(groups)}
    group_ids = len(dataset) * [None]
    group_labels = [{'labels':[], 'class_id':None} for _ in range(len(groups))]
    features, inds, all_ells = patches  
    # all_ells = np.array(all_ells)
        
        
    for (i, (img, label)) in enumerate(dataset):
        group_id = mapping[label[group_label]]
        group_ids[i] = group_id
        label['id'] = i
        filt, = np.where(inds == i)
        label['features'] = features[filt, :]
        label['ellipses'] = all_ells[i]
        group_labels[group_id]['labels'].append(label)
        group_labels[group_id]['class_id'] = label['class_id']
    return group_ids, group_labels

def getKeyNetAffNetHardNet(num_features=5000, upright=False, scale_laf=1.0):
    def init(device):
        return KF.KeyNetAffNetHardNet(num_features=num_features, upright=upright, device=device, scale_laf=scale_laf)
    def apply(image, detector, dataset_transforms, device):
        image = dataset_transforms(image)
        image = np.array(image)[None, :, :, None]
        timg = K.image_to_tensor(image, False).float()/255.
        timg = timg.to(device)
        
        lafs, _, descs = detector(timg)        

        kps_back = opencv_kpts_from_laf(lafs, scale_laf)
        patch_features = descs.cpu()[0, ]
        ells = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.size, kp.angle] for kp in kps_back])
        return patch_features, ells
    return init, apply
        
def getDISK(pretrained='depth'):
    def init(device):
        return KF.DISK.from_pretrained(pretrained, device=device)
    def apply(image, detector, dataset_transforms, device):
        image = np.array(image.convert("RGB"))[None, :, :]
        timg = K.image_to_tensor(image, False).float()/255.
        timg = timg.to(device)
        
        disk = detector(timg, pad_if_not_divisible=True)[0]
        patch_features = disk.descriptors.cpu().numpy()
        pts = disk.keypoints.cpu().numpy()
        ells = np.array([[kp[0], kp[1], 5, 5, 0] for kp in pts])
        return patch_features, ells
    return init, apply

def getHessAffNetHardNet(cfg):
    def init(device):
        return cfg["net"]
    def apply(image, detector, dataset_transforms, device):
        image = dataset_transforms(image)
        if sum(image.getextrema()) == 0:
            return np.array([]), np.array([])
        
        patches, ells = patch_extraction(image, cfg)
        if patches is None or len(patches) == 0:
            return np.array([]), np.array([])
        patch_features = torch.from_numpy(patches/255).float().unsqueeze(1)
        if cfg["use_cuda"]:
            patch_features = patch_features.cuda()
        patch_features = apply_batch_net(patch_features, detector, batch_size=cfg["batch_size"])
        return patch_features, ells
    return init, apply

def safe_vstack(ms):
    if len(ms) == 0:
        return np.zeros((0,0))
    else:
        return np.vstack(ms)

def patchify(dataset, config, init_apply=None):
    if init_apply is None:
        init_apply = getHessAffNetHardNet(config)
    result = []
    labels = []
    inds = []
    all_ells = []
    dataset_transforms = transforms.Grayscale(num_output_channels=1)

    init, apply = init_apply
    
    device = torch.device('cuda') if config['use_cuda'] else torch.device('cpu')
    
    print(f"Using device {device}")
    detector = init(device)
    
    for i, (image, img_label) in enumerate(tqdm(dataset)):
        if image is None or sum(dataset_transforms(image).getextrema()) == 0:
            all_ells.append(None)
            labels.append(img_label)
            continue
        
        patch_features, ells = apply(image, detector, dataset_transforms, device)
        
        all_ells.append(np.array(ells)/image.width)
        inds.extend([i] * patch_features.shape[0])
        labels.append(img_label)
        if len(patch_features) > 0:
            result.append(patch_features)

    labels = np.array(labels)
    return safe_vstack(result), np.array(inds), labels, all_ells

def extract_patches(dataset, config, init_apply=None):
    return (dataset, patchify(dataset, config, init_apply))

def extract_patches_single(input, config, init_apply=None):
    return extract_patches([input], config, init_apply)

def _encode_patches(dataset_patches, config, codebooks=None, group_label='file'):
    (dataset, (features, inds, labels, ellipses)) = dataset_patches
    if len(features) == 0:
        return None, None, None
    print("Calculating PCA")
    if codebooks is None:
        features, pca = encode_pca(features, n_components=config["pca"], whiten=True)
    else:
        features = apply_pca(features, codebooks["pca"])
    group_ids, group_labels = group_by(dataset, group_label, (features, inds, ellipses))
        
    group = (group_ids is not None) and (group_labels is not None)    
    if group and len(group_ids)>1:
        groups = np.array(group_ids).squeeze()
        updated_inds = groups[inds]
    else:
        updated_inds = inds
    labels = np.array(group_labels)
    
    print("Getting encoding parameters...", flush=True)
    if codebooks is None:
        encoding_params = get_encoding_parameters(features, n_clusters=config["n_clusters"], verbose=True)
    else:
        encoding_params = codebooks["gmm"]

    print("Encoding...")
    features, patch_features = encode_all_images(features, updated_inds, encoding_params)

    kpca = None
    if config["use_kpca"]:
        if codebooks["kpca"] is None:
            kpca = KernelPCA(n_components=None, kernel=config["kernel"], remove_zero_eig=True)
            features = kpca.fit_transform(features)
        elif codebooks["kpca"] is not None:
            features = codebooks["kpca"].transform(features)

    if codebooks is None:
        codebooks = {'pca': pca, 'gmm': encoding_params, 'kpca': kpca}
    
    return features, labels, codebooks
 

def do_matching(test_feats, db_feats, percentile=10, max_len=200):
    dists, sorted_inds = calculate_dists(test_feats, db_feats)
    sorted_dists = np.take_along_axis(dists, sorted_inds, axis=1)
#     print(sorted_dists.shape)
    if test_feats.size == 0 or db_feats.size == 0:
        return (np.array([]), np.array([]), np.array([]))
    
    mean_dist = np.percentile(sorted_dists[:, 0], percentile)
    filt = sorted_dists[:, 0] <= mean_dist
    sorted_inds = sorted_inds[filt, 0]
    sorted_dists = sorted_dists[filt, 0]
    
    similarity = (np.max(sorted_dists) - sorted_dists) / np.max(sorted_dists)
    filt = np.nonzero(filt)[0]
    similarity[np.isnan(similarity)] = 1.0

    if len(similarity) > max_len:
        similarity = similarity[:max_len]
        sorted_inds = sorted_inds[:max_len]
        filt = filt[:max_len]
    return (filt, sorted_inds, similarity)

def do_matching_geom(test_feats, test_patches, db_feats, db_patches, percentile=10):
    pass

def match_topk(test_features, db_features, topk, leave_one_out=False):
    dists, inds = calculate_dists(test_features, db_features, leave_one_out=leave_one_out)
    sorted_inds = np.argsort(dists, axis=1)
    dists = np.take_along_axis(dists, sorted_inds, axis=1)
    return dists[:, :topk], sorted_inds[:, :topk]


def encode_single(input, cfg, group_label='file', init_apply=None, compute_codebooks=False):
    return encode_dataset([input], cfg, group_label, init_apply, compute_codebooks)



def encode_pipeline(input, cfg):
    if input[0] is None:
        return input
    return encode_dataset([input], cfg)

def encode_dataset(dataset, cfg, group_label='file', init_apply=None, compute_codebooks=False):
    patches = extract_patches(dataset, cfg, init_apply)
    return encode_patches(patches, cfg, group_label=group_label, compute_codebooks=compute_codebooks)

def encode_patches(dataset, cfg, group_label='file', compute_codebooks=False):
    if compute_codebooks:
        codebooks = None
    else:
        codebooks = load_codebooks(cfg)
    query_features, query_labels, codebooks = _encode_patches(dataset, cfg, codebooks, group_label)
    if query_features is None:
        return [None]
    if compute_codebooks:
        return (codebooks, list(zip(query_features, query_labels)))
    else:
        return list(zip(query_features, query_labels))
    
def encode_patches_single(input, cfg, group_label='file', compute_codebooks=False):
    return encode_patches([input], cfg, group_label, compute_codebooks)

def identify_single(query, database):
    return identify([query], database,)



def get_fisher_vectors(db):
    if hasattr(db, 'get_fisher_vectors'):
        return db.get_fisher_vectors()
    else:
        return np.concatenate([f[np.newaxis,...] for (f, _) in db])
    

def get_label(db, i):
    if hasattr(db, 'get_label'):
        return db.get_label(i)
    else:
        return db[i][1]

def reaggregate_class(class_samples, cfg):
    class_id = class_samples[0][1]['class_id']
    labels = [x[1]['labels'][0] for x in class_samples]

    features = np.vstack([x['features'] for x in labels])
    codebooks = load_codebooks(cfg)
    encoding_params = codebooks["gmm"]
    agg_fisher = aggregate_features(features, encoding_params) # 

    return (agg_fisher, {'class_id':class_id, 'labels':labels})

def identify_many_leave_one_out(query, cfg, database=None, topk=5):
    if query is None:
        return [None]
    query = [x for x in query if x is not None and x[0] is not None]
    if len(query) == 0:
        return [None]
    
    dists = []
    request_ids = []

    all_matches = []
    
    if database is None:
        database = query
    def add_fisher_field(label, fisher):
        label["fisher"] = fisher
        return label
    codebooks = load_codebooks(cfg)
    encoding_params = codebooks["gmm"]
    query_labels = []
    db_features = get_fisher_vectors(database)

    for (di, (database_exclude_fisher, database_exclude)) in tqdm(enumerate(database), total=len(database), position=0):
        if len(database_exclude["labels"]) == 1:
            continue
        for (qj, query_label) in tqdm(enumerate(database_exclude["labels"]), total=len(database_exclude["labels"]), leave=False, position=1):
            class_id = query_label['class_id']
            query_features = aggregate_features(query_label['features'], encoding_params)
            query_label = {"labels":[query_label], "class_id":class_id, "fisher":query_features}

            new_db_labels = database_exclude["labels"][:qj] + database_exclude["labels"][qj+1:]
            new_db_label = {"labels":new_db_labels, "class_id":class_id}
            new_db_class_features = np.vstack([l['features'] for l in new_db_labels])
            new_db_class_features = aggregate_features(new_db_class_features, encoding_params)
            # new_database = database[:di] + [(new_db_class_features, new_db_label)] + database[di+1 :]
            database[di] = (new_db_class_features, new_db_label)
            
            # new_db_features = get_fisher_vectors(database)
            db_features[di, ...] = new_db_class_features
            
            
            dists, request_ids = match_topk(query_features[np.newaxis,...], db_features, topk)
            
            matches = [None] * request_ids.shape[0]
            for i in range(request_ids.shape[0]):
                matches[i] = [None] * request_ids.shape[1]
                for j in range(request_ids.shape[1]):
                    matches[i][j] = {"db_label": add_fisher_field(get_label(database, request_ids[i, j]), db_features[request_ids[i, j]]), "distance": dists[i, j]}
            all_matches.extend(matches)
            query_labels.append(query_label)
        database[di] = (database_exclude_fisher, database_exclude)
        db_features[di, ...] = database_exclude_fisher

    return list(zip(all_matches, query_labels))


    
def identify(query, database=None, topk=5, leave_one_out=False):
    if query is None:
        return [None]
    query = [x for x in query if x is not None and x[0] is not None]
    if len(query) == 0:
        return [None]
    
    if database is None:
        database = query
    query_features = np.concatenate([f[np.newaxis,...] for (f, _) in query])
    
    def add_fisher_field(label, fisher):
        label["fisher"] = fisher
        return label
    query_labels = [add_fisher_field(l, f) for (f, l) in query]
    
    db_features = get_fisher_vectors(database)
    
    dists, request_ids = match_topk(query_features, db_features, topk, leave_one_out=leave_one_out)
    
    matches = [None] * request_ids.shape[0]
    for i in tqdm(range(request_ids.shape[0])):
        matches[i] = [None] * request_ids.shape[1]
        for j in range(request_ids.shape[1]):
            matches[i][j] = {"db_label": add_fisher_field(get_label(database, request_ids[i, j]), db_features[request_ids[i, j]]), "distance": dists[i, j]}
    
    return list(zip(matches, query_labels))


def apply_geometric(input, params):
    if input is None:
        return [input]
    matches, query_labels = input
    order = re_evaluate(matches, params)
    for est, mask, k in order:
        matches[k]["Geom_Est"] = est
        matches[k]["Mask"] = mask
    matches = [matches[k] for _, _, k in order]
    return [(matches, query_labels)]


def apply_geometric_fisher(input, cfg):
    if input is None:
        return [input]
    matches, query_labels = input
    order = re_evaluate_fisher(matches, query_labels, cfg)
    for est, mask, k in order:
        matches[k]["Geom_Est"] = est
        matches[k]["Mask"] = mask
    matches = [matches[k] for _, _, k in order]
    return [(matches, query_labels)]



    
def create_sql_database(dataset, cfg, db_components, seal_type="norppa", compute_codebooks=False):
    create_database_table(cfg["conn"])
    create_patches_table(cfg["conn"])

    if compute_codebooks:
        codebooks = None
    else:
        codebooks = load_codebooks(cfg)

    db_features, db_labels, patch_features, patches = db_components

    now = datetime.now()
    now = now.strftime("%d-%m-%YT%H:%M:%S")
    for i, (image, image_label) in enumerate(dataset):
        img_id = insert_database(cfg["conn"],image_label["file"], db_labels[i]["class_id"], seal_type, db_features[i, ...], now)
        if patches[i] is not None:
            for j, patch in enumerate(patches[i]):
                insert_patches(cfg["conn"], img_id, patch, patch_features[i][j, ...])
    del db_features
    del patch_features
    del patches
    del db_labels
    gc.collect()
    cfg["conn"].commit()



