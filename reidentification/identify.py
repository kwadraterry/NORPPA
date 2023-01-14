from skimage.measure import label
from skimage.morphology import convex_hull_image, skeletonize
from cyvlfeat.fisher import fisher
from PIL import Image
import math

from sql import *

import torch
from torchvision import transforms

import pickle
from reidentification.encoding_utils import *
import numpy as np
import torchvision.datasets as dset
import gc
from datasets import DatasetSlice
import pickle

from HessianAffinePatches import extract_hesaff_patches

torch.autograd.set_grad_enabled(False)
from tqdm import tqdm


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

def patch_extraction(image, config):
    if config["use_hesaff"]:
        return extract_hesaff_patches(image, **config["hesaff_args"])
    else:
        return extract_dense_patches(image, **config["dense_args"])

def patchify(dataset, config):
    net = config["net"]
    result = []
    labels = []
    inds = []
    all_ells = []
    ind = 0
    num_files = len(dataset)
    dataset_transforms = transforms.Grayscale(num_output_channels=1)
    
    for i, (image, img_label) in enumerate(tqdm(dataset)):
        image = dataset_transforms(image)
        num_files-=1
        if sum(image.getextrema()) == 0:
            continue
        patches, ells = patch_extraction(image, config)

        if patches is None or len(patches) == 0:
            continue
        patch_features = torch.from_numpy(patches/255).float().unsqueeze(1)
        if config["use_cuda"]:
            patch_features = patch_features.cuda()
        patch_features = apply_batch_net(patch_features, net)
        all_ells.append(ells)
        inds.extend([ind] * patch_features.shape[0])
        ind += 1
        labels.append(img_label)
        result.append(patch_features)
        del patches
        gc.collect()
    labels = np.array(labels)
    return np.vstack(result), np.array(inds), labels, all_ells




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
                               
                               

def _encode_dataset(dataset, config, codebooks=None, sequence_ids=None, sequence_labels=None):
    
    features, inds, labels, all_ells = patchify(dataset, config)
    
    print("Calculating PCA")
    if codebooks is None:
        features, pca = encode_pca(features, n_components=config["pca"], whiten=True)
    else:
        features = apply_pca(features, codebooks["pca"])
        
    try:
        with open("patches_nn_tonemapped.pickle", 'wb') as f_file:
            pickle.dump((features, inds, labels, all_ells), f_file, protocol=4)
    except Exception as e:
        print(e)
        
    seq = (sequence_ids is not None) and (sequence_labels is not None)    
    if seq:
        seqs = np.array(sequence_ids).squeeze()
        inds = seqs[inds]

    print("Getting encoding parameters...")
    if codebooks is None:
        encoding_params = get_encoding_parameters(features, n_clusters=config["n_clusters"], verbose=False)
    else:
        encoding_params = codebooks["gmm"]

    print("Encoding...")
    features, patch_features = encode_all_images(features, inds, encoding_params)
    

    kpca = None
    if config["use_kpca"]:
        if codebooks["kpca"] is None:
            kpca = KernelPCA(n_components=None, kernel=config["kernel"], remove_zero_eig=True)
            features = kpca.fit_transform(features)
        elif codebooks["kpca"] is not None:
            features = codebooks["kpca"].transform(features)
    if seq:        
        seq_labels = np.array(sequence_labels).squeeze()
        labels = [seq_labels[i] for (i, feature) in enumerate(features)]

    if codebooks is None:
        codebooks = {'pca': pca, 'gmm': encoding_params, 'kpca': kpca}
    
    return features, labels, patch_features, all_ells, codebooks

    


def do_matching(test_feats, db_feats, percentile=10):
    dists, sorted_inds = calculate_dists(test_feats, db_feats)
    sorted_dists = np.take_along_axis(dists, sorted_inds, axis=1)
    mean_dist = np.percentile(sorted_dists[:, 0], percentile)
    filt = sorted_dists[:, 0] <= mean_dist
    sorted_inds = sorted_inds[filt, 0]
    sorted_dists = sorted_dists[filt, 0]
    
    similarity = (np.max(sorted_dists) - sorted_dists) / np.max(sorted_dists)
    filt = np.nonzero(filt)[0]
    similarity[np.isnan(similarity)] = 1.0
    max_len = 200
    if len(similarity) > max_len:
        similarity = similarity[:max_len]
        sorted_inds = sorted_inds[:max_len]
        filt = filt[:max_len]
    return (filt, sorted_inds, similarity)

def match_topk(test_features, db_features, topk):
    dists, inds = calculate_dists(test_features, db_features)
    sorted_inds = np.argsort(dists, axis=1)
    dists = np.take_along_axis(dists, sorted_inds, axis=1)
    return dists[:, :topk], sorted_inds[:, :topk]

def load_codebooks(cfg):
    if cfg['codebooks'] is None:
        with open(cfg['codebooks_path'], "rb") as codebooks_file:
            cfg['codebooks'] = pickle.load(codebooks_file)
    return cfg['codebooks']

def create_database(dataset, cfg, compute_codebooks=False):

    if compute_codebooks:
        codebooks = None
    else:
        codebooks = load_codebooks(cfg)

    features, labels, patch_features, all_ells, codebooks = _encode_dataset(dataset, cfg, codebooks)

    classes = labels
    if hasattr(dataset,'classes'):
        classes = [dataset.classes[i] for i in labels]
    return (features, classes, patch_features, all_ells), codebooks


def encode_single(image, cfg):
    if image is None:
        return image
    dataset_transforms = transforms.Grayscale(num_output_channels=1)

    image = dataset_transforms(image)

    return encode_dataset([(image, 0)], cfg)[0][0]



def encode_pipeline(input, cfg):
    if input[0] is None:
        return input
    return encode_dataset([input], cfg)

def encode_dataset(dataset, cfg):
    codebooks = load_codebooks(cfg)
    query_features, query_labels, query_patch_features, query_patches, _ = _encode_dataset(dataset, cfg, codebooks)
    return list(zip(list(zip(query_features, query_patch_features, query_patches)), query_labels))

def encode_sequence_dataset(dataset, cfg, sequence_ids=None, sequence_labels=None, sequence_dataset=None):
    if sequence_dataset is None:
        sequence_dataset = dataset
    if sequence_ids is None:
        sequence_ids = sequence_dataset.get_sequence_ids()
    if sequence_labels is None:
        sequence_labels = sequence_dataset.get_sequence_labels()
    codebooks = load_codebooks(cfg)
    query_features, query_labels, query_patch_features, query_patches, _ = _encode_dataset_no_save(dataset, cfg, codebooks, sequence_ids, sequence_labels)
    return list(zip(list(zip(query_features, query_patch_features, query_patches)), query_labels))


def identify_single(query, database, cfg):
    return identify([query], database, cfg)

# def identify(query, database, cfg):
#     query_features = np.concatenate([f[np.newaxis,...] for ((f, _, _), _) in query])
#     query_patch_features = [f for ((_, f, _), _) in query]
#     query_patches = [p for ((_, _, p), _) in query]
#     query_labels = [l for ((_, _, _), l) in query]
    
#     dists, request_ids = match_topk(query_features, database.get_fisher_vectors(), cfg["topk"])
    
#     patch_matches = [None] * request_ids.shape[0]
#     for i in tqdm(range(request_ids.shape[0])):
#         patch_matches[i] = [None] * request_ids.shape[1]
#         for j in range(request_ids.shape[1]):
#             db_patch_features, db_patches = database.get_patches(request_ids[i, j])
#             (filt, sorted_inds, similarity) = do_matching(query_patch_features[i], db_patch_features)
#             patch_matches[i][j] = {"db_label": database.get_label(request_ids[i, j]), "distance": dists[i, j]}
#             patch_matches[i][j]["matches"] = [
#                 [query_patches[i][k].tolist() for k in filt],
#                 [db_patches[k].tolist() for k in sorted_inds],
#                 similarity.tolist()
#             ]
    
#     return list(zip(patch_matches, query_labels))

def get_fisher_vectors(db):
    if hasattr(db, 'get_fisher_vectors'):
        return db.get_fisher_vectors()
    else:
        return np.concatenate([f[np.newaxis,...] for ((f, _, _), _) in db])
    
def get_patches(db, i):
    if hasattr(db, 'get_patches'):
        return db.get_patches(i)
    else:
        return db[i][0][1:]

def get_label(db, i):
    if hasattr(db, 'get_label'):
        return db.get_label(i)
    else:
        return db[i][1]

def identify(query, database, cfg):
    query_features = np.concatenate([f[np.newaxis,...] for ((f, _, _), _) in query])
    query_patch_features = [f for ((_, f, _), _) in query]
    query_patches = [p for ((_, _, p), _) in query]
    query_labels = [l for ((_, _, _), l) in query]
    
    dists, request_ids = match_topk(query_features, get_fisher_vectors(database), cfg["topk"])
    
    patch_matches = [None] * request_ids.shape[0]
    for i in tqdm(range(request_ids.shape[0])):
        patch_matches[i] = [None] * request_ids.shape[1]
        for j in range(request_ids.shape[1]):
            db_patch_features, db_patches = get_patches(database, request_ids[i, j])
            patch_matches[i][j] = {"db_label": get_label(database, request_ids[i, j]), "distance": dists[i, j]}
#             patch_matches[i][j]["matches"] = [
#                 [query_patches[i][k].tolist() for k in filt],
#                 [db_patches[k].tolist() for k in sorted_inds],
#                 similarity.tolist()
#             ]
    
    return list(zip(patch_matches, query_labels))


def identify_many(query, database, cfg):
    query_features = np.concatenate([f[np.newaxis,...] for ((f, _, _), _) in query])
    query_patch_features = [f for ((_, f, _), _) in query]
    query_patches = [p for ((_, _, p), _) in query]
    query_labels = [l for ((_, _, _), l) in query]
    
    dists, request_ids = match_topk(query_features, database.get_fisher_vectors_seq(), cfg["topk"])
    
    patch_matches = [None] * request_ids.shape[0]
    for i in tqdm(range(request_ids.shape[0])):
        patch_matches[i] = [None] * request_ids.shape[1]
        for j in range(request_ids.shape[1]):
            db_patch_features, db_patches = database.get_patches_seq(request_ids[i, j])
            patch_matches[i][j] = {"db_label": database.get_label_seq(request_ids[i, j]), "distance": dists[i, j]}
    
    return list(zip(patch_matches, query_labels))

def find_matching_regions():
    (filt, sorted_inds, similarity) = do_matching(query_patch_features[i], db_patch_features)
    patch_matches[i][j]["matches"] = [
                [query_patches[i][k].tolist() for k in filt],
                [db_patches[k].tolist() for k in sorted_inds],
                similarity.tolist()
            ]
    

def encode_fast(dataset, codebooks, config):
    features, inds, labels, all_lafs = patchify(
        dataset, config)
    # test_classes = []
    # if hasattr(dataset_test,'classes'):
    #     test_classes = [dataset_test.classes[i] for i in test_labels]
   
    pca = codebooks["pca"]

    features1 = apply_pca(features, pca)

   
    features2, patch_features = encode_all_images(features1, inds, codebooks["gmm"])

    del features1, features

    return features2, patch_features, all_lafs, labels, np.unique(inds)
    
def create_database(data_dir, cfg, seal_type="norppa", batch_size=512):
    create_database_table(cfg["conn"])
    create_patches_table(cfg["conn"])

    dataset_transforms = transforms.Grayscale(num_output_channels=1)

    dataset_full = dset.ImageFolder(root=data_dir,
                                transform=dataset_transforms
                                )
    batch_num = math.ceil(len(dataset_full)/batch_size)

    conn = cfg["conn"]

    # if (task_id is not None):
    #     update_status_val(conn, task_id,0)

    codebooks = load_codebooks(cfg)

    for i in range(batch_num):
        # print(f"Starting batch {i+1}/{batch_num}")
        slice = (i*batch_size, min(len(dataset_full), (i+1)*batch_size))
        print(slice)
        print(len(dataset_full))
        dataset = DatasetSlice(dataset_full, slice)
       
            
        db_features, patch_features, patches, db_labels, inds  = encode_fast(dataset, codebooks, cfg)

        
        db_classes = []
        if hasattr(dataset,'classes'):
            db_classes = [dataset.classes[i] for i in db_labels]

        now = datetime.now()
        now = now.strftime("%d-%m-%YT%H:%M:%S")
        # for i, image in enumerate(dataset):
        for i in inds:
            # print(f"i: {i}, dataset: {len(dataset)}")
            # print(f"imgs: {len(dataset.imgs)}, db_classes: {len(db_classes)}")
            # print(f"imgs i: {dataset.imgs[i]}")
            
            img_id = insert_database(cfg["conn"],dataset.imgs[i][0], db_classes[i], seal_type, db_features[i, ...], now)
            for j, patch in enumerate(patches[i]):
                insert_patches(cfg["conn"], img_id, patch, patch_features[i][j, ...])
        del db_features
        del patch_features
        del patches
        del db_labels
        gc.collect()
        conn.commit()