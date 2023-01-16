from scipy.spatial.distance import cdist
from sklearn.decomposition import IncrementalPCA
from cyvlfeat.gmm import gmm
from cyvlfeat.fisher import fisher
from scipy.spatial.distance import cdist, cosine
import numpy as np
import os
import shutil
from PIL import Image
import io
from base64 import encodebytes

def cdist_std(x, y):
    return cdist(x, y, "cosine")


def calculate_dists(test_features, db_features, dist_func=cdist_std):
    dists = dist_func(test_features, db_features)
   
    inds = np.argsort(dists, axis=1)

    return (dists, inds)


def get_encoding_parameters(features, n_clusters=256, verbose=False):
    means, covars, priors, _, _ = gmm(features,
                                      n_clusters,
                                      init_mode='kmeans',
                                      verbose=verbose)
    return (means, covars, priors)


def encode_image(features, encoding_params):
    encoded = np.zeros((features.shape[0],
                        2 * features.shape[1]
                        * encoding_params[0].shape[0]))
    for i in enumerate(range(features.shape[0])):
        encoded[i, :] = fisher(features[i, :], *encoding_params, improved=True)
    return encoded

def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def power_normalize(v, p=0.5):
    return np.sign(v) * np.power(np.abs(v), p)

def encode_all_images(features, inds, encoding_params):
    img_inds = np.unique(inds)
    n = img_inds[-1] + 1
    encoded = np.zeros((n,
                        2 * features.shape[1]
                        * encoding_params[0].shape[0]))
    patch_features = [None] * n
    for i in img_inds:
        filt, = np.where(inds == i)
        patch_features[i] = features[filt, :]
        encoded[i, :] = fisher(patch_features[i], *encoding_params, improved=True)

    return encoded, patch_features


def get_topk_acc(labels_q, labels_db, indices, topk):
    top_labels = np.array(labels_db)[indices[:, :topk]]
    hits = (top_labels.T == labels_q).T
    return [sum((np.sum(hits[:, :j+1], axis=1) > 0)) / len(labels_q)
            for j in range(topk)]


def encode_pca(encoded, n_components=64, whiten=False):
    # print("Computing PCA...")
    pca = IncrementalPCA(n_components=n_components, whiten=True) #, svd_solver='full')
    # print("Applying PCA...")
    reduced = pca.fit_transform(encoded)
    return (reduced, pca)


def apply_pca(encoded, pca):
    return pca.transform(encoded)




def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def getImageBytes(filePath):
    img = Image.open(filePath, mode='r')
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG') 
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') 

    return encoded_img

def copy_files(src,dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    for dirpath, _, filenames in os.walk(src): 
        for filename in filenames:
            source = os.path.join(dirpath, filename)
            destination = os.path.join(dst, filename)
            shutil.copy(source,destination)