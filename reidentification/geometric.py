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
# from reidentification.identify import fisher_single
import numpy as np
import torchvision.datasets as dset
import gc
from datasets import DatasetSlice
import pickle

import cv2
import itertools as it

from HessianAffinePatches import extract_hesaff_patches



def reestimate_distance(match, inliers, query_labels, cfg):
    db_ind = match['db_ind']
    query_inds, db_inds = match['inds'] 
    db_label = match['db_label']['labels'][db_ind]
    features = db_label['features'][db_inds, :]
    
    features = features[inliers[:, 0], :]
    db_fisher = fisher_single(features, cfg)


    query_ind = match['query_ind']
    query_label = query_labels['labels'][query_ind]
    features = query_label['features'][query_inds, :]
    features = features[inliers[:, 0], :]
    query_fisher = fisher_single(features, cfg)

    (dist, _) = calculate_dists([query_fisher], [db_fisher])

    return dist[0]

def re_evaluate_fisher(matches, query_labels, cfg):
    # for each match, get image_features, inliers, find feature for inliers, aggregate to fisher vector and calculate distance
    est_cfg = cfg["geometric"]
    inliers = geometric_verification(matches, est_cfg)
    
    if len(inliers) == 0:
        return matches
    
    # order = [(est_cfg["estimator"](dist, mask), mask, i) for i, (dist, mask) in enumerate(zip(matches, inliers))]
    order = [(reestimate_distance(match, match_inliers, query_labels, cfg), match_inliers, i) for i, (match, match_inliers) in enumerate(zip(matches, inliers))]
    order.sort(key = lambda x: (x[0], x[2]))
    
    return order

# Re-orders original results. Returns the new order of indices.
def re_evaluate(matches, est_cfg):

    dists = [match["distance"] for match in matches]
    inliers = geometric_verification(matches, est_cfg)
    
    if len(inliers) == 0:
        return matches
    
    order = [(est_cfg["estimator"](dist, mask), mask, i) for i, (dist, mask) in enumerate(zip(dists, inliers))]
    order.sort(key = lambda x: (x[0], x[2]))
    
    return order


# Returns the logical array presenting inlier point correspondences, inliers set to 1 and outliers set to 0.
def geometric_verification(matches, est_cfg):
    
    qr_patches_all = [match["patches"][0] for match in matches]
    db_patches_all = [match["patches"][1] for match in matches]
    
    qr_coordinates, db_coordinates = get_coordinates(qr_patches_all,
                                                     db_patches_all)
    homographies, inliers = estimate_homographies(qr_coordinates,
                                                  db_coordinates,
                                                  est_cfg)
    
    return inliers


def safe_max(x, *args, **kw_args):
    if len(x) == 0:
        return x
    else:
        return max(x, *args, **kw_args)

def safe_mean(x, *args, **kw_args):
    if len(x) == 0:
        return x
    else:
        return np.mean(x, *args, **kw_args) 
        
# Extracts the x,y point correspondences and translate and scale point sets inside unit circle.
def get_coordinates(qr_patches_all, db_patches_all):
    
    # get xy-pairs
    qr_all = np.array([np.array([[qr[0], qr[1]] for qr in qr_patches]) 
                       for qr_patches in qr_patches_all], dtype=object)
    
    db_all = np.array([np.array([[db[0], db[1]] for db in db_patches]) 
                       for db_patches in db_patches_all], dtype=object)
    
    # translate to origin
    qr_mean = np.array([safe_mean(qr_coords, axis=0) for qr_coords in qr_all], dtype=object)
    db_mean = np.array([safe_mean(db_coords, axis=0) for db_coords in db_all], dtype=object)
    for i, (qr, db) in enumerate(zip(qr_mean, db_mean)):
        qr_all[i] = qr_all[i] - qr
        db_all[i] = db_all[i] - db
    
    # set |p| <= 1
    max_l_qr = [safe_max(qr, key=lambda p: np.linalg.norm(p)) for qr in qr_all]
    max_l_db = [safe_max(db, key=lambda p: np.linalg.norm(p)) for db in db_all]
    for i, (qr, db) in enumerate(zip(max_l_qr, max_l_db)):
        if len(qr) == 0 or len(db) == 0:
            continue
        a, b = np.linalg.norm(qr), np.linalg.norm(db)
        qr_all[i] /= a if a > np.finfo(float).eps else 1
        db_all[i] /= b if b > np.finfo(float).eps else 1
    
    return qr_all, db_all


def save_findHomography(qr_coords,db_coords,est_cfg):
    if len(qr_coords)< 4 or len(db_coords) < 4:
        return np.eye(3),np.full((len(qr_coords), 1), True)
    return cv2.findHomography(qr_coords.astype(np.double),
                               db_coords.astype(np.double),
                               method=est_cfg["method"],
                               ransacReprojThreshold=est_cfg["max_reproj_err"],
                               maxIters=est_cfg["max_iters"]) 

# Finds homographies for each query-database image pairs.
def estimate_homographies(qr_coords_all,
                          db_coords_all,
                          est_cfg):

    models = [
        save_findHomography(qr_coords,db_coords,est_cfg)
        for qr_coords, db_coords in zip(qr_coords_all, db_coords_all)
    ]
    
    
    homographies = [H for H, _ in models]
    inliers = [I for _, I in models]
    
    return homographies, inliers

