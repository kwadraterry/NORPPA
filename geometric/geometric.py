import numpy as np
import cv2
import itertools as it


def get_data(identification_results, cfg, query_dataset, D, A, N):

    for d, a, in it.product(D, A):
        est_cfg = {
            "method": cv2.RANSAC,
            "max_reproj_err": d,
            "max_iters": N,
            "estimator": lambda m, d: (d) * (1 - m) ** a,
        }
        new = re_evaluate(identification_results, est_cfg)
        db_labels = np.array([[x['db_label']['class_id'] for x in y[0]] for y in new])
        q_labels = np.array(query_dataset.get_labels())

        hits = (db_labels.T == q_labels).T
        
        probs_new = [sum((np.sum(hits[:, :j+1], axis=1) > 0)) / len(q_labels) 
                     for j in range(cfg["topk"])]
        print(f"{d:.2f} {a:.2f}", end="; ")
        print(" ".join(f"{p*100:.2f}" for p in probs_new))
    

def re_evaluate(identification_results, est_cfg):
    new = identification_results.copy()
    for i, result in enumerate(new):
        
        matches, _ = result
        dists = [match["distance"] for match in matches]
    
        inliers = geometric_verification(result, est_cfg)
        
        means = [np.mean(mask) for mask in inliers]
        
        #probs = [1 - (1 - np.mean(mask)**len(mask))**est_cfg["max_iters"] 
                 #for mask in inliers]
        
        evaluations = sorted([(est_cfg["estimator"](m, d), i)
                              for i, (m, d) in enumerate(zip(means, dists))])
        
        new[i] = list(new[i])
        new[i][0] = [new[i][0][k] for _, k in evaluations]
        
    return new


# Returns the probabilities of unsuccessful homography
def geometric_verification(identification_result, est_cfg):
    
    matches, query_label = identification_result
    
    qr_patches_all = [match["matches"][0] for match in matches]
    db_patches_all = [match["matches"][1] for match in matches]
    # similarities = [match["matches"][2] for match in matches]
    
    qr_coordinates, db_coordinates = get_coordinates(qr_patches_all,
                                                     db_patches_all)
    
    homographies, inliers = estimate_homographies(qr_coordinates,
                                                  db_coordinates,
                                                  est_cfg) 
    
    return inliers

def get_coordinates(qr_patches_all, db_patches_all):
    
    # get xy-pairs
    qr_all = np.array([np.array([[qr[0], qr[1]] for qr in qr_patches]) 
                       for qr_patches in qr_patches_all])
    
    db_all = np.array([np.array([[db[0], db[1]] for db in db_patches]) 
                       for db_patches in db_patches_all])
    
    # translate to origin
    qr_mean = np.array([np.mean(qr_coords, axis=0) for qr_coords in qr_all])
    db_mean = np.array([np.mean(db_coords, axis=0) for db_coords in db_all])
    for i, (qr, db) in enumerate(zip(qr_mean, db_mean)):
        qr_all[i] -= qr
        db_all[i] -= db
    
    # set |p| <= 1
    max_l_qr = [max(qr, key=lambda p: np.linalg.norm(p)) for qr in qr_all]
    max_l_db = [max(db, key=lambda p: np.linalg.norm(p)) for db in db_all]
    for i, (qr, db) in enumerate(zip(max_l_qr, max_l_db)):
        a, b = np.linalg.norm(qr), np.linalg.norm(db)
        qr_all[i] /= a if a > np.finfo(float).eps else 1
        db_all[i] /= b if b > np.finfo(float).eps else 1
    
    return qr_all, db_all


def estimate_homographies(qr_coords_all,
                          db_coords_all,
                          est_cfg):

    models = [
        cv2.findHomography(qr_coords,
                           db_coords,
                           method=est_cfg["method"],
                           ransacReprojThreshold=est_cfg["max_reproj_err"],
                           maxIters=est_cfg["max_iters"])
        for qr_coords, db_coords in zip(qr_coords_all, db_coords_all)
    ]
    
    homographies = [H for H, _ in models]
    inliers = [I for _, I in models]
    
    return homographies, inliers
