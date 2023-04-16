from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from reidentification.identify import fisher_single, do_matching
from reidentification.encoding_utils import calculate_dists

def rescale_img(img, scale):
    return img.resize([int(s*scale) for s in img.size], Image.Resampling.LANCZOS)

def resize_to_img(img, dest_img):
    ratio = dest_img.size[0]/img.size[0]
    return (rescale_img(img, ratio), ratio)

"""
    Function for plotting the hotspot
    Input args:
        plt - plot
        ell - ellipse defining the hotspot, vector of 5 elements [x, y, major semi-axis length, minor semi-axis length, rotation angle]
        colors -colormap for plotting
        shift - shift to correctly display right image hotspots
        scale - scale in case if the image is resized
        n_rad - number of layers in ellipse
        max_opacity - maximum opacity
        n_pts - number of points for ellipse plotting
"""
def ell2plotMatch(plt, ell, colors, shift=[0, 0], scale=1, n_rad=15, max_opacity=0.1, n_pts=10):
    for i, rad in enumerate(np.linspace(1, 0, n_rad)):
#         generate angles from 0 to 2pi
        a = np.linspace(0, 2*np.pi, n_pts);
#     generate x and y to plot an ellipse without rotation
        x = ell[2] * rad * np.sin(a).reshape(1,-1)
        y = ell[3] * rad * np.cos(a).reshape(1,-1)
        pts =np.concatenate([x,y], axis=0)
#         calculate rotation matrix
        cos = np.cos(ell[4])
        sin = np.sin(ell[4])
        rot_mat = np.array([[cos, -sin],
                            [sin,  cos]])
#         rotate the ellipse
        pts = rot_mat @ pts
#     shift and scale of the ellipse
        pts[0, :] = shift[0] + (pts[0, :] + ell[0]) * scale
        pts[1, :] = shift[1] + (pts[1, :] + ell[1]) * scale
#         plot the ellipse
        c = colors(i)
        plt.plot(pts[0, :], pts[1, :], color=[c[0], c[1], c[2], min(1, max_opacity*(1-rad))])
#         return the x,y of ellipse center
    return [shift[0] + ell[0] * scale, shift[1] + ell[1] * scale]

def find_matches(identification_result, cfg):
#     for (i,(matches, query_label)) in enumerate(identification_result):
#         for db_match in query_match:
#             query_images = db_match["db_label"]["labels"]
    matches, query_labels = identification_result
    query_images = query_labels["labels"]
    query_fishers = [fisher_single(query_image['features'],cfg)  for query_image in query_images]
    db_encodings = []
    for (j,match) in enumerate(matches):
        db_images = match["db_label"]["labels"]
        db_fishers = [fisher_single(db_image['features'],cfg)  for db_image in db_images]

        (dists, inds) = calculate_dists(query_fishers, db_fishers)
        

        ind1, ind2 = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
        

        query_patch_features = query_images[ind1]['features']
        db_patch_features = db_images[ind2]['features']
        
        query_ellipses = query_images[ind1]['ellipses']
        db_ellipses = db_images[ind2]['ellipses']

        (filt, sorted_inds, similarity) = do_matching(query_patch_features, db_patch_features)

        matches[j]["patches"] = [
            [query_ellipses[k].tolist() for k in filt],
            [db_ellipses[k].tolist() for k in sorted_inds],
            similarity.tolist()
        ]
    return (matches, query_labels)
            



def visualise_match(input, cfg, path_to_load="file", uncropped=True, gap=20, n_rad=50, n_pts=10, figsize=(24, 24), line_color=(0.4,0.87,0.09)):
    matches, query_labels = find_matches(input, cfg=cfg)

    query_label = query_labels["labels"][0]
    if query_label.get("resize_ratio", 0) != 0:
        query_ratio = query_label["resize_ratio"]
    else:
        query_ratio = 1
    if uncropped and query_label.get("bb", 0) != 0:
        query_shift = query_label["bb"][:2]
    else:
        query_shift = [0, 0]
    query_shift = [x*query_ratio for x in query_shift]
    query_img = rescale_img(Image.open(query_label[path_to_load]), query_ratio)
    
#     loop for each match frrom the database
    for k, match in enumerate(matches):
        fig = plt.figure(figsize=figsize)
        db_label = match["db_label"]
        query_patches, db_patches, similarity = match["patches"]
        
        db_data = db_label["labels"][0]
        db_img = Image.open(db_data[path_to_load])
        db_img, ratio = resize_to_img(db_img, query_img)
        db_ratio = ratio/db_data["resize_ratio"]
        
        shift = [x*ratio for x in (db_data["bb"][:2] if uncropped else [0, 0])]
        shift = (shift[0], shift[1] + query_img.size[1] + gap)
        
#         create new image for visualisation. It will contain both query and db image
        full_img = Image.new('RGB', (query_img.size[0], query_img.size[1]+db_img.size[1]+gap), color="white")
        full_img.paste(query_img, (0, 0))
        full_img.paste(db_img, (0, query_img.size[1] + gap))
        
        colors = plt.get_cmap("viridis", n_rad)
        plt.axis("off")
        plt.imshow(full_img)
        
        plt.title(f'Distance: {match["distance"]}')
        plt.title(f'Query: class {query_label["class_id"]}', loc='left')
        plt.title(f'Top-{k+1}: class {db_label["class_id"]}', loc='right')
        
#         for each pair of matching features plot a hotspot
        for LAF_q, LAF_db, sim in zip(query_patches, db_patches, similarity):
#             Intensity depends on feature similarity
            max_opacity = 0.5*sim
            p1 = ell2plotMatch(plt, LAF_q, colors, shift=query_shift,n_rad=n_rad, max_opacity=max_opacity, n_pts=n_pts)
            p2 = ell2plotMatch(plt, LAF_db, colors, shift=shift, scale=db_ratio, n_rad=n_rad, max_opacity=max_opacity, n_pts=n_pts)
#             draw line between patch centers
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=(*line_color, min(max_opacity* 3, 1)))   
        plt.show()
    return [input]

