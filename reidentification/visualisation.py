from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def rescale_img(img, scale):
    return img.resize([int(s*scale) for s in img.size], Image.Resampling.LANCZOS)

def resize_to_img(img, dest_img):
    ratio = dest_img.size[0]/img.size[0]
    return (rescale_img(img, ratio), ratio)

def ell2plotMatch(plt, ell, colors, shift=[0, 0], scale=1, n_rad=15, max_opacity=0.1, n_pts=10):
    for i, rad in enumerate(np.linspace(1, 0, n_rad)):
        a = np.linspace(0, 2*np.pi, n_pts);
        x = ell[2] * rad * np.sin(a).reshape(1,-1)
        y = ell[3] * rad * np.cos(a).reshape(1,-1)
        pts =np.concatenate([x,y], axis=0)
        cos = np.cos(ell[4])
        sin = np.sin(ell[4])
        rot_mat = np.array([[cos, -sin],
                            [sin,  cos]])
        pts = rot_mat @ pts
        pts[0, :] = shift[0] + (pts[0, :] + ell[0]) * scale
        pts[1, :] = shift[1] + (pts[1, :] + ell[1]) * scale
        c = colors(i)
        plt.plot(pts[0, :], pts[1, :], color=[c[0], c[1], c[2], min(1, max_opacity*(1-rad))])
    return [shift[0] + ell[0] * scale, shift[1] + ell[1] * scale]

def visualise_match(input, path_to_load="file", uncropped=True, gap=20, n_rad=50, n_pts=10, figsize=(24, 24), line_color=(0.4,0.87,0.09)):
    matches, query_label = input
    query_ratio = query_label["resize_ratio"]
    query_shift = query_label["bb"][:2] if uncropped else [0, 0]
    query_shift = [x*query_ratio for x in query_shift]
    query_img = rescale_img(Image.open(query_label[path_to_load]), query_ratio)
    
    for k, match in enumerate(matches):
        fig = plt.figure(figsize=figsize)
        db_label = match["db_label"]
        query_patches, db_patches, similarity = match["matches"]
        
        db_img = Image.open(db_label[path_to_load])
        db_img, ratio = resize_to_img(db_img, query_img)
        db_ratio = ratio/db_label["resize_ratio"]
        
        shift = [x*ratio for x in (db_label["bb"][:2] if uncropped else [0, 0])]
        shift = (shift[0], shift[1] + query_img.size[1] + gap)
        full_img = Image.new('RGB', (query_img.size[0], query_img.size[1]+db_img.size[1]+gap), color="white")
        full_img.paste(query_img, (0, 0))
        full_img.paste(db_img, (0, query_img.size[1] + gap))
        colors = plt.get_cmap("viridis", n_rad)
        plt.axis("off")
        plt.imshow(full_img)
        
        plt.title(f'Distance: {match["distance"]}')
        plt.title(f'Query: class {query_label["class_id"]}', loc='left')
        plt.title(f'Top-{k+1}: class {query_label["class_id"]}', loc='right')
        for LAF_q, LAF_db, sim in zip(query_patches, db_patches, similarity):
            max_opacity = 0.5*sim
            p1 = ell2plotMatch(plt, LAF_q, colors, shift=query_shift,n_rad=n_rad, max_opacity=max_opacity, n_pts=n_pts)
            p2 = ell2plotMatch(plt, LAF_db, colors, shift=shift, scale=db_ratio, n_rad=n_rad, max_opacity=max_opacity, n_pts=n_pts)
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=(*line_color, max_opacity* 3))   
        plt.show()
    return input