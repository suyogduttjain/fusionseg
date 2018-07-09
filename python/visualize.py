import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np
from scipy.misc import imread

def visualize_image(im_path,labels,fg_prob,cmap_pascal,gt=None):
    img = imread(im_path)
    
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=fg_prob.min(), vmax=fg_prob.max())
    rgb_fg_prob = cmap(norm(fg_prob))
    rgb_fg_prob = rgb_fg_prob[:,:,:3]
    rgb_fg_prob = (rgb_fg_prob * 255).round().astype(np.uint8)
   
    labels_rgb_img = cmap_pascal[labels]; 
    labels_rgb_img_uint = (labels_rgb_img * 255).round().astype(np.uint8)
    
    if gt is not None:
        gt_rgb_img = cmap_pascal[gt]; 
        gt_rgb_img_uint = (gt_rgb_img * 255).round().astype(np.uint8)
        viz_img = np.concatenate((img,rgb_fg_prob, labels_rgb_img_uint, gt_rgb_img_uint),axis=1);
    else:
        viz_img = np.concatenate((img,rgb_fg_prob, labels_rgb_img_uint),axis=1);
    
    return viz_img

