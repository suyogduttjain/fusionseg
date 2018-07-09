import sys
import os
import fnmatch
import numpy as np
from scipy.misc import imsave
from visualize import visualize_image
from scipy.io import loadmat

caffe_root = '/home/sjain/projects/research/library/deeplab-public-ver2/distribute/'
sys.path.insert(0, caffe_root + 'python')
import caffe

target_h = 289 
target_w = 513
use_padding = True
visualize = True

model_type = sys.argv[1]
ext = sys.argv[2]
base_dir = sys.argv[3]

current_dir = os.getcwd()
base_model_dir   = os.path.join(current_dir,'models') + '/'
model_dir   = os.path.join(base_model_dir,model_type) + '/'

results_dir = os.path.join(base_dir,'results',model_type) + '/'
viz_dir     = os.path.join(base_dir,'visualization',model_type) + '/'

if model_type == 'appearance':
    image_dir = os.path.join(base_dir,'images')
    if not os.path.isdir(image_dir):
        sys.exit("Image folder not found")
    caffe_model = os.path.join(model_dir,'appearance_stream.caffemodel')
    output_layer = 'fc1_interp'
elif model_type == 'motion':
    image_dir = os.path.join(base_dir,'motion_images')
    if not os.path.isdir(image_dir):
        sys.exit("Need optical flow images stored in a folder motion_images")
    caffe_model = os.path.join(model_dir,'motion_stream.caffemodel')
    output_layer = 'fc1_interp_motion'
else:
    sys.exit("Model type not supported")


proto_file  = os.path.join(model_dir,'deploy.prototxt')

cmd = 'mkdir -p ' + results_dir
os.system(cmd)

if visualize==True:
    cmd = 'mkdir -p ' + viz_dir
    os.system(cmd)
    colormap_file = os.path.join(base_model_dir,'pascal_seg_colormap.mat')
    cmap_pascal = loadmat(colormap_file)
    cmap_pascal = cmap_pascal['colormap']


USE_GPU = True
if USE_GPU == True:
    caffe.set_device(0)
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

net = caffe.Net(proto_file,caffe_model,caffe.TEST)

def segment_image(im_path):
    img = caffe.io.load_image(im_path)
    
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    orig_h = img_h
    orig_w = img_w
    if use_padding == True:
        if img_h!=target_h or img_w!=target_w:
            z_img = np.zeros((max(target_h,img_h),max(target_w,img_w),3),dtype=np.float32)
            z_img[:orig_h,:orig_w,:] = img
            img = z_img

            img_h = img.shape[0]
            img_w = img.shape[1]
    
    s_tuple = (1, 3, img_h, img_w)

    transformer = caffe.io.Transformer({'data': s_tuple})
    transformer.set_mean('data', np.load(os.path.join(base_model_dir,'ilsvrc_2012_mean.npy')).mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    net.blobs['data'].reshape(1,s_tuple[1],s_tuple[2],s_tuple[3])
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    
    probs = net.blobs[output_layer].data
    probs = np.squeeze(probs)
    
    img_h = orig_h
    img_w = orig_w
    
    probs = probs[:,:img_h,:img_w]
    raw_probs = probs
    
    probs = np.exp(probs)

    probs_sum = np.sum(probs,axis=0)
    
    fg_prob = np.squeeze(probs[1,:,:])
    fg_prob = np.divide(fg_prob, probs_sum)
    
    fg_prob = fg_prob[:img_h,:img_w]
    
    labels = np.argmax(probs,axis=0)
    labels = labels[:img_h,:img_w]
    
    return [labels,fg_prob,raw_probs,img_h,img_w]

def run_fusionseg():
    image_list = fnmatch.filter(os.listdir(image_dir),'*.'+ext)
    image_list.sort()

    for img_name in image_list:
        im_prefix = img_name.split('.')[0]
        im_path = os.path.join(image_dir,img_name)
        
        labels,fg_prob,raw_probs,img_h,img_w = segment_image(im_path)
        labels_file = os.path.join(results_dir,im_prefix+'_mask.npy')
        probs_file  = os.path.join(results_dir,im_prefix+'_probs.npy')
        raw_probs_file  = os.path.join(results_dir,im_prefix+'_probs_raw.npy')
        
        np.save(labels_file,labels)
        np.save(probs_file,fg_prob)
        np.save(raw_probs_file,raw_probs)

        if visualize==True:
            viz_img = visualize_image(im_path,labels,fg_prob,cmap_pascal,None)
            viz_file = os.path.join(viz_dir,im_prefix+'_viz.png')
            imsave(viz_file,viz_img);
        

def main():
    run_fusionseg()

if __name__ == '__main__':
    main()
