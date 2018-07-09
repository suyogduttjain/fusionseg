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

visualize = True

model_type = sys.argv[1]
base_dir = sys.argv[2]

current_dir = os.getcwd()
base_model_dir   = os.path.join(current_dir,'models') + '/'
model_dir   = os.path.join(base_model_dir,model_type) + '/'

results_dir = os.path.join(base_dir,'results',model_type) + '/'
viz_dir     = os.path.join(base_dir,'visualization',model_type) + '/'

image_dir = os.path.join(base_dir,'images')
if not os.path.isdir(image_dir):
    sys.exit("Image folder not found")

#Verify that appearance and motion results exist
results_files = {}
for folder in ['appearance', 'motion']:
    results_folder = os.path.join(base_dir,'results',folder)
    if not os.path.isdir(results_folder):
        exit_msg = folder+' model results missing'
        sys.exit(exit_msg)
        
    results_files[folder] = fnmatch.filter(os.listdir(results_folder),'*raw*')
    results_files[folder].sort()

    if len(results_files[folder])==0:
        exit_msg = folder+' model results missing'
        sys.exit(exit_msg)

if len(results_files['appearance'])!=len(results_files['motion']):
    exit_msg = 'Mismatch between appearance and motion models'
    sys.exit(exit_msg)

#TRAINED ON DAVIS TRAIN SUBSET
#model_type = 'joint_davis_train'

#TRAINED ON DAVIS VAL SUBSET
#model_type = 'joint_davis_val'

if model_type == 'joint_davis_train' or model_type == 'joint_davis_val':
    caffe_model = os.path.join(model_dir,'joint_stream.caffemodel')
    output_layer = 'conv2'
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

def segment_image(seg_data):
    img_h = seg_data.shape[2]
    img_w = seg_data.shape[3]
    
    app_data = seg_data[:,0:2,:,:]
    mot_data = seg_data[:,2:4,:,:]
    
    nch = app_data.shape[1]
    s_tuple = (1, nch, img_h, img_w)
    net.blobs['app_data'].reshape(1,s_tuple[1],s_tuple[2],s_tuple[3])
    net.blobs['app_data'].data[...] = app_data
    
    net.blobs['mot_data'].reshape(1,s_tuple[1],s_tuple[2],s_tuple[3])
    net.blobs['mot_data'].data[...] = mot_data
    out = net.forward()
    
    probs = net.blobs[output_layer].data
    probs = np.squeeze(probs)
    probs = np.exp(probs)
    probs_sum = np.sum(probs,axis=0)
    
    fg_prob = np.squeeze(probs[1,:,:])
    fg_prob = np.divide(fg_prob, probs_sum)
    
    fg_prob = fg_prob[:img_h,:img_w]
    
    labels = np.argmax(probs,axis=0)
    labels = labels[:img_h,:img_w]
    return [labels,fg_prob]

def run_fusionseg():
    for extension in ['jpg', 'jpeg', 'bmp', 'png']:
        image_list = fnmatch.filter(os.listdir(image_dir),'*.'+extension)
        if len(image_list)>0:
            break
        
    image_list.sort()
    appearance_dir = os.path.join(base_dir,'results','appearance')
    motion_dir = os.path.join(base_dir,'results','motion')
    
    for image_name in image_list:
        im_path = os.path.join(image_dir,image_name)
        im_prefix = image_name.split('.')[0]
        print(im_path)
        
        app_probs_file  = os.path.join(appearance_dir,im_prefix+'_probs_raw.npy')
        mot_probs_file  = os.path.join(motion_dir,im_prefix+'_probs_raw.npy')
            
        app_prob = np.load(app_probs_file)
        mot_prob = np.load(mot_probs_file)
        img_h = app_prob.shape[1]
        img_w = app_prob.shape[2]
        
        seg_data  = np.zeros((1,4, img_h, img_w))
        seg_data[0,0:2,:,:] = app_prob
        seg_data[0,2:4,:,:] = mot_prob
        
        labels,fg_prob = segment_image(seg_data)
        
        labels_file = os.path.join(results_dir,im_prefix+'_mask.npy')
        probs_file  = os.path.join(results_dir,im_prefix+'_probs.npy')
        np.save(labels_file,labels)
        np.save(probs_file,fg_prob)
        
        if visualize==True:
            viz_img = visualize_image(im_path,labels,fg_prob,cmap_pascal,None)
            viz_file = os.path.join(viz_dir,im_prefix+'_viz.png')
            imsave(viz_file,viz_img);
    
def main():
    run_fusionseg()

if __name__ == '__main__':
    main()
