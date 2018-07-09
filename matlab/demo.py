import os
import sys
import fnmatch

#Caffe binary location from deeplab-v1 installation
caffe_binary = '/vision/vision_users/suyog/deep_video_segmentation/code_release/deeplab-public-ver2/distribute/bin/caffe.bin' 

model_type = sys.argv[1]
ext = sys.argv[2]

base_dir = os.getcwd()

if model_type == 'appearance':
	image_dir = os.path.join(base_dir,'images')
	if not os.path.isdir(image_dir):
		sys.exit("Image folder not found")
elif model_type == 'motion':
	image_dir = os.path.join(base_dir,'motion_images')
	if not os.path.isdir(image_dir):
		sys.exit("Need optical flow images stored in a folder motion_images")
else:
	sys.exit("Model type not supported")


model_dir = os.path.join(base_dir,model_type) + '/'

image_list = fnmatch.filter(os.listdir(image_dir),'*.'+ext)
image_list.sort()

input_list_file   = base_dir + '/' + model_type + '_image_list.txt'
output_list_file  = base_dir + '/' + model_type + '_output_list.txt'

input_list  = open(input_list_file,'w')
output_list = open(output_list_file,'w')

for img in image_list:
	input_list.write('/'+img+'\n')
	prefix = img.split('.')[0]
	output_list.write(prefix+'\n')

input_list.close()
output_list.close()

template_file = open(model_dir + model_type +'_stream_template.prototxt').readlines()

test_file_path = model_dir + model_type + '_stream.prototxt'
test_file = open(test_file_path,'w')

tokens = {}
tokens['${IMAGE_DIR}'] = 'root_folder: \"' + image_dir + '\"'
tokens['${OUTPUT_DIR}'] = 'prefix: \"' + image_dir + '/\"'

tokens['${IMAGE_LIST}']        = 'source: \"' + input_list_file + '\"'
tokens['${IMAGE_OUTPUT_LIST}'] = 'source: \"' + output_list_file + '\"'

for line in template_file:
	line = line.rstrip()

	for key in tokens:
		if line.find(key)!=-1:
			line = '\t'+tokens[key]
			break

	test_file.write(line+'\n')

test_file.close()

weight_file_path = model_dir + model_type + '_stream.caffemodel'
cmd = caffe_binary + ' test --model=' + test_file_path + ' --weights=' + weight_file_path + ' --gpu=0 --iterations='+str(len(image_list))
print cmd
os.system(cmd)
