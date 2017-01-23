# FusionSeg: Learning to combine motion and appearance for fully automatic segmention of generic objects in videos 

The following repository contains pretrained models for FusionSeg video object segementation method.

Please visit our [project page](http://vision.cs.utexas.edu/projects/fusionseg/) for the paper and visual results.

If you use this in your research, please cite the following papers:

```sh
@article{pixelobjectness,
  Author = {Jain, Suyog and Xiong, Bo and Grauman, Kristen},
  Journal = {arXiv preprint arXiv:1701.05349},
  Title = {Pixel Objectness},
  Year = {2017}
}
```

```sh
@article{fusionseg,
Author = {Jain, Suyog and Xiong, Bo and Grauman, Kristen},
Journal = {arXiv preprint arXiv:1701.05384},
Title = {FusionSeg: Learning to combine motion and appearance for fully automatic segmention of generic objects in videos},
Year = {2017}
}
```

These models are free for research and academic purposes. However it's patent pending, so please contact us for any commercial use.

## Using the pretrained models:

This model is trained using Deeplab-v2 caffe library. Please cite [1] and [2] if you use the code.

- Setup: 
 - Download and install Deeplab-v2 from [here](https://bitbucket.org/aquariusjay/deeplab-public-ver2)
 - Download pretrained models using appearance/download_model.sh and motion/download_model.sh
 - Or Direct downlowd from here:
   - [Appearance stream] (http://vision.cs.utexas.edu/projects/fusionseg/models/appearance_stream.caffemodel)
    - [Motion stream] (http://vision.cs.utexas.edu/projects/fusionseg/models/motion_stream.caffemodel)

- Set the caffe binary path in demo.py correctly.

- Refer to demo.py for step-by-step instruction on how to run the code.
  - Command: python demo.py \<model_type\> \<image_extension\> [See below for examples]  
  - Running demo.py will produce three files:
    - \<model_type\>_image_list.txt : contains list of of input images
    - \<model_type\>_output_list.txt: contains names to be used to store the output of video segementation
    - \<model_type\>_test.protoxt: prototxt file required for loading the pretrained model.

- Please resize your images so that the maximum side is < 513, otherwise update the crop_size value in appearance/appearance_stream_template.prototxt and motion/motion_stream_template.prototxt. Bigger crop sizes require larger gpu memory. If GPU runs out of memory, try with smaller image dimensions. With maximum side <=321 it should run on a basic modern GPU.

- Running the appearance stream model:
  - Appearance stream model is stored under the "appearance" folder.
  - Store the RGB images to be segmented under a folder named "images"
  - Command: python demo.py appearance jpg

- Running the motion stream model:
  - Motion stream model is stored under the "motion" folder.
  - Store the optical flow encoded as RGB images under a folder named "motion_images".
  - Please refer to compute_optical_flow.m to see an example of how to compute and encode optical flow as an RGB image.
  - Command: python demo.py motion png

- Running the joint fusion model (Coming soon)


## Visualizing the results:

After execution demo.py will store video segmentation results as matlab files.

Please refer to show_results.m to see how to visualize and extract foreground masks. You need to set model_type in the matlab script to visualize results from either the appearance stream or motion stream. 

## Please cite these too if you use the code:

[1] Caffe:

```sh
@article{jia2014caffe,
Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
Journal = {arXiv preprint arXiv:1408.5093},
Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
Year = {2014}
}
```

[2] Deeplab-v2:

```sh
@article{CP2016Deeplab,
title={DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs},
author={Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille},
journal={arXiv:1606.00915},
year={2016}
}
```
