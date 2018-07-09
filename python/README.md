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
Journal = {CVPR},
Title = {FusionSeg: Learning to combine motion and appearance for fully automatic segmention of generic objects in videos},
Year = {2017}
}
```

These models are freely available for research and academic purposes. However it's patent pending, so please contact us for any commercial use.

## Using the pretrained models:

This model is trained using Deeplab-v2 caffe library. Please cite [1] and [2] if you use the code.

- Setup: 
 - Download and install Deeplab-v2 from [here](https://bitbucket.org/aquariusjay/deeplab-public-ver2)
 - Download pretrained models using appearance/download_model.sh and motion/download_model.sh
 - Or Direct downlowd from here:
   - [Appearance stream] (http://vision.cs.utexas.edu/projects/fusionseg/models/appearance_stream.caffemodel)
    - [Motion stream] (http://vision.cs.utexas.edu/projects/fusionseg/models/motion_stream.caffemodel)

- Set the caffe binary path in fusionseg.py correctly.
- Set the caffe binary path in fusionseg_joint.py correctly.

The python implementation is self contained and only requires the deploy.prototxt and the model files for execution.

- Refer to run.py for instructions on how to run the complete fusionseg pipeline. Here is a step by step breakdown:

- Data preparation:
  - Create a directory for storing the video that needs to be segmented (e.g. ./sample/)
  - Store RGB frames inside a subdirectory named "images" (e.g. ./sample/images/)
  - Store Optical flow encode as RGB images inside a subdirectory called "motion_images" (e.g. ./sample/motion_images/)

- Running the appearance stream model:
  - Appearance stream model is stored under the "models/appearance" folder.
  - Command: python fusionseg.py appearance jpg ./sample/

- Running the motion stream model:
  - Motion stream model is stored under the "models/motion" folder.
  - Please refer to compute_optical_flow.m (in the matlab implementation) to see an example of how to compute and encode optical flow as an RGB image.
  - Command: python fusionseg.py motion png ./sample/

- Running the joint fusion model:
  - Fusionseg model (joint stream) trained on DAVIS training subset: "models/joint_davis_train" folder
  - Fusionseg model (joint stream) trained on DAVIS validation subset: "models/joint_davis_val" folder
  - Command: python fusionseg_joint.py joint_davis_train ./sample/
  - Command: python fusionseg_joint.py joint_davis_val ./sample/

- Results:
  - Segmentation results will be stored under ./sample/results/[model_type]
  - Segmentation visualization will be stored under ./sample/visualization/[model_type]

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
