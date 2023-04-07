# Automated Assembly Quality Inspection by Deep Learning with 2D and 3D Synthetic CAD Data

# 2D-domain-adpatation
Implement unsupervise domain adpatation and transfer learning models for object detection on industrial data to achieve assembly quality insecption. The models will be trained on synthetic images generated from Industrail CAD models, and make prediction on real images captured from the production line. 

## 1. DA-detection  
**Unsupervised Domain Adaptation model**: Saito K, Ushiku Y, Harada T, et al. Strong-weak distribution alignment for adaptive object detection[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 6956-6965.  

The step is to applied unsupervised domain adative object detection on synthetic and real images.   
The introduction of running the code is in 2D-DA-Detection  

## 2. Faster RCNN  
**Unsupervised learning**: Using faster rcnn to train on synthetic images generated from Industrial CAD models, and make prediction on images captured from the real production line.  
**Supervised learning**: Using faster rcnn to detect images from the real production line.    
**Transfer leaning**: Using faster rcnn to train on syntehtic images, then fine-tune on 5/all real images.   

The step is to set the baseline of 2D domain adpatation and apply the transfer learning method.   
The introduction of running the code is in Faster RCNN.  

## 3. OpenCV
**Feature based image alignment model**.   

The step is to try image matching without deep learning and see the results.   

# 3D-domain-adpatation
Implement 3D unsupervise domain adpatation  and transfer learning models for classification on industrial data to achieve assembly quality insecption. The models will be trained on synthetic  point cloud generated from Industrail CAD models, and make prediction on real point cloud captured from the production line.   

## 1. PointNet 
Using pointnet++ to train on synthetic point cloud generated from Industrail CAD models, and make prediction on point cloud captured from the real production line.
In order to compare 3D and 2D classification.   

The introduction of running the code is in pointnet++.  

## Credits & How to Cite
If you use this dataset in your research, please cite the orginal reference of the DA-detection, Faster RCNN, and PointNet. 

And possible cite our paper:

>   title={Automatic assembly quality inspection based on an unsupervised point cloud domain adaptation model},
>   author={Zhu, Xiaomeng and Manamasa, Himaja and S{\'a}nchez, Juan Luis Jim{\'e}nez and Maki, Atsuto and Hanson, Lars},
>   journal={Procedia CIRP},
>   volume={104},
>   pages={1801--1806},
>   year={2021},
>   publisher={Elsevier}
> }

>   title={Unsupervised domain adaptive object detection for assembly quality inspection},
>   author={Zhu, Xiaomeng and Maki, Atsuto and Hanson, Lars},
>   journal={Procedia CIRP},
>   volume={112},
>   pages={477--482},
>   year={2022},
>   publisher={Elsevier}
> }

