# AerialCrackDetection_Keras
AerialCrackDetection_Keras is a project about object detection from aerial imagery using pavament crack data. The project uses the open source software library Keras and Tensorflow, with a ZF or VGG16 or ZF or VGG16 or GoogleNet or ResNet50 or ResNet101 neuronal networks. AerialCrackDetection_Keras is based on Faster RCNN.
 - PS: The project is only the original version, the improved version is not open.

### First part : Collecting data

 - The first part is collecting and labeling aerial pictures. 
 - Most of the pictures are from School of Aerospace Engineering, Beijing Institute of Technology. 
 - You can use [LabelImg](https://github.com/tzutalin/labelImg) to analyze them and label them. 
 - You can find the AerialCrackDataset in my [Google Drive](https://drive.google.com/open?id=0B2gdFlquH6TORGo4azgySjlfODA).
 - If you find AerialCrackDataset useful in your research, please consider citing:
```
    @inproceedings{
        Author = {Bo Wang},
        Title = {AerialCrackDataset: Towards Object Detection with Dataset},
        Laboratory = {Key Laboratory of Optoelectronic Imaging Technology and System, 
                      Ministry of Education, School of Optoelectronics, 
                      Beijing Institute of Technology},
        Year = {2017}
    }
```

### Second part : Installing and Configuration

 - You need install Tensorflow and Keras. 
 - If you want to use the Pre-trained ImageNet models: VGG16 or ResNet50, you need download them from [here](https://github.com/fchollet/deep-learning-models/releases).
```
cd $FRCN_ROOT
mkdir model
cd model
# put the Pre-trained ImageNet models here
```

### Third part : Training with Keras

 - The third part is training the detection and classification model.
```
cd $FRCN_ROOT
./train.py [--path] [--network]
# --path is the dataset location you want to train
# --net in {ZF, VGG16, GoogleNet, ResNet50, ResNet101} is the network arch to use
```

### Fourth part : Testing with Keras

 - The Fourth part is testing the detection and classification model.
```
cd $FRCN_ROOT
./test.py [--path] [--network]
# --path is the dataset location you want to test
# --net in {ZF, VGG16, GoogleNet, ResNet50, ResNet101} is the network arch to use
```
