# Implementing CNN Architectures
In this repository you will find the implementation of ResNet and Inception-ResNet from scratch and also a CNN model for classifying natural disaster images.
### [ResNet Implementation with Keras](https://github.com/shoaibsattar823/CS893/blob/master/resnet.ipynb) <br>
In this notebook, resnet is implemented using keras and trained on cifar100 dataset.

### [Inception-ResNet Implementation with Keras](https://github.com/shoaibsattar823/CS893/blob/master/inception_resnet.ipynb)
In this jupyter notebook, inception-resnet is implemented using keras and is trained on cifar100 dataset.

# Classification of Images using Keras
### [Natural Disaster Classification with Keras](https://github.com/shoaibsattar823/CS893/blob/master/natural_disaster_classifier.py)
Classification of natural disaster images into four classes which are:
1. Cyclone
2. Wildfire
3. Flood
4. Earthquake

The classification model is created as a convolutional neural network (CNN) using keras layers. You can find how to:
* load data and split into train and test sets
* load data in batches using data generators when the dataset is large
* create a CNN model from scratch and train on your dataset
* evaluate the model's performance and view confusion matrix
* plot the learning curve
