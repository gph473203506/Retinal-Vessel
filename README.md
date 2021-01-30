# Improved Multiscale U-Net Retinal Vascular Segmentation based on Spatial positional attention

---

## Overview

### Datasets



DRIVE:https://drive.google.com/file/d/1mOjVt2_A1Q7LJ3_C8VEKRfoK2OwEeooy/view?usp=sharing
CHASE:https://drive.google.com/file/d/1RnPR3hpKIHnu0e3y9DBOXKPXuiqPN8hg/view?usp=sharing

### Pre-processing
Run sy_prepare_datasets_DRIVE.py or sy_prepare_datasets_CHASE_DB1.py
Run extract_patches.py or extract_patches.py
Run save_patches.py or save_patches.py

### Training
Run  models.py or  models.py 
Run sy_train.py or train.py

### Testing

Run evaluate.py or evaluate.py



## About Keras

Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

Keras是一个极简、高度模块化的神经网络库，用Python编写，能够运行在TensorFlow或Theano之上。它的开发重点是实现快速试验。能够以最小的延迟从想法到结果是做好研究的关键。

Use Keras if you need a deep learning library that:

allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
supports both convolutional networks and recurrent networks, as well as combinations of the two.

允许简单和快速的原型(通过完全的模块化、极简主义和可扩展性)。支持卷积网络和循环网络，以及两者的组合。

supports arbitrary connectivity schemes (including multi-input and multi-output training).
runs seamlessly on CPU and GPU.

支持任意连接方案(包括多输入多输出训练)。在CPU和GPU上无缝运行。Read the documentation [Keras.io](http://keras.io/)

Keras is compatible with: Python 2.7-3.5.


@article{Guo2020SAUNetSA,
  title={Improved Multiscale U-Net Retinal Vascular Segmentation based on Spatial positional attention},
  author={Penghui Gu1 and Yan Ding2 and Fengsheng Zhou2 and Zhiyong Xiao1*},
}
