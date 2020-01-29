# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:33:42 2020

@author: Jinsung
"""

import numpy as np
import time
import os
import csv
import copy
import wave
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from IPython.core.display import HTML

def relu(x):
    return np.maximum(x,0)

def relu_derv(y):
    return np.sign(y)

def sigmoid(x):
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))

def sigmoid_derv(x, y):
    return y*(1-y)

def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x*z + np.log(1+np.exp(-np.abs(x)))

def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)

def tanh(x):
    return 2*sigmoid(2*x) - 1

def tanh_derv(y):
    return (1.0 + y) * (1.0 - y)

