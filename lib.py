import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import random
from typing import Union, Tuple
import torch.nn as nn
from torch.nn import Module
import torch.optim as opt
from torch.cuda.amp import autocast, GradScaler
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import onnx, onnxscript
import onnxruntime as ort
import torch.onnx
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt

from model import CNN
from dataloader import Covid_XRay
from tensorrt_infer import load_engine, allocate_buffers, do_inference