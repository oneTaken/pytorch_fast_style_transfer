# comparision experiments
import torch
from utils import *
from features import Vgg16, Resnet101
from imageio import imread


def exp1():
    # compare the loss
    criterion = torch.nn.MSELoss()
    img_name_1 = './images/amber.jpg'
    img_name_2 = './images/amber_stylized_2.jpg'
    img_name_3 = './images/photo.jpg'
    img_name_4 = './images/mpsaic.jpg'

    img1 = imread(img_name_1)
    img2 = imread(img_name_2)
    img3 = imread(img_name_3)
    img4 = imread(img_name_4)

    features_vgg = Vgg16()
    features_resnet = Resnet101()
