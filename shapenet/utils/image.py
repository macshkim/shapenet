import numpy as np
from skimage.transform import AffineTransform
import cv2

def load_(path):
    # read img
    img = cv2.imread(path)
    #

def crop(data):
    pass

def resize(data, scale):
    # scale = np.asarray(target) / np.asarray(data.shape[:-1])
    # return AffineTransform(scale=scale)
    pass


if __name__ == '__main__':
    load_img('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2165593916_1.jpg')