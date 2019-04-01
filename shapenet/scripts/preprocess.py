from make_pca import load_landmarks
import numpy as np
import cv2 
from skimage.color import rgb2gray
from skimage.transform import AffineTransform, warp, resize
import os 
from matplotlib import pyplot as plt
# import matplotlib.patches as patches

def load_img(img_path):
    """read data from a single image. crop and rotate if necessary"""
    # read image
    img = cv2.imread(img_path)
    # crop
    return img

def crop(img, lmks):
    
    min_y, max_y = lmks[:,1].min(), lmks[:,1].max()
    min_x, max_x = lmks[:,0].min(), lmks[:,0].max() 
    # crop img data
    offset = 2
    min_y, max_y = min_y - offset, max_y + offset
    min_x, max_x = min_x - offset, max_x + offset
    # print ('crop bound ', min_y, min_x, (max_x - min_x), (max_y - min_y))
    img = img[min_y:max_y, min_x:max_x]
    # crop lmks
    lmks = lmks - np.array([min_x, min_y])
    return img, lmks

def grayscale(img):
    return rgb2gray(img)# .reshape(img.shape[:-1], 1)

def view_img(img, lmks):
    plt.imshow(img, cmap="gray")
    # top, left, w, h = bound
    # p = patches.Rectangle((top,left),w, h,linewidth=1,edgecolor='r',facecolor='none')
    plt.scatter(lmks[:, 0], lmks[:, 1], c="C0", s=15)
    # plt.add_patch(p)
    plt.show()

def resize_lmks(img, lmks, img_size):
    target_shape = (img_size, img_size)
    scale = np.asarray(target_shape) / np.asarray(img.shape[:-1])
    # print('scale = ', scale)
    trafo = AffineTransform(scale=scale)
    # img = warp(np.ascontiguousarray(img), trafo.inverse, output_shape=target_shape)
    lmks = trafo(np.ascontiguousarray(lmks[:, [1, 0]]))[:, [1, 0]]
    # lmks = warp(np.ascontiguousarray(lmks[:, [1, 0]]), trafo.inverse, output_shape=target_shape)[:, [1, 0]]
    return lmks

def preprocess(lmk_xml):    
    base_dir = os.path.dirname(lmk_xml)
    points, img_sizes, imgs = load_landmarks(lmk_xml)
    img_size = 224
    data = np.array(())
    for i in range(0, len(imgs)):        
        img_path = os.path.join(base_dir, imgs[i])
        _, _, bound= img_sizes[i]
        im = load_img(img_path)
        lmks = points[i]
        
        im, lmks = crop(im, lmks)
        lmks = resize_lmks(im, lmks, img_size)        
        im   = resize(im, (img_size, img_size))     
        im = grayscale(im)
        view_img(im, lmks)
        break

if __name__ == '__main__':
    preprocess('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')