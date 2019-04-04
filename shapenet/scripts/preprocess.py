from .make_pca import load_landmarks
import numpy as np
import cv2 
from skimage.color import rgb2gray
from skimage.transform import AffineTransform, warp, resize
import os 
from matplotlib import pyplot as plt
# import matplotlib.patches as patches
IMAGE_SIZE = 224
CROP_OFFSET = 0.05

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
    offset = int((max_x - min_x) * CROP_OFFSET)
    min_y, max_y = min_y - offset, max_y + offset
    min_x, max_x = min_x - offset, max_x + offset
    min_y = min_y if min_y > 0 else 0
    min_x = min_x if min_x > 0 else 0
    max_x = max_x if max_x < img.shape[1] else img.shape[1]
    max_y = max_y if max_y < img.shape[0] else img.shape[0]
    # print ('crop bound ', min_y, min_x, (max_x - min_x), (max_y - min_y))
    img = img[min_y:max_y, min_x:max_x]
    # crop lmks
    lmks = lmks - np.array([min_x, min_y])
    return img, lmks

def grayscale(img):
    return rgb2gray(img)# .reshape(img.shape[:-1], 1)

def view_img(img, lmks, ref_lmks = None):
    plt.imshow(img, cmap="gray")
    # top, left, w, h = bound
    # p = patches.Rectangle((top,left),w, h,linewidth=1,edgecolor='r',facecolor='none')
    plt.scatter(lmks[:, 0], lmks[:, 1], c="C0", s=15)
    # plt.add_patch(p)
    if ref_lmks is not None:
        plt.scatter(ref_lmks[:, 0], ref_lmks[:, 1], c="C1", s=15)
    plt.show()

def resize_lmks(img, lmks, img_size, name):
    target_shape = (img_size, img_size)
    # print('target_shape', target_shape, 'image shape ', img.shape[:-1], ' file name', name)
    scale = np.asarray(target_shape) / np.asarray(img.shape[:-1])
    # print('scale = ', scale)
    trafo = AffineTransform(scale=scale)
    # img = warp(np.ascontiguousarray(img), trafo.inverse, output_shape=target_shape)
    lmks = trafo(np.ascontiguousarray(lmks[:, [1, 0]]))[:, [1, 0]]
    # lmks = warp(np.ascontiguousarray(lmks[:, [1, 0]]), trafo.inverse, output_shape=target_shape)[:, [1, 0]]
    return lmks

def read_data(lmk_xml):    
    base_dir = os.path.dirname(lmk_xml)
    points, img_sizes, imgs = load_landmarks(lmk_xml)    
    img_size = IMAGE_SIZE
    data = np.ndarray((len(imgs), img_size, img_size), dtype=np.float32)
    labels = np.ndarray((len(imgs), *points[0].shape), dtype=np.int32)
    for i in range(0, len(imgs)):        
        img_path = os.path.join(base_dir, imgs[i])
        _, _, bound= img_sizes[i]

        im = load_img(img_path)
        lmks = points[i]        
        # print(i, ' img shape ', im.shape[:-1])        
        im, lmks = crop(im, lmks)
        # print(i, ' img shape after crop ', im.shape[:-1])
        lmks = resize_lmks(im, lmks, img_size, imgs[i])        
        im   = resize(im, (img_size, img_size), anti_aliasing=True, mode='reflect')  
        im = grayscale(im)
        data[i] = im 
        labels[i] = lmks        
    return data, labels

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def preprocess(lmk_xml):
    # save
    save_f = os.path.join(os.path.dirname(lmk_xml), os.path.basename(lmk_xml).replace('.xml', '.npz'))
    if os.path.exists(save_f):
        print ('preprocessed file exist: ', save_f)
        return

    data, labels = read_data(lmk_xml)
    # shuffle
    data, labels = randomize(data, labels)    
    # visualize to test after randomize
    view_img(data[1], labels[1])
    np.savez(save_f, data=data, labels=labels)


if __name__ == '__main__':
    # preprocess train data
    preprocess('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
    # preprocess test data
    preprocess('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_test.xml')
    # t = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/245871800_1.jpg'
    # im = load_img(t)
    # print (im.shape)