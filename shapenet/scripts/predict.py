import dlib
from .train import load_model, predict, load_pretrain_model
from ..dataset import DataSet
from .preprocess import grayscale, IMAGE_SIZE, CROP_OFFSET, view_img
from skimage.transform import resize
import numpy as np
import torch
import os
from torch.nn.modules.module import _addindent

def detect_face(img):
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)    
    return dets

def crop(img, bound):
    print ('boudn = ', bound)
    l, t, r, b = bound
    min_y, max_y = t, b
    min_x, max_x = l, r
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
    return img

def get_lmks(data_dir, img_path):
    # detect bounding rect
    img = dlib.load_rgb_image(img_path)    
    d = detect_face(img)[0]
    face_bound = (d.left(), d.top(), d.right(), d.bottom())
    # pre-process, #crop
    img = crop(img, face_bound)
    img = grayscale(img)
    img = resize(img, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True, mode='reflect')   
    # predict
    model, _, input_device, output_device = load_pretrain_model(data_dir)
    lmks = predict(model, img, input_device)
    print(len(lmks))
    view_img(img, lmks)

def test_on_train(data_dir):
    train_data = os.path.join(data_dir, 'labels_ibug_300W_train.npz')
    ds = DataSet(train_data)
    model, _, input_device, output_device = load_model(data_dir, 0.0001)
    lmks = predict(model, ds.data[491:492], input_device)
    img = ds.data[491]
    img = img.reshape(*img.shape[1:])
    # view_img(img, lmks[:, [1, 0]])
    view_img(img, lmks, ds.labels[491])

def examine(data_dir):
    print('Pre-trained')
    model, _, _, _ = load_pretrain_model(data_dir)
    print(torch_summarize(model, show_weights=False))
    print('=======================')
    print('Self-trained')
    model, _, _, _ = load_model(data_dir, 0.0001)
    print(torch_summarize(model, show_weights=False))

def torch_summarize(model, show_weights=True, show_parameters=True):
    """
    https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
    Summarizes torch model by showing trainable parameters and weights.
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'   

    tmpstr = tmpstr + ')'
    return tmpstr

if __name__ == '__main__':
    data_dir = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset'
    target = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2344967158_1.jpg'
    # get_lmks(data_dir, target)
    test_on_train(data_dir)