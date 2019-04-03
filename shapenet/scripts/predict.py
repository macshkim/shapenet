import dlib
from .train import load_model
from .preprocess import grayscale, IMAGE_SIZE
from skimage.transform import resize
import numpy as np
import torch

def detect_face(img):
    detector = dlib.get_frontal_face_detector()
    dets = detector(img, 1)    
    return dets

def crop(img, bound):
    return img

def predict(model, img, input_device):
    data = np.array([img])
    data = torch.from_numpy(data).to(input_device).to(torch.float)
    model.eval()
    with torch.no_grad():
        preds = model(data)
        return preds.cpu()[0]

def get_lmks(data_dir, img_path):
    # detect bounding rect
    img = dlib.load_rgb_image(img_path)    
    dets = detect_face(img)
    face_bound = dets[0]

    # pre-process, #crop
    img = crop(img, face_bound)
    img = grayscale(img)
    img = resize(img, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True, mode='reflect')   
    # predict
    model, _, input_device, output_device = load_model(data_dir)
    lmks = predict(model, img, input_device)
    print(lmks)

if __name__ == '__main__':
    data_dir = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset'
    get_lmks(data_dir, '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/helen/trainset/2165593916_1.jpg')