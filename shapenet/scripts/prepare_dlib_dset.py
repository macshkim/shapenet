import glob
import os
from shapedata.io import pts_importer
from copy import deepcopy
from shapedata import SingleShapeDataProcessing
import numpy as np
from sklearn.decomposition import PCA

def lmk_pca(landmarks, scale = True, center = True):
    """
    perform PCA on samples' landmarks
    Parameters
    ----------
    scale : bool
        whether or not to scale the principal components with the
        corresponding eigen value
    center : bool
        whether or not to substract mean before pca
    Returns
    -------
    np.array
        eigen_shapes
    """
    if center:
        print ('  centering...')
        mean = np.mean(landmarks.reshape(-1, landmarks.shape[-1]), axis=0)
        landmarks = landmarks - mean
    landmarks_transposed = landmarks.transpose((0, 2, 1))

    reshaped = landmarks_transposed.reshape(landmarks.shape[0], -1)

    print ('  pca fit...')
    pca = PCA()
    pca.fit(reshaped)

    if scale:
        print ('  scaling...')
        components = pca.components_ * pca.singular_values_.reshape(-1, 1)
    else:
        components = pca.components_


    return np.array([pca.mean_] + list(components)).reshape(
        components.shape[0] + 1,
        *landmarks_transposed.shape[1:]).transpose(0, 2, 1)

def list_files(directory):
    files = []
    extensions = [".pts"]    
    for ext in extensions:
        ext = ext.strip(".")
        files += glob.glob(directory + "/*." + ext)        
    files.sort()
    # pt_files = [f.rsplit(".", 1)[0] + ".pts" for f in files]
    # ne = [f for f in pt_files if not os.path.exists(f)]
    # print (ne)
    # assert len(ne) == 0
    # return pt_files
    return files 

def load_pts(data_dir):
    pt_files = list_files(data_dir)
    return [pts_importer(f) for f in pt_files]


def make_pca(data_dir, out_file, normalize_rot=False, rotation_idxs=()):
    """
    Creates a PCA from data in a given directory
    
    Parameters
    ----------
    data_dir : str
        directory containing the image and landmark files
    out_file : str
        file the pca will be saved to
    normalize_rot : bool, optional
        whether or not to normalize the data's rotation
    rotation_idxs : tuple, optional
        indices for rotation normalization, msut be specified if 
        ``normalize_rot=True``
    
    """

    data_dir = os.path.abspath(data_dir)
    out_file = os.path.abspath(out_file)

    lmks = load_pts(data_dir)
    print ('no imgs with points = ', len(lmks))
    assert lmks is not None
    # make use of existing class
    # data = [SingleShapeDataProcessing(None, d) for d in lmks]

    if normalize_rot:
        print('normalize rotation...')
        for idx in range(len(lmks)):
            processor = SingleShapeDataProcessing(None, lmks[idx])
            lmks[idx] = processor.normalize_rotation(rotation_idxs[0], rotation_idxs[1])

    print('calculate pca...')
    pca = lmk_pca(np.array(lmks))
    # print (pca)
    assert out_file.endswith('.npz')
    np.savez(out_file, shapes=pca)

def prepare_dlib_dset(data_dir):
    """
    Prepare datasets similar to one available in dlib.
    Download according to this instruction 
    https://medium.com/datadriveninvestor/training-alternative-dlib-shape-predictor-models-using-python-d1d8f8bd9f5c
    """
    print('Prepare dlib dataset ', data_dir)    
    dsets = ['afw', 'helen', 'ibug', 'lfpw']
    # dsets = ['helen', 'ibug', 'lfpw']
    for ds in dsets:
        p = os.path.join(data_dir, ds)
        train_dir = os.path.join(p, 'trainset')
        train_dir = train_dir if os.path.exists(train_dir) else p
        output = os.path.join(data_dir, ds + ".train_pca.npz")
        if not os.path.exists(output):
            print ('make pca for ', ds)
            make_pca(train_dir,
                  output, 
                  rotation_idxs=(37, 46))
        else:
            print (ds, ' is already preprocessed') 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        help="Path to dataset dir",
                        type=str)
    args = parser.parse_args()
    data_dir = args.dataset
    prepare_dlib_dset(data_dir if data_dir is not None else '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset')