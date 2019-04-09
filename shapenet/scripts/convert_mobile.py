import os
import torch
from .train import load_model, load_pretrain_model
from ..dataset import DataSet
import numpy as np

def get_dummy_data():
    # train_data = os.path.join(data_dir, 'labels_ibug_300W_train.npz')
    # ds = DataSet(train_data)
    # return ds.data[0:1]
    batch_size = 1
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    return x

def get_img_data(data_dir):
    train_data = os.path.join(data_dir, 'labels_ibug_300W_train.npz')
    ds = DataSet(train_data)
    return ds.data[0:1]

def run_on_caffe2():
    import onnx
    import caffe2.python.onnx.backend as onnx_caffe2_backend
    model = onnx.load('shapenet2.onnx')
    # print('model ', model)
    prepared_backend = onnx_caffe2_backend.prepare(model)
    # print('model input ', 
    W = {model.graph.input[0].name: get_img_data('/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset')}
    # W = {model.graph.input[0].name: get_dummy_data().data.numpy()}
    c2_out = prepared_backend.run(W)[0]
    # print('output', c2_out[0])

def convert(data_dir):
    # model, _, _, _ = load_pretrain_model(data_dir)
    model, _, _, _, _ = load_model(data_dir, 0.0001)
    model.eval()
    # aa = get_dummy_data(data_dir)
    # print ('aa shape = ', aa.shapenet)
    x = get_dummy_data()
    torch_out = torch.onnx._export(model, x, 'shapenet.onnx', export_params=True)
    # print('result = ', r2)
    verify = True
    if verify:
        import onnx
        import caffe2.python.onnx.backend as onnx_caffe2_backend

        # Load the ONNX ModelProto object. model is a standard Python protobuf object
        model = onnx.load("shapenet.onnx")

        # prepare the caffe2 backend for executing the model this converts the ONNX model into a
        # Caffe2 NetDef that can execute it. Other ONNX backends, like one for CNTK will be
        # availiable soon.
        prepared_backend = onnx_caffe2_backend.prepare(model)

        # run the model in Caffe2

        # Construct a map from input names to Tensor data.
        # The graph of the model itself contains inputs for all weight parameters, after the input image.
        # Since the weights are already embedded, we just need to pass the input image.
        # Set the first input.
        W = {model.graph.input[0].name: x.data.numpy()}

        # Run the Caffe2 net:
        c2_out = prepared_backend.run(W)[0]

        # Verify the numerical correctness upto 3 decimal places
        np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",
                        help="Path to dataset dir",
                        type=str)
    args = parser.parse_args()
    data_dir = args.datadir
    if True:
        data_dir = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset'
    convert(data_dir)
    # run_on_caffe2()