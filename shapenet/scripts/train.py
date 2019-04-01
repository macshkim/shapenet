import numpy as np
from ..networks import ShapeNet
import torch

def load_pca(pca_path, n_components):
    return np.load(pca_path)['shapes'][:(n_components + 1)]

def create_nn(pca):
    net = ShapeNet(pca)
    # check device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            
            gpu_ids = [i for i in range(0, device_count)]
            input_device = torch.device('cuda:%d' % gpu_ids[0])
            net = torch.nn.DataParallel(net.to(input_device),
                    device_ids=gpu_ids,
                    output_device=gpu_ids[1]
                )
        else:
            input_device = torch.device('cuda:0')
            net = net.to(input_device)
    else:
        raise Error('Not implemented yet')
    return net

def create_optimizer(model, lr=0.0001):
    # TODO: read more about mix-precision optimizer https://forums.fast.ai/t/mixed-precision-training/20720
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, momentum=momentum)


# save_path: "../Results/helen"
# gpu_ids: [0]
# save_freq: 1
# num_epochs: 200
# val_score_key: "val_MSE"
def train(num_epochs = 200):
    n_components = 25
    pca_path = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/train_pca.npz'

    # load PCA
    pca = load_pca(pca_path, n_components)

    # create network 
    net = create_nn(pca)

    # load data set
    # TODO:

    # define optimizers

    # load latest epoch if available    
    # TODO:
    start_epoch = 0

    

    # train - just set the mode to 'train'
    net.train()

    # save state to file
    # TODO:

    #
    for epoch in range(start_epoch, num_epochs+1):
        # train a single epoch
        net.train()

        with torch.enable_grad():
            inputs = None
            preds = net(inputs)

