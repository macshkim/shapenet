import numpy as np
from ..networks import ShapeNet
import torch
from ..dataset import DataSet

BATCH_SIZE = 1

def load_pca(pca_path, n_components):
    return np.load(pca_path)['shapes'][:(n_components + 1)]

def load_dataset(path):
    return DataSet(path)

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


def train_single_epoch(model, optimizer, criterions, dataset):
    batch_size = 

# save_path: "../Results/helen"
# gpu_ids: [0]
# save_freq: 1
# num_epochs: 200
# val_score_key: "val_MSE"
def train(num_epochs = 200):
    n_components = 25
    pca_path = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/train_pca.npz'
    train_data = '/home/tamvm/Downloads/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.npz'
    
    # load PCA
    pca = load_pca(pca_path, n_components)

    # create network 
    net = create_nn(pca)

    # load data set
    dataset = load_dataset(train_data)

    # define optimizers
    optimizer = create_optimizer(net)
    # loss function
    criterions = {"L1": torch.nn.L1Loss()}

    # load latest epoch if available        
    start_epoch = 0
    # train - just set the mode to 'train'
    net.train()    
    for epoch in range(start_epoch, num_epochs+1):


        # train a single epoch
        train_single_epoch(net, optimizer, criterions, dataset)
        # save state to file
        # TODO:

        #
        net.train()





        with torch.enable_grad():
            inputs = None
            preds = net(inputs)

            #cal loss
            for key, fn in criterion.items():

