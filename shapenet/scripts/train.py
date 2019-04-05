from ..dataset import DataSet
from ..networks import ShapeNet
from .preprocess import view_img
import numpy as np
import torch
from tqdm import trange
from tqdm.auto import tqdm
import math
import os
from random import randrange
import re

BATCH_SIZE = 1
N_COMPONENTS = 68
TRAIN_EPOCHS = 1000
DEBUG_SINGLE_IMG = None
PCA_FILE = 'unrot_train_pca.npz'

def load_pca(pca_path, n_components):
    return np.load(pca_path)['shapes'][:(n_components + 1)]

def load_dataset(path):    
    return DataSet(path, 1 if DEBUG_SINGLE_IMG is not None else None)

def create_nn(pca):
    net = ShapeNet(pca)
    input_device, output_device = None, None
    # check device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print('cuda available. no gpu = ', device_count)
        if device_count > 1:            
            gpu_ids = [i for i in range(0, device_count)]            
            input_device = torch.device('cuda:%d' % gpu_ids[0])
            net = torch.nn.DataParallel(net.to(input_device),
                    device_ids=gpu_ids,
                    output_device=gpu_ids[1]
                )
            output_device = torch.device('cuda:%d' % gpu_ids[1])
        else:
            input_device = torch.device('cuda:0')
            net = net.to(input_device)
            output_device = torch.device('cuda:0')        
    else:
        print('gpu not available. train using cpu instead')
        input_device = torch.device('cpu') 
        output_device = torch.device('cpu') 
        net = net.to(input_device)
    return net, input_device, output_device

def create_optimizer(model, lr=0.0001):
    # TODO: read more about mix-precision optimizer https://forums.fast.ai/t/mixed-precision-training/20720
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer

def train_single_epoch(model, optimizer, criteria, dataset, input_device, output_device):
    batch_size = BATCH_SIZE
    if DEBUG_SINGLE_IMG is None:
        total_batch = math.ceil(dataset.set_size() / batch_size)
    else:
        total_batch = 50

    total_loss = 0

    for i in trange(0, total_batch):
        # if DEBUG_SINGLE_IMG is not None:
        #     data, labels = dataset.data[DEBUG_SINGLE_IMG], dataset.labels[DEBUG_SINGLE_IMG]
        #     data = data.reshape(data.shape[0], 1, *data.shape[1:])
        # else:
        data, labels = dataset.next_batch(batch_size)
        data = torch.from_numpy(data).to(input_device).to(torch.float)
        labels = torch.from_numpy(labels).to(output_device).to(torch.float)
        model.train()

        with torch.enable_grad():            
            preds = model(data)
            #cal loss
            loss_vals = {}
            train_loss = 0
            for key, fn in criteria.items():
                _loss_val = fn(preds, labels)
                loss_vals[key] = _loss_val.detach()
                train_loss += _loss_val
                total_loss += _loss_val
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()           

    # print('avg. train loss %.2f' % (total_loss / total_batch))
    # total_loss = 0
    # test 
    # t = randrange(0, 100)
    # img = dataset.data[t]
    # img = img.reshape(*img.shape[1:])
    # lmks = predict(model, img, input_device)
    # view_img(img, lmks, dataset.labels[t])


def eval(model, val_dataset, criteria, metrics, input_device, output_device):   
    eval_batch_size = 20
    total_batch = math.ceil(val_dataset.set_size()/ eval_batch_size)
    results = np.ndarray((val_dataset.set_size(), *val_dataset.get_label(0).shape))
    last_idx = 0

    loss_vals = {k:0 for k, _ in criteria.items()}
    metric_vals = {k:0 for k,_ in metrics.items()}
    
    for i in trange(0, total_batch):
        data, labels = val_dataset.next_batch(eval_batch_size)
        data = torch.from_numpy(data).to(input_device).to(torch.float)
        labels = torch.from_numpy(labels).to(output_device).to(torch.float)
        model.eval()
        with torch.no_grad():
            preds = model(data)            
            total_loss = 0
            for key, fn in criteria.items():
                _loss_val = fn(preds, labels)
                loss_vals[key] += _loss_val.detach()
            for key, metric_fn in metrics.items():
                metric_vals[key] += metric_fn(preds, labels).detach()

        results[last_idx:(last_idx + len(preds)), :, :] = preds.cpu()
    metric_vals = {k:(v/total_batch) for k, v in metric_vals.items()}
    loss_vals = {k:(v/total_batch) for k, v in loss_vals.items()}
    return metric_vals, loss_vals, results


def load_pretrain_model(data_dir):
    input_device = output_device = torch.device('cpu')
    net = torch.jit.load(os.path.join(data_dir, 'pretrained_face.ptj'), map_location=input_device)    
    return net, None, input_device, output_device

def load_model(data_dir, lr):
    model_dir = os.path.join(data_dir, 'model')
    pca_path = os.path.join(data_dir, PCA_FILE)
    n_components = N_COMPONENTS    
    # load PCA
    pca = load_pca(pca_path, n_components) 
    # create network 
    net, input_device, output_device = create_nn(pca)
    optimizer = create_optimizer(net, lr)
    last_epoch = 0
    checkpoints = [n for n in os.listdir(model_dir) if n.startswith('shapenet_epoch_')]
    if len(checkpoints) > 0:
        get_epoch = lambda x: int(re.search('\\d+', x).group(0))
        checkpoints = sorted(checkpoints, key=get_epoch, reverse=True)
        last_epoch_f = checkpoints[0]
        print ('load saved state from ', last_epoch_f)
        saved_state = torch.load(os.path.join(model_dir, last_epoch_f), map_location=input_device)        
        net.load_state_dict(saved_state['model']) 
        optimizer.load_state_dict(saved_state['optimizer'])
        for g in optimizer.param_groups:
            g['lr'] = lr
        last_epoch = get_epoch(last_epoch_f)
    return net, optimizer, input_device, output_device, last_epoch

def save_model(data_dir, model, optimizer, name):
    model_dir = os.path.join(data_dir, 'model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(dict(model=model.state_dict(), 
                    optimizer=optimizer.state_dict()), os.path.join(model_dir, name))

def predict(model, img, input_device):
    if len(img.shape) < 4:
        data = np.array([img.reshape(1, *img.shape)])
        
    else:
        data = img    
    data = torch.from_numpy(data).to(input_device).to(torch.float)
    model.eval()
    with torch.no_grad():
        preds = model(data)
        return preds.cpu()[0]

def train(data_dir, train_data, val_data, lr, eval_only = False, num_epochs = TRAIN_EPOCHS):    
    print('start training. lr = ', lr)
    net, optimizer, input_device, output_device, last_epoch = load_model(data_dir, lr)
    # load data set
    train_dataset = load_dataset(train_data) if not eval_only else None
    val_dataset = load_dataset(val_data)

    start_epoch = last_epoch
    criteria = {"L1": torch.nn.L1Loss()}    
    metrics = {"MSE": torch.nn.MSELoss()}

    if eval_only:
        print('test on test set')
        metric_vals, loss_vals, preds = eval(net, val_dataset, criteria, metrics, input_device, output_device)
        print('val loss', loss_vals, ' metrics ', metric_vals)        
    else:
        # train - just set the mode to 'train'    
        net.train()        
        save_freq = 1 if DEBUG_SINGLE_IMG is None else 100
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                mode='min', threshold=1e-4, threshold_mode='rel',
                verbose=True,
                factor=0.5, patience=5, cooldown=0, min_lr=0, eps=1e-8)
        for epoch in range(start_epoch, start_epoch +num_epochs):
            # train a single epoch            
            train_single_epoch(net, optimizer, criteria, train_dataset, input_device, output_device)                        
            # save model after epoch
            if (epoch + 1) % save_freq == 0 or epoch == start_epoch +num_epochs:
                save_model(data_dir, net, optimizer, 'shapenet_epoch_%d.pth'% epoch)
                # # test 
                # img = train_dataset.data[0]
                # img = img.reshape(*img.shape[1:])
                # lmks = predict(net, img, input_device)
                # view_img(img, lmks, train_dataset.labels[0])
            # validate 
            # print('eval at end of epoch')            
            metric_vals, loss_vals, preds = eval(net, val_dataset if DEBUG_SINGLE_IMG is None else train_dataset, criteria, metrics, input_device, output_device)
            tqdm.write('val loss = %.2f metric = %.2f ' % (loss_vals['L1'], metric_vals['MSE']))
            # TODO: save best
            scheduler.step(metrics=metric_vals['MSE'])

def run_train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",
                        help="Path to dataset dir",
                        type=str)
    parser.add_argument("--pcafile",
                        help="pca file name",
                        type=str)
    parser.add_argument("--evalonly", action="store_true",
                        help="do not train. only test on validation set",
                        default=False)
    parser.add_argument("--learnrate", type=float, default=0.0001,
                        help="Learning rate")
    args = parser.parse_args()
    data_dir = args.datadir
    evalonly = args.evalonly
    lr = args.learnrate
    if args.pcafile is not None:
        PCA_FILE = args.pcafile

    assert data_dir is not None
    train_data = os.path.join(data_dir, 'labels_ibug_300W_train.npz')
    val_data = os.path.join(data_dir, 'labels_ibug_300W_test.npz')    
    train(data_dir, train_data, val_data, lr, evalonly)  


if __name__ == '__main__':
    run_train()
