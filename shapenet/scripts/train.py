from ..dataset import DataSet
from ..networks import ShapeNet
import numpy as np
import torch
from tqdm import trange
from tqdm.auto import tqdm
import math
import os

BATCH_SIZE = 1
N_COMPONENTS = 25

def load_pca(pca_path, n_components):
    return np.load(pca_path)['shapes'][:(n_components + 1)]

def load_dataset(path):
    return DataSet(path)

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
    total_batch = math.ceil(dataset.set_size() / batch_size)
    print_every = 100
    total_loss = 0

    for i in trange(0, total_batch):
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

        if (i + 1) % print_every == 0:
            tqdm.write('avg. train loss %.2f' % (total_loss / print_every))
            total_loss = 0


def eval(model, val_dataset, criteria, metrics, input_device, output_device):   
    eval_batch_size = 20
    total_batch = math.ceil(val_dataset.set_size()/ eval_batch_size)
    results = np.ndarray(val_dataset.set_size(), **val_dataset.get_label(0).shape)
    last_idx = 0

    loss_vals = {k:0 for k in criteria}
    metric_vals = {k:0 for k in metrics}

    for i in trange(0, total_batch):
        data, labels = val_dataset.next_batch(eval_batch_size)
        data = torch.from_numpy(data).to(input_device).to(torch.float)
        labels = torch.from_numpy(labels).to(output_device).to(torch.float)
        model.eval()
        with torch.no_grad():
            preds = model(data)
            loss_vals = {}
            total_loss = 0
            for key, fn in criteria.items():
                _loss_val = fn(preds, labels)
                loss_vals[key] += _loss_val.detach()
            for key, metric_fn in metrics.items():
                metric_vals[key] += metric_fn(preds, labels)

        results[last_idx:(last_idx + len(preds)), :, :] = preds
    metric_vals = {k:(v/total_batch) for k, v in metric_vals.items()}
    loss_vals = {k:(v/total_batch) for k, v in loss_vals.items()}
    return metric_vals, loss_vals, results


def get_last_train_state(model_dir):
    checkpoints = [n for n in os.listdir(model_dir) if n.startswith('shapenet_epoch_')]
    if len(checkpoints) > 0:
        checkpoints.sort(reverse=True)
        last_epoch = checkpoints[0]
        return torch.load(last_epoch)
    return None



def train(pca_path, train_data, val_data, model_dir, num_epochs = 200):    

    n_components = N_COMPONENTS    
    # load PCA
    pca = load_pca(pca_path, n_components)  
                
    # create network 
    net, input_device, output_device = create_nn(pca)
    
    # load data set
    train_dataset = load_dataset(train_data)    
    val_dataset = load_dataset(val_data)

    # define optimizers
    optimizer = create_optimizer(net)
    start_epoch = 0
    # load latest epoch if available        
    saved_state = get_last_train_state(model_dir)
    if saved_state is not None:
        print ('load saved state')
        net.load_state_dict(saved_state['model'])    
        optimizer.load_state_dict(saved_state['optimizer'])
        start_epoch = saved_state['epoch']
    # loss function
    criteria = {"L1": torch.nn.L1Loss()}    
    # train - just set the mode to 'train'    
    net.train()        
    save_freq = 1
    for epoch in range(start_epoch, num_epochs+1):
        # train a single epoch
        train_single_epoch(net, optimizer, criteria, train_dataset, input_device, output_device)

        # save model after epoch
        if (epoch + 1) % save_freq == 0 or epoch == num_epochs:
            torch.save(dict(epoch=epoch, 
                model=net.state_dict(), 
                optimizer=optimizer.state_dict()), os.path.join(model_dir, 'shapenet_epoch_%d.pth'% epoch))

        #validate 
        print('eval at end of epoch')
        metric_vals, loss_vals, preds = eval(net, val_dataset, criteria, {}, input_device, output_device)

        print('val loss', loss_vals, ' metrics ', metric_vals)
        #save state
        
        
def run_train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir",
                        help="Path to dataset dir",
                        type=str)
    args = parser.parse_args()
    data_dir = args.datadir
    assert data_dir is not None
    pca_path = os.path.join(data_dir, 'train_pca.npz')
    train_data = os.path.join(data_dir, 'labels_ibug_300W_train.npz')
    val_data = os.path.join(data_dir, 'labels_ibug_300W_test.npz')
    model_dir = os.path.join(data_dir, 'model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    train(pca_path, train_data, val_data, model_dir)  

if __name__ == '__main__':
    run_train()
