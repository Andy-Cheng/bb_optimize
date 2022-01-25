import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import argparse
from models import Scale_XYZ
from util import CarDetectionSet, GIOU_Loss, get_cam_matrix
import sys
sys.path.append('/home/andy/bb_optimize')

def default_argument_parser():
    parser = argparse.ArgumentParser(description='Scale and pitch optimization')
    parser.add_argument('--img_size', type=int, help='image w, h', nargs='+', default=[1440, 864])
    parser.add_argument('--cam_matrix', default='0929', help='camera matrix type')
    parser.add_argument('--training_data_path', default='/home/andy/bb_optimize/data/cruw/2019_09_29_ONRD001.txt', help='training data file path')
    parser.add_argument('--model_path', default='/home/andy/bb_optimize/weights/global_scale.pth', help='file path to save the model weight')

    return parser


def trainLoop(dataLoader, model, loss_fn, optimizer, epoch, device):
    model.train()
    print('Epoch {}\n-------------------------------'.format(epoch+1))
    running_loss = 0.0
    batch_count = len(dataLoader)
    batch_to_show = int(batch_count/4)  # Show 4 batchs' results in each epoch
    acc_sample = 0
    loss_list = []
    giou_list = []
    for batch, (car_3D, car_bbox) in enumerate(dataLoader):
        acc_sample += car_3D.shape[0]
        car_3D, car_bbox = car_3D.to(device), car_bbox.to(device)
        optimizer.zero_grad()
        scaled_bbox = model(car_3D)
        loss = loss_fn(scaled_bbox, car_bbox)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # loss_list.append(loss.item())
        # giou_list.append(loss.item()/car_3D.shape[0])
        
        if batch % batch_to_show == (batch_to_show-1):
            running_loss /= float(acc_sample)
            print(f'Loss: {running_loss:>7f}, GIoU: {1 - running_loss:>7f} per sample, Scale: {model.scale_xyz.weight.view(3,)}  [{batch+1}/{batch_count}]') # Scale: {model.xyz_scale.flatten()}
            running_loss = 0.0
            acc_sample = 0
            
    


def train(n_epoch, model, loss_fn, optimizer, device, train_set, batch_size, n_valid = 0):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    for epoch in tqdm(range(n_epoch)):
    # In each epoch, split training dataset into training and validation datasets
        # train_dataset, val_dataset  = torch.utils.data.random_split(train_set, [len(train_set) - n_valid, n_valid])
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        trainLoop(train_loader, model, loss_fn, optimizer, epoch, device)
    print('Done training!')

def main(args):
    n_epoch = 20
    device = torch.device('cuda')
    K = get_cam_matrix(args.cam_matrix)
    scale_xyz  = Scale_XYZ(K, args.img_size)
    loss_fn = GIOU_Loss()
    optimizer = optim.SGD(scale_xyz.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    train_set = CarDetectionSet(args.training_data_path)
    batch_size = 1
    train(n_epoch, scale_xyz, loss_fn, optimizer, device, train_set, batch_size)
    torch.save(scale_xyz.state_dict(), args.model_path)

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print('Command Line Args:', args)
    print(args.img_size)
    main(args)