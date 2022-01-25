import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import os
from tqdm import tqdm 
import numpy as np
from numpy import linalg as LA
from lap import lapjv
import pandas as pd
from util.loss import GIOU_Loss
from smoke.modeling.smoke_coder import SMOKECoder


def get_cam_matrix(cam_matrix):
    if cam_matrix == '0929':
        K = [ 1189.964744, 0.000000, 735.409035, 
              0.000000, 1189.824753, 518.136149, 
              0.000000, 0.000000, 1.000000 ]
    elif cam_matrix == '0529':
        K = [ 849.177455, 0.000000, 712.166787, 
             0.000000, 854.207389, 543.445028, 
             0.000000, 0.000000, 1.000000 ]
    elif 'kitti':
        K = [ 7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 
             0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 
             0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00 ]

    return torch.tensor(K).reshape(1, 3, 3)

def seq_to_instances(gt_file, type='car'):
    result = {}
    with open(gt_file) as gt_f:
        for line in gt_f:
            line = line.strip().split(' ')
            seq_num, instance_type, x_min, y_min, x_max, y_max = line[0], line[2], float(line[4]), float(line[5]), float(line[6]), float(line[7])
            if instance_type == type:
                if not seq_num in result.keys():
                    result[seq_num] = [[x_min, y_min, x_max, y_max]]
                else:
                    result[seq_num] += [[x_min, y_min, x_max, y_max]]

    return result # {'seqnum': [ [x_min, y_min, x_max, y_max] of cars]}




# gt_cars, pred_cars: (N, 4), (N, 11)
def gather_gt(gt_cars, pred_cars):
    gt_cars, pred_cars = np.array(gt_cars, dtype=np.float32), np.array(pred_cars, dtype=np.float32)
    N, M = pred_cars.shape[0], gt_cars.shape[0]
    pred_car_center = np.expand_dims(np.concatenate(( (pred_cars[:, 0:1] + pred_cars[:, 2:3]) / 2, (pred_cars[:, 1:2] + pred_cars[:, 3:4]) / 2 ), axis=1), axis=1) # (N, 1, 2)
    gt_car_center = np.expand_dims(np.concatenate(( (gt_cars[:, 0:1] + gt_cars[:, 2:3]) / 2, (gt_cars[:, 1:2] + gt_cars[:, 3:4]) / 2 ), axis=1), axis=0) # (1, M, 2)
    
    gt_car_center = np.repeat(gt_car_center, N, axis=0) # (N, M, 2)
    cost_mat = LA.norm(gt_car_center - pred_car_center, axis=2) # (N, M)
    _, gt_index, _ = lapjv(cost_mat, extend_cost=True)
    idxs = [i for i in range(N)]
    return gt_cars[gt_index], pred_cars, idxs # (N, 4), (N, 11), [ N ]

# filter out 2d bbox with giou < giou_threshold
def giou_filter(gt_cars, pred_cars, idxs, giou_threshold=0.4):
    giou = GIOU(torch.tensor(pred_cars[:, :4]), torch.tensor(gt_cars)) # (N, ) tensor
    new_idxs = (giou > giou_threshold).nonzero().flatten().tolist()
    new_idxs = list(set(idxs) & set(new_idxs))
    return gt_cars, pred_cars, new_idxs

# filter out 3d bbox outside of FOV
def bbox_3d_filter(gt_cars, pred_cars, idxs, scale_xyz=(1.3686, 1.1231, 1.3977)):
    pred_car_3d = torch.tensor(pred_cars[:, 4:])
    dims = pred_car_3d[:, :3] # (N, 3)
    # change dims from h w l to l h w
    dims = dims.roll(shifts=1, dims=1)
    locs = pred_car_3d[:, 3:6] # (N, 3)
    locs[:, 0] = locs[:, 0] * scale_xyz[0]
    locs[:, 1] = locs[:, 1] * scale_xyz[1]
    locs[:, 2] = locs[:, 2] * scale_xyz[2]
    yaws = pred_car_3d[:, 6] # (N, )
    w_h = torch.tensor([1440, 864])
    pred_bb = sc.encode_box2d(K, yaws, dims, locs, w_h, truncate=False)
    new_idxs = ((pred_bb[:, 0] > -1) & (pred_bb[:, 2] > -1) & (pred_bb[:, 1] < w_h[0]) & (pred_bb[:, 3] < w_h[1])).nonzero().flatten().tolist()
    new_idxs = list(set(idxs) & set(new_idxs))
    return gt_cars, pred_cars, new_idxs

    
# Match the 2D bbox from MaskRCNN (gt) with SMOKE (pred)
def prepare_training_data(pred_folder, gt_file, save_file):
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    seq_to_cars = seq_to_instances(gt_file)
    pred_files = [name for name in sorted(os.listdir(pred_folder))]
    save_file_lines = []
    for index in tqdm(range(len(pred_files))):
        prediction_file = os.path.join(pred_folder, pred_files[index])
        with open(prediction_file, 'r') as p_f: 
            objs = [] # [ [x_min, y_min, x_max, y_max, h, w, l, x, y, z, yaw] of objs]
            lines = []
            other_lines = [] # objects other than cars
            for line in p_f:
                line = line.strip()
                line_info = line.split(' ')
                if line_info[0] == 'Car':
                    lines.append(f'{index} {line}')
                    objs.append([float(line_info[4]), float(line_info[5]), float(line_info[6]), float(line_info[7]), float(line_info[8]),  float(line_info[9]), float(line_info[10]), float(line_info[11]), float(line_info[12]), float(line_info[13]), float(line_info[14]) ] )
                else:
                    other_lines.append(f'{index} {line}\n')

            if len(objs) > 0:
                gt_cars_bbox, _, index = bbox_3d_filter( *giou_filter( *gather_gt(seq_to_cars[str(index)], objs) ) )
                for i, line in enumerate(lines):
                    if i in index:
                        lines[i] = f'{line} {gt_cars_bbox[i][0]} {gt_cars_bbox[i][1]} {gt_cars_bbox[i][2]} {gt_cars_bbox[i][3]}\n'
                    else: 
                        lines[i] = f'{line} \n'
            save_file_lines += (lines + other_lines)
    save_file_lines[-1] = save_file_lines[-1].strip()
    with open(save_file, 'w') as s_f:
        s_f.writelines(save_file_lines)

def test_read_line(file_path):
    lines = []
    with open(file_path, 'r+') as f:
        for line in f:
            lines.append(line.strip() + ' 0\n')
        
        lines[-1] = lines[-1].strip()
        f.seek(0)
        f.truncate()
        f.writelines(lines)

def test_write_lines(save_file):
    lines = ['1\n', '2\n', '3\n']
    with open(save_file, 'w') as s_f:
        s_f.writelines(lines)


def extract_car_info(line):
    index = [9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
    car_info = []
    for i in index:
        car_info.append(float(line[i]))
    return car_info


class CarDetectionSet(Dataset):
    def __init__(self, file):
        print('Loading car detection dataset...')
        self.cars_info = [] # (N, 11): h, w, l, x, y, z, yaw (rot_y), bbox from maskrcnn: x_min, y_min, x_max, y_max
        with open(file, 'r') as f: 
            for line in f:
                line = line.strip().split(' ')
                # ignore objects which don't have bbox from maskrcnn
                if len(line) == 21:
                    car_info = extract_car_info(line)
                    self.cars_info.append(car_info)

    def __len__(self):
        return len(self.cars_info)

    def __getitem__(self, idx):
        smoke_result = torch.tensor(self.cars_info[idx][:7], dtype=torch.float32) # (7, )
        bbox = torch.tensor(self.cars_info[idx][7:], dtype=torch.float32) # (4, )
        return smoke_result, bbox


if __name__ == '__main__':
    pred_folder = '/home/andy/SMOKE/tools/logs/inference/CRUW4/data' # Car, Cyclist, Pedestrian
    gt_file = '/mnt/disk1/CRUW/ROD2021/maskrcnn_results/train/txts/2019_09_29_ONRD001_.txt' # car, person, truck, rider, bus, bicycle, motorcycle
    save_file = '/home/andy/bb_optimize/data/cruw/2019_09_29_ONRD001_filtered.txt'
    
    # Prepare the data for optimize scale and ptich angles
    GIOU = GIOU_Loss(return_giou_arr=True)
    sc = SMOKECoder(depth_ref=(28.01, 16.32),
                dim_ref=((3.88, 1.63, 1.53),
                            (1.78, 1.70, 0.58),
                            (0.88, 1.73, 0.67)), device='cpu')
    K = get_cam_matrix('0929') # change here
    # prepare_training_data(pred_folder, gt_file, save_file)


    # Test lapjav
    # cost_mat = np.array([[0, 1], [1, 0]])
    # cost, gt_index, _ = lapjv(cost_mat, extend_cost=True)

    # Test dataset
    # train_set = CarDetectionSet(save_file)
    # print(len(train_set))
    # print(train_set[0])

    # Inspect data file
    # data = pd.read_csv(gt_file, sep=" ", header=None)
    # print(data[2].drop_duplicates())