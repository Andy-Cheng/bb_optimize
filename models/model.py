import torch
import torch.nn as nn
import numpy as np
from smoke.modeling.smoke_coder import SMOKECoder

class Scale_XYZ(nn.Module):
    def __init__(self, K, img_size):
        super(Scale_XYZ, self).__init__()
        self.scale_xyz = nn.Conv1d(3, 3, 1, groups=3, bias=False) # (3, 1, 1)
        # initialize scale to be 1
        self.scale_xyz.weight = torch.nn.Parameter(torch.ones_like(self.scale_xyz.weight))
        self.sc = SMOKECoder(depth_ref=(28.01, 16.32),
                        dim_ref=((3.88, 1.63, 1.53),
                                 (1.78, 1.70, 0.58),
                                 (0.88, 1.73, 0.67)))
        self.K = K # camera matrix          
        self.img_size = torch.tensor(img_size) # (w, h)
        # self.xyz_scale = nn.Parameter(torch.tensor([1., 1., 1.]).reshape((1, 3))) # requires_grad attribute is set True by default for nn.Parameters
        
        

    # x is (N, 7), 7: h, w, l, x, y, z, yaw (rot_y)
    def forward(self, cars):
        dims = cars[:, :3] # (N, 3)
        # change dims from h w l to l h w
        dims = dims.roll(shifts=1, dims=1)
        locs = cars[:, 3:6].unsqueeze(2) # (N, 3, 1)
        yaws = cars[:, 6] # (N, )

        locs = self.scale_xyz(locs)
        # locs = (locs @ self.xyz_scale).diagonal(dim1=-2, dim2=-1)
        

        pred_bb = self.sc.encode_box2d(self.K, yaws, dims, locs, self.img_size)
        return pred_bb # pred_bb: (N, 4)

if __name__ == '__main__':
    K = [ 1189.964744, 0.000000, 735.409035, 
    0.000000, 1189.824753, 518.136149, 
    0.000000, 0.000000, 1.000000 ]
    K = torch.tensor(K).reshape(1, 3, 3)
    model = Scale_XYZ(K, [1440, 864])
    cars = torch.tensor([[1.386199951171875, 1.5825999975204468, 3.5445001125335693, -0.583899974822998, 0.73089998960495, 30.701099395751953, -1.516700029373169]])
    pred_bb = model(cars)
    print(pred_bb)


