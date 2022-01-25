import torch
from torch import nn

class GIOU_Loss(nn.Module):
  def __init__(self, return_giou_arr=False):
    super(GIOU_Loss, self).__init__()
    self.return_giou_arr = return_giou_arr
  
  def forward(self, input_boxes, target_boxes):
    """
    Args:
        input_boxes: Tensor of shape (N, 4) or (4,).
        target_boxes: Tensor of shape (N, 4) or (4,).
    """
    x_p1, x_p2, y_p1, y_p2 = input_boxes[:, 0], input_boxes[:, 2], input_boxes[:, 1], input_boxes[:, 3]
    x_g1, x_g2, y_g1, y_g2 = target_boxes[:, 0], target_boxes[:, 2], target_boxes[:, 1], target_boxes[:, 3]

    A_g = (x_g2 - x_g1) * (y_g2 - y_g1)
    A_p = (x_p2 - x_p1) * (y_p2 - y_p1)
    x_I1, x_I2, y_I1, y_I2 = torch.max(x_p1, x_g1), torch.min(x_p2, x_g2), torch.max(y_p1, y_g1), torch.min(y_p2, y_g2) # intersection
    A_I = torch.clamp((x_I2 - x_I1) * (y_I2 - y_I1), min=0.) 
    A_U = A_g + A_p - A_I

    # area of the smallest enclosing box
    min_box = torch.min(input_boxes, target_boxes)
    max_box = torch.max(input_boxes, target_boxes)
    A_C = (max_box[:, 2] - min_box[:, 0]) * (max_box[:, 3] - min_box[:, 1])

    iou = A_I / A_U
    giou = iou - (A_C - A_U) / A_C
    loss = 1 - giou
    if self.return_giou_arr:
      return giou # (N, )
    return loss.sum()


if __name__ == '__main__':
    loss_fn = GIOU_Loss()
    pred_bbox = torch.tensor([[675.042, 491.1466, 746.4937, 548.2394]])
    gt_bbox = torch.tensor([[685.4099731445312, 493.8599853515625, 738.1400146484375, 540.1500244140625,]])
    print(loss_fn(pred_bbox, gt_bbox))