from .points_ops import *
# from mmdet.ops.points_op import points_op_cpu
import torch
import ctypes

def pts_in_boxes3d(pts, boxes3d):
    N = len(pts)
    M = len(boxes3d)
    lib = ctypes.CDLL('./points_op_cpu.cpython-36m-x86_64-linux-gnu.so')
    pts_in_flag = torch.IntTensor(M, N).fill_(0)
    reg_target = torch.FloatTensor(N, 3).fill_(0)
    # points_op_cpu.pts_in_boxes3d(pts.contiguous(), boxes3d.contiguous(), pts_in_flag, reg_target)
    lib.pts_in_boxes3d(pts.contiguous(), boxes3d.contiguous(), pts_in_flag, reg_target)
    return pts_in_flag, reg_target

