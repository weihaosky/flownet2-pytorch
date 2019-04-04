import torch
import numpy as np
import argparse

from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module
from utils import flowlib

import cvbase as cvb

import IPython

if __name__ == '__main__':
    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument("--rgb_max", type=float, default=255.)
    args = parser.parse_args()

    # initial a Net
    net = FlowNet2(args).cuda()
    # load the state_dict
    dict = torch.load("model/FlowNet2_checkpoint.pth.tar")
    net.load_state_dict(dict["state_dict"])

    # load the image pair, you can find this operation in dataset.py
    pim1 = read_gen("dataset/my_photo-2cropped.png")
    pim2 = read_gen("dataset/my_photo-3cropped.png")

    # pad to (384, 1248)
    height = pim1.shape[0]  # 384 #736 #384  #384
    width = pim1.shape[1]  # 1248 #1504 #704     #1248
    pad_h = (height // 64 + 1) * 64
    pad_w = (width // 64 + 1) * 64
    top_pad = pad_h - pim1.shape[0]
    left_pad = pad_w - pim1.shape[1]
    pim1 = np.lib.pad(pim1, ((top_pad, 0), (0, left_pad), (0, 0)), mode='constant', constant_values=0)
    pim2 = np.lib.pad(pim2, ((top_pad, 0), (0, left_pad), (0, 0)), mode='constant', constant_values=0)

    images = [pim1, pim2]
    images = np.array(images).transpose(3, 0, 1, 2)
    im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

    # process the image pair to obtian the flow
    result = net(im).squeeze()


    # save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project
    def writeFlow(name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()


    data = result.data.cpu().numpy().transpose(1, 2, 0)
    output = "output/img.flo"
    writeFlow(output, data)

    cvb.show_flow(data)