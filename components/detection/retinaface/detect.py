from __future__ import print_function

import os
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from components.detection.retinaface.layers.functions.prior_box import PriorBox
from components.detection.retinaface.data import run_face_extractor_cfg, model_cfg_re50
from components.detection.retinaface.load import load_model
from components.detection.retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from components.detection.retinaface.definitions.retinaface import RetinaFace
from components.detection.retinaface.utils.box_utils import decode, decode_landm


def _circle2xy(x_c, y_c, r):
    return (x_c - r, y_c - r), (x_c + r, y_c + r)


class ModelManager:
    _model = None

    @classmethod
    def _load_model(cls, model_cfg, run_cfg):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        # Load model
        torch.set_grad_enabled(False)
        net = RetinaFace(cfg=model_cfg, phase='test')
        net = load_model(net, os.path.join(dir_path, run_cfg['pretrained_model_path']), run_cfg['use_cpu'])
        net.eval()

        # Configure device
        cudnn.benchmark = True
        device = torch.device("cpu" if run_cfg['use_cpu'] else "cuda")
        net = net.to(device)

        return net, device

    @classmethod
    def get_model(cls, model_cfg, run_cfg):
        if cls._model is not None:
            return cls._model
        else:
            loaded = cls._load_model(model_cfg, run_cfg)
            cls._model = loaded
            return loaded


def _save_img(img, dets, run_cfg, run_id):
    draw = ImageDraw.Draw(img)
    for b in dets:
        if b[4] < run_cfg['vis_thres']:
            continue
        text = "{:.4f}".format(b[4])
        draw.rectangle(b[:4], outline='#0000ea', width=2)

        cx = b[0]
        cy = b[1] + 12

        draw.text((cx, cy), text, '#000000', ImageFont.load_default(16), align="left")

        # Landmarks
        draw.ellipse(_circle2xy(b[5], b[6], 3), '#ff0000', width=3)
        draw.ellipse(_circle2xy(b[7], b[8], 3), '#00ffff', width=3)
        draw.ellipse(_circle2xy(b[9], b[10], 3), '#ff00ff', width=3)
        draw.ellipse(_circle2xy(b[11], b[12], 3), '#00ff00', width=3)
        draw.ellipse(_circle2xy(b[13], b[14], 3), '#ffff00', width=3)

        # Save image
        img.save(f'./log/imgs/{run_id}/detect.jpg')


def _transform_input(img, device):
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    return img


def _filter_prior_boxes(img, net_output, device, model_cfg, run_cfg):
    resize = run_cfg['resize']

    loc, conf, landms = net_output

    _, _, im_height, im_width = img.shape  # BCHW
    scale = torch.Tensor([im_width, im_height, im_width, im_height])
    scale = scale.to(device)

    priorbox = PriorBox(model_cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, model_cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, model_cfg['variance'])
    scale1 = torch.Tensor([im_width, im_height, im_width, im_height,
                           im_width, im_height, im_width, im_height,
                           im_width, im_height])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > run_cfg['confidence_threshold'])[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:run_cfg['top_k']]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, run_cfg['nms_threshold'])

    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:run_cfg['keep_top_k'], :]
    landms = landms[:run_cfg['keep_top_k'], :]

    dets = np.concatenate((dets, landms), axis=1)

    return dets


def run(input_image, run_id=None):
    # Set configuration
    model_cfg = model_cfg_re50
    run_cfg = run_face_extractor_cfg
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H_%M_%S_%f")

    # Load model
    net, device = ModelManager.get_model(model_cfg, run_cfg)

    # Pre-run transformations
    img = np.asarray(input_image, dtype=np.float32)
    img = _transform_input(img, device)

    # Forward pass
    net_output = net(img)
    dets = _filter_prior_boxes(img, net_output, device, model_cfg, run_cfg)

    # Show image
    if run_cfg['save_image']:
        _save_img(input_image, dets, run_cfg, run_id)

    return dets


if __name__ == '__main__':
    image_path = './curve/test.jpg'

    run(Image.open(image_path))

    run(Image.open(image_path))
