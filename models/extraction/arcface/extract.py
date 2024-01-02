import argparse

import numpy as np
import torch
from PIL import Image

from backbones import get_model


class ModelManager:
    _model = None

    @classmethod
    @torch.no_grad()
    def _load_model(cls, weights_path, arch):
        net = get_model(arch, fp16=True)
        net.load_state_dict(torch.load(weights_path))
        net.eval()
        return net

    @classmethod
    def get_model(cls, weights_path, arch):
        if cls._model is not None:
            return cls._model
        else:
            loaded = cls._load_model(weights_path, arch)
            cls._model = loaded
            return loaded


@torch.no_grad()
def run(img, weights='./weights/backbone.pth', network='r50'):

    img = np.asarray(img, dtype=np.float16)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    model = ModelManager.get_model(weights, network)

    id_vector = model(img).numpy()
    return id_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='./weights/backbone.pth')
    parser.add_argument('--img', type=str, default='./images/test.jpg')
    args = parser.parse_args()

    img = Image.open(args.img)
    print(run(img, args.weight, args.network))
