import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import imageio
import numpy as np
import matplotlib.pyplot as plt
import sys

class gta2segmDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of '/trainA/x.png' and '/trainB/x.png'.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # get the A image directory
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # get the A image directory
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # get A image paths
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # get image paths
        #assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        # print("A  shape = ", A.size)
        # print(A_path)
        # split AB image into A and B
        # w, h = AB.size
        # w2 = int(w / 2)
        # A = AB.crop((0, 0, w2, h))
        # B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        # transform_params_A = get_params(self.opt, A.size)
        # transform_params_B = get_params(self.opt, B.size)
        #print(np.array(B).min(), np.array(B).max())

        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # imageio.imwrite("/scratch2/dinka/pix2pix/gta2cityscapes/web/A_t.png", np.array(B))
        # sys.exit()
        #imageio.imwrite( "/scratch2/dinka/pix2pix/gta2cityscapes/web/A.png", np.array(A))
        A_tr = A_transform(A)
        B_tr = B_transform(B)

        # import ipdb; ipdb.set_trace()
        # imageio.imwrite("/scratch2/dinka/pix2pix/gta2cityscapes/web/A_t.png", np.array(A))
        #
        # imageio.imwrite("/scratch2/dinka/pix2pix/gta2cityscapes/web/B_t.png", np.array(B_tr.permute(1, 2, 0).numpy()))
        # sys.exit()
        return {'A': A_tr, 'B': B_tr, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return min(len(self.A_paths), len(self.B_paths))
