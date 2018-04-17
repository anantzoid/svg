import os
import io
from scipy.misc import imresize 
import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread 


class EpicKitchen(object):
    def __init__(self, data_root, train=True, seq_len=20, image_size=64):
        self.path = '/beegfs/ag4508/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/test/P01/P01_13'
        if train:
            self.ordered = False
        else:
            self.ordered = True 
 
        self.image_size = image_size
        self.seq_len = seq_len
        self.num_img = len(os.listdir(self.path))
        self.d = 0
        self.seed_is_set = False # multi threaded loading

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
 
    def get_seq(self):
        if self.ordered:
            if self.d + self.seq_len >= self.num_img:
                self.d = 1
            else:
                self.d += 1
        else:
            self.d = np.random.randint(self.num_img-self.seq_len)
        image_seq = []
        for i in range(self.seq_len):
            i_str = str(self.d + i)
            fname = '%s/frame_%s%s.jpg' % (self.path, '0'*(10-len(i_str)) ,i_str)
            im = imread(fname)
            im = imresize(im[:, 100:356, :], (64, 64, 3))
            im = im.reshape(1, 64, 64, 3)
            image_seq.append(im/255.)
        image_seq = np.concatenate(image_seq, axis=0)
        return image_seq

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()
