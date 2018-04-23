import os
import io
from scipy.misc import imresize 
import numpy as np
from PIL import Image
from scipy.misc import imresize
from scipy.misc import imread 


class EpicKitchen(object):
    def __init__(self, data_root, train=True, seq_len=20, image_size=64, skip=1, frame_limit=1000):
        self.root_dir = data_root
        if train:
            self.data_dir = '%s/train_processed' % self.root_dir 
            self.ordered = False
        else:
            self.data_dir = '%s/test_processed' % self.root_dir 
            self.ordered = True 
 
        self.image_size = image_size
        self.skip = skip
        self.seq_len = seq_len * self.skip
        self.dirs, self.num_img = [], []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir(os.path.join(self.data_dir, d1)):
                _f = os.path.join(self.data_dir, d1, d2)
                self.dirs.append(_f)
                #self.num_img.append(os.listdir(_f))

        self.num_img = len(os.listdir(self.path))
        self.d, self.img_counter = 0, 0
        self.seed_is_set = False # multi threaded loading
        self.frame_limit = frame_limit
        ##NOTE replace self.frame_limit by self.num_img[self.d] if using all frames

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def get_seq(self):
        if self.ordered:
            if self.d >= len(self.dirs):
                self.d = 1
            else:
                self.d += 1
            #random determinsitc choice:
            self.img_counter = self.frame_limit//2
        else:
            self.d = np.random.randint(0, len(self.dirs))
            self.img_counter = np.random.randint(1, self.frame_limit - self.seq_len)
        image_seq = []
        try:
            for i in range(self.seq_len):
                i_str = str(self.img_counter + i)
                fname = '%s/frame_%s%s.jpg' % (self.dirs[self.d], '0'*(10-len(i_str)) ,i_str)
                im = imread(fname)
                im = im.reshape(1, 128, 128, 3)
                image_seq.append(im/255.)
        except:
            # to account for any missing indices while retreiveing data
            return self.get_seq()

        image_seq = image_seq[::self.skip]
        image_seq = np.concatenate(image_seq, axis=0)
        return image_seq

    def __len__(self):
        return 10000

    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()
