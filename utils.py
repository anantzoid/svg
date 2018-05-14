import math
import torch
import socket
import argparse
import os
import numpy as np
#from sklearn.manifold import TSNE
import scipy.misc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import functools
from skimage.measure import compare_psnr as psnr_metric
from skimage.measure import compare_ssim as ssim_metric
from scipy import signal
from scipy import ndimage
from PIL import Image, ImageDraw


from torchvision import datasets, transforms
from torch.autograd import Variable
import imageio


hostname = socket.gethostname()

def load_dataset(opt):
    if opt.dataset == 'smmnist':
        from data.moving_mnist import MovingMNIST
        train_data = MovingMNIST(
                train=True,
                data_root=opt.data_root,
                seq_len=opt.max_step,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits)
        test_data = MovingMNIST(
                train=False,
                data_root=opt.data_root,
                seq_len=opt.n_eval,
                image_size=opt.image_width,
                deterministic=False,
                num_digits=opt.num_digits)
    elif opt.dataset == 'bair':
        from data.bair import RobotPush 
        train_data = RobotPush(
                data_root=opt.data_root,
                train=True,
                seq_len=opt.max_step,
                image_size=opt.image_width,
                filterdata=opt.filterdata)
        test_data = RobotPush(
                data_root=opt.data_root,
                train=False,
                seq_len=opt.n_eval,
                image_size=opt.image_width)
    elif opt.dataset == 'epic':
        from data.epic import EpicKitchen 
        train_data = EpicKitchen(
                data_root=opt.data_root,
                train=True,
                seq_len=opt.max_step,
                image_size=opt.image_width,
                skip=opt.skip_frames+1)
        test_data = EpicKitchen(
                data_root=opt.data_root,
                train=False,
                seq_len=opt.n_eval,
                image_size=opt.image_width,
                skip=opt.skip_frames+1)
    return train_data, test_data

def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def normalize_data(opt, dtype, sequence):
    if opt.dataset == 'smmnist' or opt.dataset == 'kth' or opt.dataset == 'bair' or opt.dataset == 'epic':
        sequence.transpose_(0, 1)
        sequence.transpose_(3, 4).transpose_(2, 3)
    else:
        sequence.transpose_(0, 1)

    return sequence_input(sequence, dtype)

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def save_np_img(fname, x):
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    img = scipy.misc.toimage(x,
                             high=255*x.max(),
                             channel_axis=0)
    img.save(fname)

def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    return scipy.misc.toimage(tensor.numpy(),
                              high=255*tensor.max(),
                              channel_axis=0)

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        images.append(img.numpy())
    imageio.mimsave(filename, images, duration=duration)

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()
        images.append(img)
    imageio.mimsave(filename, images, duration=duration)

def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)

def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)

def prod(l):
    return functools.reduce(lambda x, y: x * y, l)

def batch_flatten(x):
    return x.resize(x.size(0), prod(x.size()[1:]))

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                ssim[i, t] += ssim_metric(gt[t][i][c], pred[t][i][c])
                psnr[i, t] += psnr_metric(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                res = finn_ssim(gt[t][i][c], pred[t][i][c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr


def finn_psnr(x, y):
    mse = ((x - y)**2).mean()
    return 10*np.log(1/mse)/np.log(10)


def gaussian2(size, sigma):
    A = 1/(2.0*np.pi*sigma**2)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = A*np.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
    return g

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
  
def finn_ssim(img1, img2, cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        try:
            m.bias.data.fill_(0)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        try:
            m.bias.data.fill_(0)
        except:
            pass



import torch.nn.functional as F
import operator
import functools
import torch.nn as nn
class Laplacian(nn.Module):
    def __init__(self, use_factor=True, max_levels=3):
        super(Laplacian, self).__init__()
        if use_factor == 1:
            self._factor = [100, 1, 0.1, 0.01, 0.001, 0.0001]#(2**(-2*(en+1)))
        else:
            self._factor = [1]*(max_levels+1)
        self.max_levels = max_levels

    def gauss_kernel(self, size=5, sigma=1.0):
        grid = np.float32(np.mgrid[0:size,0:size].T)
        gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
        kernel = np.sum(gaussian(grid), axis=2)
        kernel /= np.sum(kernel)
        return kernel

    def conv_gauss(self, t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
        #kernel = np.array(([self.gauss_kernel(k_size, sigma)]*9)).reshape(3,3,k_size,k_size)
        #kernel = Variable(torch.from_numpy(kernel).cuda()).float()       
        kernel = Variable(torch.from_numpy(self.gauss_kernel(k_size, sigma)).unsqueeze(0).unsqueeze(0)).cuda().float()
        pyr = Variable(torch.FloatTensor(t_input.size())).cuda()
        for i in range(t_input.size()[1]):
             pyr[:, i,:,:] = F.conv2d(t_input[:, i,:,:].unsqueeze(1), kernel, padding=2).squeeze(1)
        return pyr

    def make_laplacian_pyramid(self, x, max_levels):
        t_pyr = []
        current = x
        for level in range(max_levels):
            t_gauss = self.conv_gauss(current, stride=1, k_size=5, sigma=2.0)
            #save_tensors_image('diff.png', [current, t_gauss])
            #save_tensors_image('current.png', current)
            #save_tensors_image('gauss.png', t_gauss)
            #exit()
            t_diff = current - t_gauss
            t_pyr.append(t_diff)
            current = F.avg_pool2d(t_gauss, 2, 2)
        t_pyr.append(current)
        return t_pyr

    def tograyscale(self, x):
        x = torch.cat([(0.299*x[:,:1,:,:]), (0.587*x[:,1:2,:,:]), (0.114*x[:,2:,:,:])], dim=1)
        return torch.sum(x, dim=1).unsqueeze(1)


    def laploss(self, x_pred, x):
        #x_pred = self.tograyscale(x_pred)
        #x = self.tograyscale(x)

        t_pyr1 = self.make_laplacian_pyramid(x_pred, self.max_levels) 
        t_pyr2 = self.make_laplacian_pyramid(x, self.max_levels)
        #t_losses = [(a-b).norm(1)/float(functools.reduce(operator.mul, a.size())) for a,b in zip(t_pyr1, t_pyr2)]

        #t_losses = [(a-b).norm(1)*self._factor[en]/float(functools.reduce(operator.mul, a.size())) for en,(a,b) in enumerate(zip(t_pyr1, t_pyr2))]
        t_losses = [(a-b).norm(1)**(2**(-2*(en+1)))/float(functools.reduce(operator.mul, a.size())) for en,(a,b) in enumerate(zip(t_pyr1, t_pyr2))]
        #print(t_losses)
        #exit()
        return torch.sum(torch.cat(t_losses))#*x_pred.size()[0])
    
    def laplossnew(self, scales, x):
        t_pyr2 = []
        for s in range(len(scales)):
            t_gauss = self.conv_gauss(scales[s], stride=1, k_size=5, sigma=2.0)
            #current = F.avg_pool2d(scales[s], 2, 2)
            t_pyr2.append(scales[s] - t_gauss)

        t_pyr2 = t_pyr2[::-1]
        t_pyr1 = self.make_laplacian_pyramid(x, self.max_levels) 

        t_losses = [(a-b).norm(1)**(2**(-2*(en+1)))/float(functools.reduce(operator.mul, a.size())) for en,(a,b) in enumerate(zip(t_pyr1, t_pyr2))]
        return torch.sum(torch.cat(t_losses))#*x_pred.size()[0])

    def forward(self, x_pred, scales, x):
        return self.laplossnew(scales + [x_pred], x)
        #return self.laploss(x_pred, x)
