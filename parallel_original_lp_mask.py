
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/lp', help='base directory to save logs')
parser.add_argument('--model_dir', default='', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifier for directory')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--dataset', default='smmnist', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=5, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--n_eval', type=int, default=30, help='number of frames to predict during eval')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--z_dim', type=int, default=10, help='dimensionality of z_t')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--beta', type=float, default=0.0001, help='weighting on KL to prior')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--data_threads', type=int, default=5, help='number of data loading threads')
parser.add_argument('--num_digits', type=int, default=2, help='number of digits for moving mnist')
parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')
parser.add_argument('--mse', default=0, type=int, help='use mse else use masked mse loss')

parser.add_argument('--skip_frames', default=0, type=int, help='# of frames to skip in between when using epic dataset')



opt = parser.parse_args()
if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    #opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    #opt.log_dir = '%s/continued' % opt.log_dir

name = 'model=%s%dx%d-rnn_size=%d-rnn_layers=%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
if opt.dataset == 'smmnist':
    opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
else:
    opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)



os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)
opt.max_step = opt.n_past+opt.n_future

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
torch.cuda.set_device(0)
dtype = torch.cuda.FloatTensor
writer = SummaryWriter(log_dir=os.path.join('plots', opt.name))


# ---------------- load the models  ----------------

print(opt)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
    raise ValueError('Unknown optimizer: %s' % opt.optimizer)



import models.lstm_parallel as lstm_models
#import models.original_lstm as lstm_models

frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
if opt.model_dir != '':
    try:
        frame_predictor.load_state_dict(saved_model['frame_predictor'].state_dict())
        posterior.load_state_dict(saved_model['posterior'].state_dict())
    except:
        frame_predictor.load_state_dict(saved_model['frame_predictor'].module.state_dict())
        posterior.load_state_dict(saved_model['posterior'].module.state_dict())
    #prior = saved_model['prior']
else:
    #prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    #prior.apply(utils.init_weights)

if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as model 
    elif opt.image_width == 128:
        import models.dcgan_128 as model  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as model
    elif opt.image_width == 128:
        import models.vgg_128 as model
else:
    raise ValueError('Unknown model: %s' % opt.model)
       
encoder = model.encoder(opt.g_dim, opt.channels)
decoder = model.decoder(opt.g_dim, opt.channels)
if opt.model_dir != '':
    try:
        encoder.load_state_dict(saved_model['encoder'].state_dict())
        decoder.load_state_dict(saved_model['decoder'].state_dict())
    except:
        encoder.load_state_dict(saved_model['encoder'].module.state_dict())
        decoder.load_state_dict(saved_model['decoder'].module.state_dict())
else:
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
if opt.mse == 0:
    pixel_mse_criterion = nn.MSELoss(reduce=False)
    pixel_mse_criterion.cuda()

def kl_criterionL2(mu, logvar):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= opt.batch_size
  return KLD

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
#prior.cuda()
encoder.cuda()
decoder.cuda()
mse_criterion.cuda()

encoder = torch.nn.DataParallel(encoder, device_ids=range(torch.cuda.device_count()))
decoder = torch.nn.DataParallel(decoder, device_ids=range(torch.cuda.device_count()))
frame_predictor = torch.nn.DataParallel(frame_predictor, device_ids=range(torch.cuda.device_count()))
posterior = torch.nn.DataParallel(posterior, device_ids=range(torch.cuda.device_count()))
#prior = torch.nn.DataParallel(prior, device_ids=range(torch.cuda.device_count()))



# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()

# --------- plotting funtions ------------------------------------

def init_hidden():
    all_hidden = []
    for j in range(2): 
        hidden = []
        for i in range(2):
            hidden.append((Variable(torch.zeros(opt.batch_size, opt.rnn_size).cuda()),
                            Variable(torch.zeros(opt.batch_size, opt.rnn_size).cuda())))
        all_hidden.append(hidden)
    return all_hidden

def plot_rec(x, epoch, _type):
    all_hidden0, all_hidden1 = init_hidden()

    gen_seq, gt_seq, pred_seq = [], [], []
    gen_seq.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[i-1])
        h_target = encoder(x[i])
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h, _ = h
        h_target, _ = h_target
        h = h.detach()
        h_target = h_target.detach()
        z_t, _, _, all_hidden0= posterior(h_target, all_hidden0)
        if i < opt.n_past:
            #frame_predictor(torch.cat([h, z_t], 1)) 
            gen_seq.append(x[i])
        else:
            h_pred, all_hidden1 = frame_predictor(torch.cat([h, z_t], 1), all_hidden1)
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)
            pred_seq.append(x_pred.data.cpu().numpy())
            gt_seq.append(x[i].data.cpu().numpy())

    _, ssim, psnr = utils.eval_seq(gt_seq, pred_seq)
    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/%s_rec_%d.png' % (opt.log_dir, _type, epoch) 
    utils.save_tensors_image(fname, to_plot)
    
    return np.mean(ssim, axis=0), np.mean(psnr, axis=0)

# --------- training funtions ------------------------------------
def train(x):
    frame_predictor.zero_grad()
    posterior.zero_grad()
    #prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # initialize the hidden state.
    all_hidden0, all_hidden1 = init_hidden()

    mse = 0
    kld = 0

    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[i-1])
        h_target = encoder(x[i])[0]
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h = h[0]
        z_t, mu, logvar, all_hidden0 = posterior(h_target, all_hidden0)
        #_, mu_p, logvar_p = prior(h)
        h_pred, all_hidden1 = frame_predictor(torch.cat([h, z_t], 1), all_hidden1)
        x_pred = decoder([h_pred, skip])

        if opt.mse == 1:
            mse += mse_criterion(x_pred, x[i])
            _m = mse
        else:
            pmse = pixel_mse_criterion(x_pred, x[i])
            _m = mse_criterion(x_pred, x[i])
            #note: temp. refactor later
            pixel_weights = Variable(torch.zeros(opt.batch_size, opt.channels, opt.image_width, opt.image_width))
            pixel_weights = pixel_weights.cuda()
            pixel_weights[pmse.data > _m.data] = 1.0
            mse += torch.mean(pmse.mul(pixel_weights.detach()))

        kld += kl_criterionL2(mu, logvar)

    loss = mse + kld*opt.beta
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    #prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past), _m
# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    frame_predictor.train()
    posterior.train()
    #prior.train()
    encoder.train()
    decoder.train()
    epoch_mse = 0
    epoch_kld = 0
    epoch__mse = 0

    #progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        #progress.update(i+1)
        x = next(training_batch_generator)
        # train frame_predictor 
        mse, kld, _mse = train(x)
        epoch_mse += mse
        epoch_kld += kld
        epoch__mse += _mse


    #progress.finish()
    #utils.clear_progressbar()

    #print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
    print('[%02d] mse loss: %.5f | kld loss: %.5f | true_mse loss: %.5f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, epoch__mse/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
    writer.add_scalar('mse', epoch_mse/opt.epoch_size, epoch)
    writer.add_scalar('kld', epoch_kld/opt.epoch_size, epoch)
    writer.add_scalar('true_mse', epoch__mse/opt.epoch_size, epoch)

    # plot some stuff
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    #prior.eval()
    
    #### NOTE uncomment this line while only eval
    #x = next(training_batch_generator)
    ssim, psnr = plot_rec(x, epoch, 'train')
    print("recon Train ssim: %.4f, psnr: %.4f"%(ssim[-1], psnr[-1]))
    #ssim, psnr = plot(x, epoch)
    #print("gen Train ssim: %.4f, psnr: %.4f"%(ssim, psnr))
    #x = next(testing_batch_generator)
    #ssim, psnr = plot_rec(x, epoch, 'test')
    #print("recon Test ssim: %.4f, psnr: %.4f"%(ssim[-1], psnr[-1]))
    #ssim, psnr = plot(x, epoch)
    #print("gen Test ssim: %.4f, psnr: %.4f"%(ssim, psnr))

    # save the model
    torch.save({
        'encoder': encoder.module,
        'decoder': decoder.module,
        'frame_predictor': frame_predictor.module,
        'posterior': posterior.module,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)



