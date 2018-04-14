#### ORIGINAL script
# with option for high capacity enc-dec and LR decay (hard-coded)
# and using oringinal lstm models

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


parser.add_argument('--rec1', type=float, default=0, help='mse 1st part')
parser.add_argument('--beta2', type=float, default=0, help='kld 2nd part')
parser.add_argument('--load_all', type=int, default=0, help='load both models')
parser.add_argument('--joint', type=int, default=0, help='backprop from L2 to L1')

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
opt.max_step = opt.n_past+opt.n_future

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
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


import models.lstm as lstm_models
#import models.original_lstm as lstm_models
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
else:
    frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

if opt.load_all == 1:
    posterior_2 = saved_model['posterior_2']
    prior_2 = saved_model['prior_2']
else:
    #posterior_2 = lstm_models.gaussian_lstm(opt.g_dim+opt.z_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    #prior_2 = lstm_models.gaussian_lstm(opt.g_dim+opt.z_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    #posterior_2.apply(utils.init_weights)
    #prior_2.apply(utils.init_weights)
    posterior_2 = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    prior_2 = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    posterior_2.load_state_dict(posterior.state_dict())
    prior_2.load_state_dict(posterior.state_dict())
    latent_encoder =  nn.Linear(opt.z_dim, opt.g_dim)
    latent_encoder.apply(utils.init_weights)

if opt.model == 'highcap':
    import models.dcgan_64_high as model
elif opt.model == 'dcgan':
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
       
if opt.model_dir != '':
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
else:
    encoder = model.encoder(opt.g_dim, opt.channels)
    decoder = model.decoder(opt.g_dim, opt.channels)
    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)

if opt.load_all == 1:
    pred_encoder = saved_model['pred_encoder']
    pred_decoder = saved_model['pred_decoder']
else:
    pred_encoder = model.encoder(opt.g_dim, opt.channels)
    pred_decoder = model.decoder(opt.g_dim+opt.z_dim, opt.channels)
    #pred_encoder.apply(utils.init_weights)
    #pred_decoder.apply(utils.init_weights)
    #### Preload weights #####
    pred_encoder.load_state_dict(saved_model['pred_encoder'].state_dict())
    pred_decoder.load_state_dict(saved_model['pred_decoder'].state_dict())


'''
print("======Encoder==========")
print(encoder)
print("======Predictor==========")
print(frame_predictor)
print("======Decoder==========")
print(decoder)
print("======Posterior==========")
print(posterior)
print("======Prior==========")
print(prior)
exit()
'''

frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


posterior_2_optimizer = opt.optimizer(posterior_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_2_optimizer = opt.optimizer(prior_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
pred_decoder_optimizer = opt.optimizer(pred_decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
pred_encoder_optimizer = opt.optimizer(pred_encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
latent_encoder_optimizer = opt.optimizer(latent_encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()

def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
    #   log( sqrt(
    # 
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / opt.batch_size

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()
mse_criterion.cuda()

posterior_2.cuda()
prior_2.cuda()
pred_encoder.cuda()
pred_decoder.cuda()
latent_encoder.cuda()
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

def plot_rec(x, epoch, _type):
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()

    posterior_2.hidden = posterior_2.init_hidden()

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
        z_t, _, _= posterior(h_target)
        if i < opt.n_past:
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = decoder([h_pred, skip]).detach()
            
            x_pred_h = pred_encoder(x_pred)[0]
            x_pred_h.detach()
            #####z_t_2, mu_2, logvar_2 = posterior_2(torch.cat([h_target, z_t], 1))
            z_t_2, mu_2, logvar_2 = posterior_2(h_target + latent_encoder(z_t))
            x_pred_2 = pred_decoder([torch.cat([x_pred_h, z_t_2], 1), skip])

            gen_seq.append(x_pred_2)
            gt_seq.append(x[i].data.cpu().numpy())
            pred_seq.append(x_pred_2.data.cpu().numpy())
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
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    posterior_2.zero_grad()
    prior_2.zero_grad()
    pred_encoder.zero_grad()
    pred_decoder.zero_grad()

    # initialize the hidden state.
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    prior.hidden = prior.init_hidden()
    posterior_2.hidden = posterior_2.init_hidden()
    prior_2.hidden = prior_2.init_hidden()

    mse = 0
    kld = 0
    mse_2 = 0
    kld_2 = 0
    for i in range(1, opt.n_past+opt.n_future):
        h = encoder(x[i-1])
        h_target = encoder(x[i])[0]
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h = h[0]
        z_t, mu, logvar = posterior(h_target)
        _, mu_p, logvar_p = prior(h)
        h_pred = frame_predictor(torch.cat([h, z_t], 1))
        x_pred = decoder([h_pred, skip])

        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, mu_p, logvar_p)

        # 2nd level hierarchy
        if opt.joint == 0:
            x_pred = x_pred.detach()
            h_target = h_target.detach()
            skip = [_.detach() for _ in skip]
            h = h.detach()
            z_t = z_t.detach()

        x_pred_h = pred_encoder(x_pred)[0]
        #####z_t_2, mu_2, logvar_2 = posterior_2(torch.cat([h_target, z_t], 1))
        z_t_2, mu_2, logvar_2 = posterior_2(h_target + latent_encoder(z_t))
        x_pred_2 = pred_decoder([torch.cat([x_pred_h, z_t_2], 1), skip])

        ####_, mu_p_2, logvar_p_2 = prior_2(torch.cat([h, z_t], 1))
        _, mu_2, logvar_2 = prior_2(h + latent_encoder(z_t))

        mse_2 += mse_criterion(x_pred_2, x[i])
        kld_2 += kl_criterion(mu_2, logvar_2, mu_p_2, logvar_p_2)


    loss = opt.rec1*mse + kld*opt.beta + mse_2 + kld_2*opt.beta2
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    decoder_optimizer.step()
    encoder_optimizer.step()

    posterior_2_optimizer.step()
    prior_2_optimizer.step()
    pred_decoder_optimizer.step()
    pred_encoder_optimizer.step()
    latent_encoder_optimizer.step()


    _losses = (
           mse.data.cpu().numpy()/(opt.n_past+opt.n_future),
           kld.data.cpu().numpy()/(opt.n_past+opt.n_future),
           mse_2.data.cpu().numpy()/(opt.n_past+opt.n_future),
           kld_2.data.cpu().numpy()/(opt.n_past+opt.n_future),
           )
    return _losses

    

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    frame_predictor.train()
    posterior.train()
    prior.train()
    encoder.train()
    decoder.train()

    posterior_2.train()
    prior_2.train()
    pred_encoder.train()
    pred_decoder.train()
    latent_encoder.train()

    epoch_mse = 0
    epoch_kld = 0
    epoch_mse_2 = 0
    epoch_kld_2 = 0

    #progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        #progress.update(i+1)
        x = next(training_batch_generator)

        # train frame_predictor 
        mse, kld, mse_2, kld_2 = train(x)
        epoch_mse += mse
        epoch_kld += kld
        epoch_mse_2 += mse_2
        epoch_kld_2 += kld_2


    print('[%02d] mse loss: %.5f | kld loss: %.5f | mse_2 loss: %.5f | kld_2 loss: %.5f (%d)' % (epoch, 
            epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, 
            epoch_mse_2/opt.epoch_size, epoch_kld_2/opt.epoch_size, 
            epoch*opt.epoch_size*opt.batch_size))
    #progress.finish()
    #utils.clear_progressbar()
    writer.add_scalar('mse', epoch_mse/opt.epoch_size, epoch)
    writer.add_scalar('kld', epoch_kld/opt.epoch_size, epoch)
    writer.add_scalar('mse_2', epoch_mse_2/opt.epoch_size, epoch)
    writer.add_scalar('kld_2', epoch_kld_2/opt.epoch_size, epoch)
    
    # plot some stuff
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    prior.eval()
   
    pred_encoder.eval()
    pred_decoder.eval()
    posterior_2.eval()
    prior_2.eval()
    latent_encoder.eval()

    ssim, psnr = plot_rec(x, epoch, 'train')
    print("Train ssim: %.4f, psnr: %.4f at t=%d"%(ssim[-1], psnr[-1], ssim.shape[0]))
    x = next(testing_batch_generator)
    #plot(x, epoch)
    ssim, psnr = plot_rec(x, epoch, 'test')
    print("Test ssim: %.4f, psnr: %.4f at t=%d"%(ssim[-1], psnr[-1], ssim.shape[0]))

    # save the model
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'pred_encoder': pred_encoder,
        'pred_decoder': pred_decoder,
        'prior_2': prior_2,
        'posterior_2': posterior_2,
        'latent_encoder': latent_encoder,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)

    ''' 
    lr = opt.lr * (0.1 ** (epoch // 30))
    print("LR changed to: ", lr)
    for param_group in frame_predictor_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in posterior_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in prior_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in encoder_optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in decoder_optimizer.param_groups:
        param_group['lr'] = lr

    ''' 
