#Changes to HLM
#Fixed prior in L2
#No skip connections in L2 decoder

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


class Wrapper(nn.Module):
    def __init__(self):
        super(Wrapper, self).__init__()

    def forward(self, x, i, skip, hidden):
        hidden1, hidden2, hidden3, hidden4 = hidden

        h = self.encoder(x[i-1])
        h_target = self.encoder(x[i])[0]
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h = h[0]
        z_t, mu, logvar, hidden1 = self.posterior(h_target, hidden1)
        _, mu_p, logvar_p, hidden2 = self.prior(h, hidden2)
        h_pred, hidden3 = self.frame_predictor(torch.cat([h, z_t], 1), hidden3)
        x_pred = self.decoder([h_pred, skip])


        # 2nd level hierarchy
        if opt.joint == 0:
            x_pred = x_pred.detach()
            h_target = h_target.detach()
            skip = [_.detach() for _ in skip]
            h = h.detach()
            z_t = z_t.detach()


        x_pred_h = self.pred_encoder(x_pred)[0]

        z_t_2, mu_2, logvar_2, hidden4 = self.posterior_2(h_target + self.latent_encoder(z_t), hidden4)
        x_pred_2 = self.pred_decoder([x_pred_h + self.latent_encoder1(z_t_2), skip])

        return (x_pred, mu, logvar, mu_p, logvar_p, x_pred_2, mu_2, logvar_2, skip, (hidden1, hidden2, hidden3, hidden4))

model = Wrapper()

import models.lstm_parallel as lstm_models
#import models.original_lstm as lstm_models
'''
if opt.model_dir != '':
    model.frame_predictor = saved_model['frame_predictor']
    model.posterior = saved_model['posterior']
    model.prior = saved_model['prior']
else:
    model.frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    model.posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    model.prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
    model.frame_predictor.apply(utils.init_weights)
    model.posterior.apply(utils.init_weights)
    model.prior.apply(utils.init_weights)
'''

model.frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
model.posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
model.prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
model.frame_predictor.apply(utils.init_weights)
model.posterior.apply(utils.init_weights)
model.prior.apply(utils.init_weights)

model.posterior_2 = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
#prior_2 = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size)
#posterior_2.load_state_dict(posterior.state_dict())
#prior_2.load_state_dict(posterior.state_dict())
model.latent_encoder =  nn.Linear(opt.z_dim, opt.g_dim)
model.latent_encoder.apply(utils.init_weights)
model.latent_encoder1 =  nn.Linear(opt.z_dim, opt.g_dim)
model.latent_encoder1.apply(utils.init_weights)


if opt.model_dir != '':
    model.frame_predictor.load_state_dict(saved_model['frame_predictor'].state_dict())
    model.posterior.load_state_dict(saved_model['posterior'].state_dict())
    model.prior.load_state_dict(saved_model['prior'].state_dict())

if opt.load_all == 1:
    model.posterior_2 = saved_model['posterior_2']
    #prior_2 = saved_model['prior_2']
    model.latent_encoder = saved_model['latent_encoder']
    model.latent_encoder1 = saved_model['latent_encoder1']

if opt.model == 'highcap':
    import models.dcgan_64_high as modelll
elif opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as modelll 
    elif opt.image_width == 128:
        import models.dcgan_128 as modelll  
elif opt.model == 'vgg':
    if opt.image_width == 64:
        import models.vgg_64 as modelll
    elif opt.image_width == 128:
        import models.vgg_128 as modelll
else:
    raise ValueError('Unknown model: %s' % opt.model)
       
if opt.model_dir != '':
    model.decoder = saved_model['decoder']
    model.encoder = saved_model['encoder']
else:
    model.encoder = modelll.encoder(opt.g_dim, opt.channels)
    model.decoder = modelll.decoder(opt.g_dim, opt.channels)
    model.encoder.apply(utils.init_weights)
    model.decoder.apply(utils.init_weights)

if opt.load_all == 1:
    model.pred_encoder = saved_model['pred_encoder']
    model.pred_decoder = saved_model['pred_decoder']
else:
    model.pred_encoder = modelll.encoder(opt.g_dim, opt.channels)
    model.pred_decoder = modelll.decoder(opt.g_dim, opt.channels)

    #model.pred_decoder = modelll.decoder_noskip(opt.g_dim, opt.channels)

    #pred_encoder.apply(utils.init_weights)
    #pred_decoder.apply(utils.init_weights)

    #### Preload weights #####
    #pred_encoder.load_state_dict(encoder.state_dict())
    #pred_decoder.load_state_dict(decoder.state_dict())

#####print(model)

frame_predictor_optimizer = opt.optimizer(model.frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(model.posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(model.prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(model.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(model.decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

posterior_2_optimizer = opt.optimizer(model.posterior_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#prior_2_optimizer = opt.optimizer(prior_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
pred_decoder_optimizer = opt.optimizer(model.pred_decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
pred_encoder_optimizer = opt.optimizer(model.pred_encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
latent_encoder_optimizer = opt.optimizer(model.latent_encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
latent_encoder1_optimizer = opt.optimizer(model.latent_encoder1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


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


def kl_criterionL2(mu, logvar):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= opt.batch_size
  return KLD


model.cuda()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))


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
    #frame_predictor.hidden = frame_predictor.module.init_hidden()
    #posterior.hidden = posterior.module.init_hidden()
    #posterior_2.hidden = posterior_2.module.init_hidden()

    all_hidden0, all_hidden1, all_hidden2, _ = init_hidden()
    
    gen_seq, gt_seq, pred_seq = [], [], []
    gen_seq0, gt_seq0 = [x[0]], [x[0]]
    gen_seq.append(x[0])
    x_in = x[0]
    for i in range(1, opt.n_past+opt.n_future):
        h = model.module.encoder(x[i-1])
        h_target = model.module.encoder(x[i])
        if opt.last_frame_skip or i < opt.n_past:	
            h, skip = h
        else:
            h, _ = h
        h_target, _ = h_target
        h = h.detach()
        h_target = h_target.detach()
        z_t, _, _, all_hidden0 = model.module.posterior(h_target, all_hidden0)
        h_pred, all_hidden1 = model.module.frame_predictor(torch.cat([h, z_t], 1), all_hidden1)
        x_pred = model.module.decoder([h_pred, skip]).detach()

        x_pred_h = model.module.pred_encoder(x_pred)[0]

        x_pred_h.detach()
        z_t_2, mu_2, logvar_2, all_hidden2 = model.module.posterior_2(h_target + model.module.latent_encoder(z_t), all_hidden2)
        x_pred_2 = model.module.pred_decoder([x_pred_h + model.module.latent_encoder1(z_t_2), skip])

        if i < opt.n_past:
            gen_seq.append(x[i])
            gen_seq0.append(x[i])
            gt_seq0.append(x[i])
        else:
            gen_seq0.append(x_pred)
            gt_seq0.append(x[i])
            gen_seq.append(x_pred_2)
            gt_seq.append(x[i].data.cpu().numpy())
            pred_seq.append(x_pred_2.data.cpu().numpy())
    _, ssim, psnr = utils.eval_seq(gt_seq, pred_seq)
   
    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gt_seq0[t][i])
        to_plot.append(row)

        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq0[t][i])
        to_plot.append(row)

        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/%s_rec_%d.png' % (opt.log_dir, _type, epoch) 
    utils.save_tensors_image(fname, to_plot)
    
    return np.mean(ssim, axis=0), np.mean(psnr, axis=0)

def init_hidden():
    all_hidden = []
    for j in range(4): 
        hidden = []
        for i in range(2):
            hidden.append((Variable(torch.zeros(opt.batch_size, opt.rnn_size).cuda()),
                            Variable(torch.zeros(opt.batch_size, opt.rnn_size).cuda())))
        all_hidden.append(hidden)
    return all_hidden
# --------- training funtions ------------------------------------
def train(x):
    model.module.frame_predictor.zero_grad()
    model.module.posterior.zero_grad()
    model.module.prior.zero_grad()
    model.module.encoder.zero_grad()
    model.module.decoder.zero_grad()

    model.module.posterior_2.zero_grad()
    #model.module.prior_2.zero_grad()
    model.module.pred_encoder.zero_grad()
    model.module.pred_decoder.zero_grad()
    model.module.latent_encoder.zero_grad()
    model.module.latent_encoder1.zero_grad()

    # initialize the hidden state.
    '''
    model.module.frame_predictor.hidden = model.module.frame_predictor.init_hidden()
    model.module.posterior.hidden = model.module.posterior.init_hidden()
    model.module.prior.hidden = model.module.prior.init_hidden()
    model.module.posterior_2.hidden = model.module.posterior_2.init_hidden()
    #prior_2.hidden = prior_2.module.init_hidden()
    '''

    all_hidden = init_hidden()
    mse = 0
    kld = 0
    mse_2 = 0
    kld_2 = 0
    skip = None
    for i in range(1, opt.n_past+opt.n_future):
        #_, mu_p_2, logvar_p_2 = prior_2(h + latent_encoder(z_t))

        x_pred, mu, logvar, mu_p, logvar_p, x_pred_2, mu_2, logvar_2, skip, all_hidden = model(x, i, skip, all_hidden)

        mse += mse_criterion(x_pred, x[i])
        kld += kl_criterion(mu, logvar, mu_p, logvar_p)

        mse_2 += mse_criterion(x_pred_2, x[i])
        #kld_2 += kl_criterion(mu_2, logvar_2, mu_p_2, logvar_p_2)
        kld_2 += kl_criterionL2(mu_2, logvar_2)


    loss = opt.rec1*mse + kld*opt.beta + mse_2 + kld_2*opt.beta2
    loss.backward()

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    decoder_optimizer.step()
    encoder_optimizer.step()

    posterior_2_optimizer.step()
    pred_decoder_optimizer.step()
    pred_encoder_optimizer.step()
    latent_encoder_optimizer.step()
    latent_encoder1_optimizer.step()


    _losses = (
           mse.data.cpu().numpy()/(opt.n_past+opt.n_future),
           kld.data.cpu().numpy()/(opt.n_past+opt.n_future),
           mse_2.data.cpu().numpy()/(opt.n_past+opt.n_future),
           kld_2.data.cpu().numpy()/(opt.n_past+opt.n_future),
           )
    return _losses

    

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    model.module.frame_predictor.train()
    model.module.posterior.train()
    model.module.prior.train()
    model.module.encoder.train()
    model.module.decoder.train()

    model.module.posterior_2.train()
    #model.module.prior_2.train()
    model.module.pred_encoder.train()
    model.module.pred_decoder.train()
    model.module.latent_encoder.train()
    model.module.latent_encoder1.train()

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
    model.module.frame_predictor.eval()
    model.module.encoder.eval()
    model.module.decoder.eval()
    model.module.posterior.eval()
    model.module.prior.eval()
   
    model.module.pred_encoder.eval()
    model.module.pred_decoder.eval()
    model.module.posterior_2.eval()
    #model.module.prior_2.eval()
    model.module.latent_encoder.eval()
    model.module.latent_encoder1.eval()

    ssim, psnr = plot_rec(x, epoch, 'train')
    print("Train ssim: %.4f, psnr: %.4f at t=%d"%(ssim[-1], psnr[-1], ssim.shape[0]))
    x = next(testing_batch_generator)
    #plot(x, epoch)
    ssim, psnr = plot_rec(x, epoch, 'test')
    print("Test ssim: %.4f, psnr: %.4f at t=%d"%(ssim[-1], psnr[-1], ssim.shape[0]))

    # save the model
    torch.save({
        'encoder': model.module.encoder,
        'decoder': model.module.decoder,
        'frame_predictor': model.module.frame_predictor,
        'posterior': model.module.posterior,
        'prior': model.module.prior,
        'pred_encoder': model.module.pred_encoder,
        'pred_decoder': model.module.pred_decoder,
        #'prior_2': model.module.prior_2,
        'posterior_2': model.module.posterior_2,
        'latent_encoder': model.module.latent_encoder,
        'latent_encoder1': model.module.latent_encoder1,
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
