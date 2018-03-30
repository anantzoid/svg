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
parser.add_argument('--multi', type=int, default=0, help='Use mulitple gpus')
parser.add_argument('--noskip', type=int, default=0, help='Dont use skip connections (possible cause of blurring)')
parser.add_argument('--skip_part', type=int, default=0, help='Only use last 2 layers of skip connections')
parser.add_argument('--skip_weight', type=float, default=1.0, help='Trying weight factor on skip connection instead of complete removal')
parser.add_argument('--lstm_singledir', type=int, default=0, help='single-direction lstm for frame_preditor & prior')
parser.add_argument('--lstm_singledir_posterior', type=int, default=0, help='BiLSTM posterior')
parser.add_argument('--decoder_updates', type=int, default=0, help='')
parser.add_argument('--msloss', type=int, default=0, help='Use mulitple gpus')



opt = parser.parse_args()
if opt.model_dir != '':
    # load model and continue training from checkpoint
    saved_model = torch.load('%s/model.pth' % opt.model_dir)
    optimizer = opt.optimizer
    model_dir = opt.model_dir
    opt = saved_model['opt']
    opt.optimizer = optimizer
    opt.model_dir = model_dir
    opt.log_dir = '%s/continued' % opt.log_dir
else:
    name = 'model=%s%dx%d-rnn_size=%d-rnn_layers=%d-n_past=%d-n_future=%d-lr=%.4f-g_dim=%d-z_dim=%d-last_frame_skip=%s-beta=%.7f%s' % (opt.model, opt.image_width, opt.image_width, opt.rnn_size, opt.rnn_layers, opt.n_past, opt.n_future, opt.lr, opt.g_dim, opt.z_dim, opt.last_frame_skip, opt.beta, opt.name)
    if opt.dataset == 'smmnist':
        opt.log_dir = '%s/%s-%d/%s' % (opt.log_dir, opt.dataset, opt.num_digits, name)
    else:
        opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.dataset, name)

os.makedirs('%s/gen/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/plots/' % opt.log_dir, exist_ok=True)
opt.max_step = opt.n_past+opt.n_future
writer = SummaryWriter(log_dir=os.path.join('plots', opt.name))

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor

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
if opt.model_dir != '':
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    prior = saved_model['prior']
else:
    frame_predictor = lstm_models.lstm(opt.g_dim+opt.z_dim, opt.g_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size, not(opt.lstm_singledir))
    posterior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size, not(opt.lstm_singledir_posterior))
    prior = lstm_models.gaussian_lstm(opt.g_dim, opt.z_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size, not(opt.lstm_singledir))
    frame_predictor.apply(utils.init_weights)
    posterior.apply(utils.init_weights)
    prior.apply(utils.init_weights)

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
    if opt.noskip:
        decoder = model.decoder_noskip(opt.g_dim, opt.channels)
    else:
        if opt.skip_part:
            decoder = model.decoder_skippart(opt.g_dim, opt.channels)
        else:
            decoder = model.decoder(opt.g_dim, opt.channels)

    encoder.apply(utils.init_weights)
    decoder.apply(utils.init_weights)



frame_predictor_optimizer = opt.optimizer(frame_predictor.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
posterior_optimizer = opt.optimizer(posterior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
prior_optimizer = opt.optimizer(prior.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
encoder_optimizer = opt.optimizer(encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
decoder_optimizer = opt.optimizer(decoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

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

if opt.multi == 1:
    encoder = torch.nn.DataParallel(encoder, device_ids=range(torch.cuda.device_count()))
    decoder = torch.nn.DataParallel(decoder, device_ids=range(torch.cuda.device_count()))
    frame_predictor = torch.nn.DataParallel(frame_predictor, device_ids=range(torch.cuda.device_count()))
    posterior = torch.nn.DataParallel(posterior, device_ids=range(torch.cuda.device_count()))
    prior = torch.nn.DataParallel(prior, device_ids=range(torch.cuda.device_count()))


if opt.msloss:
    multiscale_loss = utils.MultiScaleLoss()
    multiscale_loss.cuda()

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


def plot(x, epoch):
    nsample = 20 
    gen_seq = [[] for i in range(nsample)]
    gt_seq = [x[i] for i in range(len(x))]

    for s in range(nsample):
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        gen_seq[s].append(x[0])
        x_in = x[0]
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i < opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])
                h_target = h_target[0].detach()
                z_t, _, _ = posterior(h_target)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                gen_seq[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq[s].append(x_in)

    to_plot = []
    gifs = [ [] for t in range(opt.n_eval) ]
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = [] 
        for t in range(opt.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        # best sequence
        min_mse = 1e7
        for s in range(nsample):
            mse = 0
            for t in range(opt.n_eval):
                mse +=  torch.sum( (gt_seq[t][i].data.cpu() - gen_seq[s][t][i].data.cpu())**2 )
            if mse < min_mse:
                min_mse = mse
                min_idx = s

        s_list = [min_idx, 
                  np.random.randint(nsample), 
                  np.random.randint(nsample), 
                  np.random.randint(nsample), 
                  np.random.randint(nsample)]
        for ss in range(len(s_list)):
            s = s_list[ss]
            row = []
            for t in range(opt.n_eval):
                row.append(gen_seq[s][t][i]) 
            to_plot.append(row)
        for t in range(opt.n_eval):
            row = []
            row.append(gt_seq[t][i])
            for ss in range(len(s_list)):
                s = s_list[ss]
                row.append(gen_seq[s][t][i])
            gifs[t].append(row)

    fname = '%s/gen/sample_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen/sample_%d.gif' % (opt.log_dir, epoch) 
    utils.save_gif(fname, gifs)


def plot_rec_new(x, epoch):
    if opt.multi:
        frame_predictor.hidden = frame_predictor.module.init_hidden()
        posterior.hidden = posterior.module.init_hidden()
    else:
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()

    gen_seq = []
    gen_seq.append(x[0])
    x_in = x[0]
    h_encoded, posteriors = [], []  
    for i in range(0, opt.n_past+opt.n_future):
        h_encoded.append(encoder(x[i]))

    h = torch.stack([h_encoded[i][0] for i in range(0, opt.n_past+opt.n_future-1)])
    h_target = torch.stack([h_encoded[i][0] for i in range(1, opt.n_past+opt.n_future)])
    posteriors = posterior(h)
    pred_ip = torch.stack([torch.cat([h[i], posteriors[0][i]], 1) for i in range(opt.n_past+opt.n_future-1)])
    h_preds = frame_predictor(pred_ip)
    for i in range(opt.n_past+opt.n_future-1):
        if i < opt.n_past-1:
            skip = h_encoded[i][1]
            gen_seq.append(x[i+1])
        else:
            gen_seq.append(decoder([h_preds[i], [_*opt.skip_weight for _ in skip]])[0])

    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/new_rec_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)
    

def plot_rec(x, epoch):
    frame_predictor.hidden = frame_predictor.init_hidden()
    posterior.hidden = posterior.init_hidden()
    gen_seq = []
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
            frame_predictor(torch.cat([h, z_t], 1)) 
            gen_seq.append(x[i])
        else:
            h_pred = frame_predictor(torch.cat([h, z_t], 1))
            x_pred = decoder([h_pred, skip]).detach()
            gen_seq.append(x_pred)
   
    to_plot = []
    nrow = min(opt.batch_size, 10)
    for i in range(nrow):
        row = []
        for t in range(opt.n_past+opt.n_future):
            row.append(gen_seq[t][i]) 
        to_plot.append(row)
    fname = '%s/gen/rec_%d.png' % (opt.log_dir, epoch) 
    utils.save_tensors_image(fname, to_plot)


# --------- training funtions ------------------------------------
def train(x):
    frame_predictor.zero_grad()
    posterior.zero_grad()
    prior.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # initialize the hidden state.
    if opt.multi:
        frame_predictor.hidden = frame_predictor.module.init_hidden()
        posterior.hidden = posterior.module.init_hidden()
        prior.hidden = prior.module.init_hidden()
    else:
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()

    mse = 0
    kld = 0

    h_encoded, priors, posteriors = [], [], []     
    for i in range(0, opt.n_past+opt.n_future):
        h_encoded.append(encoder(x[i]))
    
    h = torch.stack([h_encoded[i][0] for i in range(0, opt.n_past+opt.n_future-1)])
    h_target = torch.stack([h_encoded[i][0] for i in range(1, opt.n_past+opt.n_future)])
    posteriors = posterior(h)
    priors = prior(h_target)
   
    pred_ip = torch.stack([torch.cat([h[i], posteriors[0][i]], 1) for i in range(opt.n_past+opt.n_future-1)])
    h_preds = frame_predictor(pred_ip)

    for i in range(opt.n_past+opt.n_future-1):
        if i < opt.n_past-1:
            skip = h_encoded[i][1]
        _t, _scales = decoder([h_preds[i], [_*opt.skip_weight for _ in skip]])
        if opt.msloss:
            mse += mse_criterion(_t, x[i+1])
            mse += multiscale_loss(_scales, x[i+1])
        else:
            mse += mse_criterion(_t, x[i+1])

        kld += kl_criterion(posteriors[1][i], posteriors[2][i], priors[1][i], priors[2][i])


    #####Decoder Multiple updates#######
    '''
    _mse = 0
    h_preds = [_.detach() for _ in h_preds]
    skip = [_.detach() for _ in skip]
    for nu in range(opt.decoder_updates):
        x_pred = []
        for i in range(opt.n_past+opt.n_future-1):
            if i < opt.n_past-1:
                skip = h_encoded[i][1]
            x_pred.append(decoder([h_preds[i], skip]))
            _mse += mse_criterion(x_pred[-1], x[i+1])
    '''
    ##################################

    loss = mse + kld*opt.beta# + _mse
    loss.backward()
    torch.nn.utils.clip_grad_norm(frame_predictor.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm(posterior.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm(prior.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm(encoder.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), 1.0)

    frame_predictor_optimizer.step()
    posterior_optimizer.step()
    prior_optimizer.step()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return mse.data.cpu().numpy()/(opt.n_past+opt.n_future), kld.data.cpu().numpy()/(opt.n_future+opt.n_past), 0# _mse.data.cpu().numpy()/((opt.n_future+opt.n_past)*opt.decoder_updates)

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    frame_predictor.train()
    posterior.train()
    prior.train()
    encoder.train()
    decoder.train()
    epoch_mse = 0
    epoch_kld = 0
    #progress = progressbar.ProgressBar(max_value=opt.epoch_size).start()
    for i in range(opt.epoch_size):
        #progress.update(i+1)
        x = next(training_batch_generator)

        # train frame_predictor 
        mse, kld, _mse = train(x)
        epoch_mse += mse
        epoch_kld += kld

    #progress.finish()
    #utils.clear_progressbar()

    print('[%02d] mse loss: %.5f | kld loss: %.5f (%d), %.5f' % (epoch, epoch_mse/opt.epoch_size, epoch_kld/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size, _mse))
    writer.add_scalar('mse', epoch_mse/opt.epoch_size, epoch)
    writer.add_scalar('kld', epoch_kld/opt.epoch_size, epoch)
    #writer.add_scalar('_mse', _mse, epoch)
    #writer.add_scalars('train/losses', {'kld':epoch_kld/opt.epoch_size, 'mse':epoch_mse/opt.epoch_size}, epoch)

    # plot some stuff
    frame_predictor.eval()
    encoder.eval()
    decoder.eval()
    posterior.eval()
    prior.eval()
    
    x = next(testing_batch_generator)
    #plot(x, epoch)
    #plot_rec(x, epoch)
    plot_rec_new(x, epoch)
    #exit()

    # save the model
    torch.save({
        'encoder': encoder,
        'decoder': decoder,
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'opt': opt},
        '%s/model.pth' % opt.log_dir)
    try:
        torch.save({
            'encoder': encoder.module,
            'decoder': decoder.module,
            'frame_predictor': frame_predictor.module,
            'posterior': posterior.module,
            'prior': prior.module,
            'opt': opt},
            '%s/model_parallel.pth' % opt.log_dir)
    except:
        pass
if epoch % 10 == 0:
        print('log dir: %s' % opt.log_dir)
        

