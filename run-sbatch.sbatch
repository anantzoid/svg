#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=pw10
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:p100 -c1
#SBATCH --mail-type=END
#SBATCH --mail-user=ag4508@nyu.edu

module purge
#module load tensorflow/python2.7/20170707
source activate env

#python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root data/mnist --log_dir logs --name mnist_lp_original
#python generate_svg_lp.py --model_path pretrain_models/svglp_bair.pth --log_dir pretrained_lp --data_root /scratch/ag4508/svg --dataset bair --z_dim 10 --g_dim 128
#python data/convert_bair.py  --data_dir /beegfs/ag4508/svg

#3/9
#python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root data/mnist --log_dir logs --name launchtest --data_threads 10 --multi 1 --niter 10 --epoch_size 50 --n_past 3 --n_future 3
#python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root data/mnist --log_dir logs --name bilstm1 --data_threads 10 --multi 1
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 1 --name bairbilstm1

#3/10
### grad clipping->fixes nan in bilistm
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm2
### Cause of region based bluriness speculated to be skip connection(since past+future gets averaged), so no skip_conn:
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm3 --noskip 1

#3/11
### No skip takes time to learn to reconstruct, so add small weightage to skip conn
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm4 --noskip 0 --skip_weight 0.01
### Another strategy->use skip_conn in last 2 layers only
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --noskip 0 --skip_part 1 --name bairbilstm5

#3/12
#eval mode original
#python generate_svg_lp.py --log_dir gen/bilstm2 --model_path logs/bair/model\=dcgan64x64-rnn_size\=256-rnn_layers\=2-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000bairbilstm2/model.pth --num_threads 5 --nsample 20 

#3/13
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm6 --lstm_singledir 1

#3/14
# Gen for prev. exps.
#python generate_svg_lp.py --log_dir gen/bilstm3 --model_path logs/bair/model\=dcgan64x64-rnn_size\=256-rnn_layers\=2-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000bairbilstm3/model.pth --num_threads 5 --nsample 20 --noskip 1
#python generate_svg_lp.py --log_dir gen/bilstm4 --model_path logs/bair/model\=dcgan64x64-rnn_size\=256-rnn_layers\=2-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000bairbilstm4/model.pth --num_threads 5 --nsample 20 --noskip 0
#python generate_svg_lp.py --log_dir gen/bilstm4 --model_path logs/bair/model\=dcgan64x64-rnn_size\=256-rnn_layers\=2-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000bairbilstm4/model.pth --num_threads 5 --nsample 20 --noskip 0
#python generate_svg_lp.py --log_dir gen/bilstm6 --model_path logs/bair/model\=dcgan64x64-rnn_size\=256-rnn_layers\=2-n_past\=2-n_future\=10-lr\=0.0020-g_dim\=128-z_dim\=64-last_frame_skip\=False-beta\=0.0001000bairbilstm6/model.pth --num_threads 5 --nsample 20

#3/19
#0 kl loss
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm7 --lstm_singledir 1 --beta 0 --lstm_singledir_posterior 1
#decoder mulit-update --- didn't run
###python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm8 --lstm_singledir 1 --beta 0.0001 --lstm_singledir_posterior 1 --decoder_updates 5 --batch_size 50
#python mdecoder_train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm9 --lstm_singledir 1 --beta 0.0001 --lstm_singledir_posterior 1 --decoder_updates 10
# high capacity 0 kl loss
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm10 --lstm_singledir 1 --beta 0 --lstm_singledir_posterior 1 --model highcap

#multi-scale decoder loss
#default model
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm11 --lstm_singledir 1 --beta 0.0001 --lstm_singledir_posterior 1 
# kl0
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm12 --lstm_singledir 1 --beta 0 --lstm_singledir_posterior 1 


#3/20
##super highcap model with lr decay & old lstm
#python original_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name bairbilstm13 --rnn_size 512 --lr 0.01 --model highcap --model_dir logs/bair/model=highcap64x64-rnn_size=512-rnn_layers=2-n_past=2-n_future=10-lr=0.0100-g_dim=256-z_dim=128-last_frame_skip=False-beta=0.0001000bairbilstm13  
##run above with kl 0
#python original_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name bairbilstm14 --rnn_size 512 --lr 0.01 --model highcap

#Identity function x->enc->posterior->fp->dec->x
#python identity.py --dataset bair --g_dim 512 --z_dim 256 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name identity1 --rnn_size 512 --lr 0.01 --model highcap --noskip 1 --epoch_size 100 --niter 50
#above continued
#python identity.py --dataset bair --g_dim 512 --z_dim 256 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name identity1 --rnn_size 512 --lr 0.0001 --model highcap --noskip 1 --epoch_size 600 --niter 100 --model_dir logs/bair/model=highcap64x64-rnn_size=512-rnn_layers=2-n_past=2-n_future=10-lr=0.0100-g_dim=512-z_dim=256-last_frame_skip=False-beta=0.0000000identity1/continued

# 3/22
#python identity.py --dataset bair --g_dim 512 --z_dim 256 --beta 0 --n_past 4 --n_future 4 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name identity2 --rnn_size 512 --lr 0.01 --model highcap --noskip 1 --epoch_size 600 --niter 100 

####Ignore highcap0 and 1####
#original highcap kl0 lrscheduling
#python original_lp.py --dataset bair --g_dim 512 --z_dim 256 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name highcap0 --rnn_size 512 --lr 0.01 --model highcap --niter 100
#original highcap kl0 
#python original_lp.py --dataset bair --g_dim 512 --z_dim 256 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name highcap1 --rnn_size 512 --model highcap --niter 100


#original kl0
#python original_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name orig0  --niter 100

#identity default lr
#python identity.py --dataset bair --g_dim 512 --z_dim 256 --beta 0 --n_past 4 --n_future 4 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name identity3 --rnn_size 512  --model highcap --noskip 1 --epoch_size 600 --niter 100 

#continuing train_svg_lp kl0
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 1 --name bairbilstm7cont --lstm_singledir 1 --beta 0 --lstm_singledir_posterior 1 --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000bairbilstm7 --lr 0.0003


# 3/23
# Continuing identity with lesser lr
#python identity.py --dataset bair --g_dim 512 --z_dim 256 --beta 0 --n_past 4 --n_future 4 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name identity3cont --rnn_size 512  --model highcap --noskip 1 --epoch_size 600 --niter 100 --model_dir logs/bair/model=highcap64x64-rnn_size=512-rnn_layers=2-n_past=4-n_future=4-lr=0.0020-g_dim=512-z_dim=256-last_frame_skip=False-beta=0.0000000identity3 --lr 0.0003 

#continue original highcap kl0
#python original_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name bairbilstm14cont --rnn_size 512 --lr 0.0003 --model highcap --model_dir logs/bair/model=highcap64x64-rnn_size=512-rnn_layers=2-n_past=2-n_future=10-lr=0.0100-g_dim=256-z_dim=128-last_frame_skip=False-beta=0.0000000bairbilstm14
# continue msloss for kl0
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm12 --lstm_singledir 1 --beta 0 --lstm_singledir_posterior 1 --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000bairbilstm12 
#kl 0 high cap model
#python train_svg_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name highcapmsloss --rnn_size 512 --lr 0.0003 --model highcap --msloss 1 

# 3/24
#continuing bilstm7cont
## Did this RUN??
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 1 --name bairbilstm7contcont --lstm_singledir 1 --beta 0 --lstm_singledir_posterior 1 --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000bairbilstm7/continued --lr 0.0001

# 3/26
#continuing good performing highcapmsloss
#python train_svg_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name highcapmsloss --rnn_size 512 --lr 0.0003 --model highcap --msloss 1 --model_dir logs/bair/model=highcap64x64-rnn_size=512-rnn_layers=2-n_past=2-n_future=10-lr=0.0003-g_dim=256-z_dim=128-last_frame_skip=False-beta=0.0000000highcapmsloss  

#highcap msloss klset
#python train_svg_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name highcapmslosslp --rnn_size 512 --lr 0.0003 --model highcap --msloss 1 

#highcap msloss uniLSTM model kl0
#python train_svg_lp.py --dataset bair --g_dim 256 --z_dim 128 --lstm_singledir 1 --beta 0 --lstm_singledir_posterior 1 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name highcapmslossunilstm --rnn_size 512 --lr 0.0003 --model highcap --msloss 1 

#highcap msloss uniLSTM model kl set
#python train_svg_lp.py --dataset bair --g_dim 256 --z_dim 128 --lstm_singledir 1 --beta 0.0001 --lstm_singledir_posterior 1 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name highcapmslossunilstmkl --rnn_size 512 --lr 0.0003 --model highcap --msloss 1 

# 3/28
#msloss on unilstm (original), since unilstm on train_sv_lp is giving unexpected higher error
#python original_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name orighighcapmsloss --rnn_size 512 --lr 0.0003 --model highcap --msloss 1 --batch_size 128 

#python original_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name orighighcapmslosskl --rnn_size 512 --lr 0.0003 --model highcap --msloss 1 --batch_size 128

# Orignal highcap kl0 with smaller lr from start --contrary to bairbilstm14
#python original_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name orighighcap --rnn_size 512 --lr 0.0003 --model highcap --msloss 0 --batch_size 128
#above with klset
#python original_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name orighighcapkl --rnn_size 512 --lr 0.0003 --model highcap --msloss 0 --batch_size 128

#identiy with good lr init
#python identity.py --dataset bair --g_dim 512 --z_dim 256 --beta 0 --n_past 4 --n_future 4 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name identity4 --rnn_size 512  --model highcap --noskip 1  --lr 0.0003
#python identity.py --dataset bair --g_dim 256 --z_dim 256 --beta 0 --n_past 4 --n_future 4 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name identity5 --rnn_size 512  --model highcap --noskip 1  --lr 0.0003

# 3/31
#working command
#python original_lp.py --dataset bair --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 2 --name temp --msloss 0 --batch_size 16 --epoch_size 10 --niter 1

# 1/4
#python original_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs2 --plot_dir plots2 --data_threads 10 --name pw1  --batch_size 128 --pw 1
#python original_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs2 --plot_dir plots2 --data_threads 10 --name pw5  --batch_size 128 --pw 5
python original_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs2 --plot_dir plots2 --data_threads 10 --name pw10 --batch_size 128 --pw 10
