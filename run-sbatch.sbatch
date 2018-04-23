#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=hlmnew3
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --mem=250GB
#SBATCH --gres=gpu:4 -c4
#SBATCH --mail-type=END
#SBATCH --mail-user=ag4508@nyu.edu

module purge
#module load tensorflow/python3.6/1.5.0
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

#3/20
##super highcap model with lr decay & old lstm
#python original_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name bairbilstm13 --rnn_size 512 --lr 0.01 --model highcap
##run above with kl 0
#python original_lp.py --dataset bair --g_dim 256 --z_dim 128 --beta 0 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --name bairbilstm14 --rnn_size 512 --lr 0.01 --model highcap

#python generate_svg_lp.py --data_root /beegfs/ag4508/svg/ --dataset bair --batch_size 50 --log_dir logs2/orig_lap1 --model_path logs2/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000orig_lap1/model.pth

# 4/10
# HLM
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 1 --name hlm0 --model_dir pretrain_models --beta2 0 --rec1 0 --beta 0 --joint 0
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 1 --name hlm1 --model_dir pretrain_models --beta2 0.0001 --rec1 0 --beta 0 --joint 0
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 1 --name hlm2 --model_dir pretrain_models --beta2 0.0001 --rec1 1 --beta 0.0001 --joint 1
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 1 --name hlm3 --model_dir pretrain_models --beta2 0.0001 --rec1 0.1 --beta 0.0001 --joint 1

# 4/11
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm4  --beta2 0.0001 --rec1 1 --beta 0.0001 --joint 1


# 4/13
# encoder, decoder is same in L2 heirarchy
# dec(h +lin(z)
#python hlm1.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm5 --model_dir /scratch/ag4508/svg/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000bairold1 --beta2 0 --rec1 0 --beta 0 --joint 0 --lr 0.0002
#L1Loss variant
#python hlm1.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm6 --model_dir /scratch/ag4508/svg/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000bairold1 --beta2 0 --rec1 0 --beta 0 --joint 0 --l1 1

#4/14
#L2 models initialized from L1
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm7 --model_dir /scratch/ag4508/svg/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000bairold1  --beta2 0 --rec1 0 --beta 0 --joint 0 --lr 0.0002

#continuining hlm6 with lower LR (learnt Lr decay is needed in L1Loss)
#python hlm1.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm6cont --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000hlm6 --beta2 0 --rec1 0 --beta 0 --joint 0 --l1 1 --lr 0.0001 --load_all 1

#residual error, also skip conn coming from L2 enc
#python hlm2.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm8 --model_dir /scratch/ag4508/svg/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000bairold1  --beta2 0 --rec1 0 --beta 0 --joint 0 --lr 0.0002

#skip connection in L2 decoder comes from L2 encoder (rather than L1 encoder)
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm9 --model_dir /scratch/ag4508/svg/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000bairold1  --beta2 0 --rec1 0 --beta 0 --joint 0 --lr 0.0002 --new_skip 1

#python hlm1.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm5 --model_dir /scratch/ag4508/svg/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000bairold1 --beta2 0 --rec1 0 --beta 0 --joint 0 --lr 0.0002 --new_skip 1

# residual with higher lr
#python hlm2.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm8_l1 --model_dir /scratch/ag4508/svg/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000bairold1  --beta2 0 --rec1 0 --beta 0 --joint 0 --lr 0.002

#4/16
#continuing l2 init from l1 with skip from l1
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm7cont --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0002-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000hlm7  --beta2 0 --rec1 0 --beta 0 --joint 0 --lr 0.0002 --load_all 1

#continuining residual with le 2e-3->gave lowest mse in all hlms
#python hlm2.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm8_l1cont --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000hlm8_l1  --beta2 0 --rec1 0 --beta 0 --joint 0 --lr 0.002

#continued hlm7cont from some epoch & enable other loss objectives too
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm7contcont --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0002-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000hlm7cont  --beta2 0.0001 --rec1 1 --beta 0.0001 --joint 0 --lr 0.0002 --load_all 1

#continue hlm8_l1 as loss keeps decreasing
#hlm8_l1cont trained again from scratch it seems
#python hlm2.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm8_l1contcont --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000hlm8_l1 --beta2 0 --rec1 0 --beta 0 --joint 0 --lr 0.002 --load_all 1


#continue hlm7_l1 as loss keeps decreasing
#stopped: wrong name
####python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm7contcont --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0002-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0000000hlm7cont  --beta2 0 --rec1 0 --beta 0 --joint 0 --lr 0.0002 --load_all 1

##python generate_svg_lphlm.py --model_path logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0002-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000hlm7contcont/model.pth --log_dir logs2/hlm7contcont --nsample 20 --N 20 --n_future 15 --data_root /beegfs/ag4508/svg/ --dataset bair


#continue hlm7contcont with higher lr
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm7contcont1 --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0002-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000hlm7contcont  --beta2 0.0001 --rec1 1 --beta 0.0001 --joint 0 --lr 0.0002 --load_all 1

#above with joint enabled
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm7contcont2 --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0002-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000hlm7contcont  --beta2 0.0001 --rec1 1 --beta 0.0001 --joint 1 --lr 0.0002 --load_all 1

#above with lower lr. above one decreases loss further, so testing if loweing lr will take it further
#python hlm.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name hlm7contcont3 --model_dir logs/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0002-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000hlm7contcont  --beta2 0.0001 --rec1 1 --beta 0.0001 --joint 1 --lr 0.00005 --load_all 1

#4/22
#python hlm3.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name newhlm3 --model_dir /scratch/ag4508/svg/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000bairold1  --beta 0 --rec1 0 --beta2 0.0001 --joint 0 --batch_size 400 
#Added skip connection from L1 enc to L2 decoder
python hlm3.py --dataset bair --g_dim 128 --z_dim 64 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs  --data_threads 10 --name newhlm4 --model_dir /scratch/ag4508/svg/bair/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000bairold1  --beta 0 --rec1 0 --beta2 0.0001 --joint 0 --batch_size 400 



