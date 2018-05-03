#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=epic1
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


#4/16
#python original_lp_mask.py --g_dim 128 --z_dim 64 --beta 0 --channels 3 --dataset epic --mse 1  --batch_size 64 --name epic0 --data_threads 10 --log_dir logs

#python original_lp_mask.py --g_dim 128 --z_dim 64 --beta 0.0001 --channels 3 --dataset epic --mse 1  --batch_size 100 --name epic1 --data_threads 10 --log_dir logs --skip_frames 3
#python original_lp_mask.py --g_dim 128 --z_dim 64 --beta 0.0001 --channels 3 --dataset epic --mse 1  --batch_size 100 --name epic2 --data_threads 10 --log_dir logs --skip_frames 3 --model_dir logs/epic/model=dcgan64x64-rnn_size=256-rnn_layers=2-n_past=5-n_future=10-lr=0.0020-g_dim=128-z_dim=64-last_frame_skip=False-beta=0.0001000epic1 

#python parallel_original_lp_mask.py --g_dim 256 --z_dim 128  --beta 0.0001 --channels 3 --dataset epic --mse 1  --batch_size 16 --name newepic --data_threads 10 --log_dir logs --skip_frames 3 --epoch_size 300 --image_width 128 --model vgg --data_root /beegfs/ag4508/ 

#continued midway from 46 since it seemed to overfit, increased epoch_size 
python parallel_original_lp_mask.py --g_dim 256 --z_dim 128  --beta 0.0001 --channels 3 --dataset epic --mse 1  --batch_size 16 --name newepic1 --data_threads 10 --log_dir logs --skip_frames 3 --epoch_size 600 --image_width 128 --model vgg --data_root /beegfs/ag4508/ --model_dir logs/epic/model=vgg128x128-rnn_size=256-rnn_layers=2-n_past=5-n_future=10-lr=0.0020-g_dim=256-z_dim=128-last_frame_skip=False-beta=0.0001000newepic 





