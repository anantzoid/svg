#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=bair
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
#SBATCH --gres=gpu:2 -c2
#SBATCH --mail-type=END
#SBATCH --mail-user=ag4508@nyu.edu

module purge
#module load tensorflow/python3.6/1.5.0
source activate nlp

#python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root data/mnist --log_dir logs --name mnist_lp_original
#python generate_svg_lp.py --model_path pretrain_models/svglp_bair.pth --log_dir pretrained_lp --data_root /scratch/ag4508/svg --dataset bair --z_dim 10 --g_dim 128
#python data/convert_bair.py  --data_dir /beegfs/ag4508/svg

#3/9
#python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root data/mnist --log_dir logs --name launchtest --data_threads 10 --multi 1 --niter 10 --epoch_size 50 --n_past 3 --n_future 3
#python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root data/mnist --log_dir logs --name bilstm1 --data_threads 10 --multi 1
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 1 --name bairbilstm1

#3/10
#grad clipping
#python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm2
python train_svg_lp.py --dataset bair --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --data_root /beegfs/ag4508/svg/ --log_dir logs --data_threads 10 --multi 0 --name bairbilstm3 --noskip 1



