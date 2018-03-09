#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=bair
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=20GB
#SBATCH --gres=gpu:1 -c1
#SBATCH --mail-type=END
#SBATCH --mail-user=ag4508@nyu.edu

module purge
#module load tensorflow/python3.6/1.5.0
source activate nlp

python train_svg_lp.py --dataset smmnist --num_digits 2 --g_dim 128 --z_dim 10 --beta 0.0001 --data_root data/mnist --log_dir logs --name mnist_lp_original
#python generate_svg_lp.py --model_path pretrain_models/svglp_bair.pth --log_dir pretrained_lp --data_root /scratch/ag4508/svg --dataset bair --z_dim 10 --g_dim 128
#python data/convert_bair.py  --data_dir /beegfs/ag4508/svg



