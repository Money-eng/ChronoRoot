#!/bin/bash
#SBATCH --job-name="15_03d"
#SBATCH -A vey@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/vey/unq35lq/out/rpi15_03d.out
#SBATCH --error=/lustre/fswork/projects/rech/vey/unq35lq/err/rpi15_03d.err

module load arch/h100

source ~/.bashrc
mamba activate chrono
cd /lustre/fswork/projects/rech/vey/unq35lq/RSA_reconstruction/Method/ChronoRoot_DICE

srun python3 ./segmentEnsembleCustom.py --input_dir /lustre/fsn1/projects/rech/vey/unq35lq/expe_rpi15/input_img/rpi15_2020-03-12_17-01/ --output_dir /lustre/fsn1/projects/rech/vey/unq35lq/expe_rpi15/output_inference_dice/