#!/bin/bash
#SBATCH --job-name="15_01b"
#SBATCH -A vey@h100
#SBATCH --partition=gpu_p6
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/vey/unq35lq/out/rpi15_01b.out
#SBATCH --error=/lustre/fswork/projects/rech/vey/unq35lq/err/rpi15_01b.err

module load arch/h100

source ~/.bashrc
mamba activate chrono
cd /lustre/fswork/projects/rech/vey/unq35lq/RSA_reconstruction/Method/ChronoRoot_BCE

srun python3 ./segmentEnsembleCustom.py --input_dir /lustre/fswork/projects/rech/vey/unq35lq/ChronoRoot_expe/input_img/rpi15_2020-01-08_17-24/ --output_dir /lustre/fswork/projects/rech/vey/unq35lq/ChronoRoot_expe/output_chronoroo_bce/