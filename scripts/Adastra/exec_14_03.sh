#!/bin/bash
#SBATCH --job-name=box_14_03
#SBATCH --output=14_03.out
#SBATCH --error=14_03.err
#SBATCH --time=04:00:00
#SBATCH --account=cad16409

#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --hint=nomultithread

module purge

source ~/.bashrc
mamba activate chrono

cd /lus/work/CT10/cad16409/lgandeel/RSA_reconstruction/Method/ChronoRoot/

srun python chronoRoot.py --imgpath /lus/work/CT10/cad16409/lgandeel/RSA_reconstruction/Method/input_img/rpi14_2020-03-12_17-00/ --segpath /lus/work/CT10/cad16409/lgandeel/RSA_reconstruction/Method/segs/DICE/rpi14_2020-03-12_17-00/ --savepath /lus/scratch/CT10/cad16409/lgandeel/output_DICE/rpi14_2020-03-12_17-00/ > /lus/scratch/CT10/cad16409/lgandeel/output_DICE/14_03.log 2>&1
