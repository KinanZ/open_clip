#PBS -N clip_no_0_no_dup_RN18_aug
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4:gpus=2:nvidiaMin12GB,mem=16gb,walltime=24:00:00
#PBS -j oe
#PBS -q student
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/outputs_3/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate open_clip

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/src/training/main.py \
  --name='clip_no_0_no_dup_RN18_aug' \
  --save-frequency 99 \
  --report-to tensorboard \
  --train-data="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/train_data_no_0_no_dup.csv"  \
  --val-data="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/val_data_no_0_no_dup.csv"  \
  --csv-img-key filepath \
  --csv-caption-key sentence \
  --csv-separator="," \
  --warmup 2500 \
  --batch-size=32 \
  --lr=0.0002 \
  --wd=0.1 \
  --epochs=200 \
  --workers=8 \
  --model RN18 \
  --custom-aug \
  --dist-url 'tcp://localhost:10018'
