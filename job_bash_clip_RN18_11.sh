#PBS -N clip_RN18_custom_loss_3_3090
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=4:gpus=2:ubuntu2004:nvidiaRTX3090,mem=16gb,walltime=24:00:00
#PBS -j oe
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/outputs/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate open_clip

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'


echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/src/training/main.py \
  --name='clip_RN18_custom_loss_3_3090' \
  --save-frequency 199 \
  --report-to tensorboard \
  --train-data="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/train_data_no_dup_w_labels.csv"  \
  --val-data="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/val_data_no_dup_w_labels.csv"  \
  --csv-img-key filepath \
  --csv-caption-key sentence \
  --csv-label-key labels \
  --csv-separator="," \
  --warmup 2500 \
  --batch-size=64 \
  --lr=0.0002 \
  --wd=0.1 \
  --epochs=3 \
  --workers=8 \
  --model RN18 \
  --custom-loss-3 \
  --default-aug \
  --eval-train \
  --custom-eval \
  --dist-url 'tcp://localhost:10017'