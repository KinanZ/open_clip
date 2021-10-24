#PBS -N clip_RN18_t_sne_test
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=6:gpus=4:ubuntu2004:nvidiaTITANX,mem=16gb,walltime=24:00:00
#PBS -j oe
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/outputs/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate open_clip

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/src/training/main.py \
  --name='clip_RN18_t_sne_test' \
  --save-frequency 199 \
  --report-to tensorboard \
  --logs='/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/outputs/' \
  --train-data="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/train_data_no_dup_w_labels.csv"  \
  --val-data="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/val_data_no_dup_w_labels.csv"  \
  --csv-img-key filepath \
  --csv-caption-key sentence \
  --csv-label-key labels \
  --csv-separator="," \
  --warmup 100 \
  --batch-size=35 \
  --lr=0.0002 \
  --wd=0.1 \
  --epochs=10 \
  --workers=4 \
  --model RN18 \
  --custom-loss-3 \
  --default-aug-img \
  --eval-train \
  --set-aug-text \
  --custom-eval \
  --seed=101 \
  --t-sne \
  --dist-url 'tcp://localhost:10026'
