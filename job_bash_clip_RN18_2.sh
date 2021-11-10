#PBS -N clip_RN18_closs3_hflip
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=6:gpus=4:ubuntu2004:nvidiaTITANX,mem=16gb,walltime=24:00:00
#PBS -j oe
#PBS -o /misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/outputs_final/


homePath='/misc/student/alzouabk/miniconda3'
source $homePath/bin/activate open_clip

echo "pid, gpu_utilization [%], mem_utilization [%], max_memory_usage [MiB], time [ms]"
nvidia-smi --query-accounted-apps="pid,gpu_util,mem_util,max_memory_usage,time" --format=csv | tail -n1

echo 'Training Should start'
python3 /misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/src/training/main.py \
  --name='clip_RN18_closs3_hflip' \
  --save-frequency 49 \
  --report-to tensorboard \
  --t-sne \
  --logs='/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/outputs_final/' \
  --train-data="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/train_data_no_dup_w_labels_bboxes.csv"  \
  --val-data="/misc/student/alzouabk/Thesis/self_supervised_pretraining/open_clip/val_data_no_dup_w_labels_bboxes.csv"  \
  --csv-img-key filepath \
  --csv-caption-key sentence \
  --csv-label-key labels \
  --csv-bbox-key bboxes \
  --csv-separator="," \
  --warmup 1000 \
  --batch-size=35 \
  --lr=0.000075 \
  --wd=0.1 \
  --epochs=100 \
  --workers=4 \
  --custom-loss-3 \
  --default-aug-img \
  --hflip-aug \
  --eval-train \
  --use-de-tokenizer \
  --custom-eval \
  --seed=101 \
  --model RN18 \
  --new-model \
  --embid-dim=512 \
  --IN-pretrained \
  --transformer-dim=768 \
  --dist-url 'tcp://localhost:10021'