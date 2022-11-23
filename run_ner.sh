#!/usr/bin/env bash
# @Author: nijiahui
# @Date:   2021-09-07 14:19:36

TASK_NAME="ner"
MODEL_NAME="chinese_L-12_H-768_A-12"

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
# # 【horovod】get GPU - number
# gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

export PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
export BERT_BASE_DIR=$PRETRAINED_MODELS_DIR/$MODEL_NAME
export DATA_DIR=$CURRENT_DIR/ner_dataset
# 【减少打印】
export OMPI_MCA_btl_vader_single_copy_mechanism='none'

# run task
cd $CURRENT_DIR
echo "Start running..."

python run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DATA_DIR/$TASK_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=48 \
  --train_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$CURRENT_DIR/${TASK_NAME}_output \
