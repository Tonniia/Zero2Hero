export MODEL_DIR="/root/tongtong.tt/_ckpt/huggingface/black-forest-labs/FLUX.1-dev" # your flux path
export OUTPUT_DIR="./_result"  # your save path
export CONFIG=""./default_config.yaml""
export TRAIN_DATA="/root/tongtong.tt/_code/Zero2Hero/_input_dev/_json/hero_stage/car_turn/rainbow_subinp_train.jsonl" # your data jsonl file
export LOG_PATH="$OUTPUT_DIR/log"
# accelerate launch --config_file $CONFIG
CUDA_VISIBLE_DEVICES=0 python ./Hero_Stage/train/train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --lora_num=2 \
    --cond_size=512 \
    --noise_size=512 \
    --subject_column="None" \
    --spatial_column="source_zero,source_tgt" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 128 128 \
    --network_alphas 128 128 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "" \
    --num_train_epochs=600 \
    --validation_steps=200 \
    --checkpointing_steps=200 \
    --eval_json_path /root/tongtong.tt/_code/Zero2Hero/_input_dev/_json/hero_stage/car_turn/rainbow_subinp_eval.json \
    --subject_test_images None \
    --test_h 512 \
    --test_w 512 \
    --num_validation_images=1 \
    --gradient_checkpointing
    


