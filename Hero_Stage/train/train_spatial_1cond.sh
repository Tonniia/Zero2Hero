export MODEL_DIR="./_pretrained_model/FLUX.1-dev" # your flux path
export OUTPUT_DIR="./_result/hero_stage/Mode1/subject_driven/colorful_sneaker_style1"  # your save path
# export CONFIG=""./default_config.yaml""
export TRAIN_DATA="./_input/_json/hero_stage/subject_driven/colorful_sneaker/style1_1cond_pixelswap.jsonl" # your data jsonl file
export LOG_PATH="$OUTPUT_DIR/log"
# accelerate launch --config_file $CONFIG
CUDA_VISIBLE_DEVICES=0 python ./Hero_Stage/train/train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --lora_num=1 \
    --cond_size=512 \
    --noise_size=512 \
    --subject_column="None" \
    --spatial_column="source" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 128 \
    --network_alphas 128 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "a 3D model" \
    --num_train_epochs=600 \
    --validation_steps=100 \
    --checkpointing_steps=200 \
    --eval_json_path ./_input/_json/hero_stage/subject_driven/colorful_sneaker/style1_1cond_pixelswap_eval.json \
    --subject_test_images None \
    --test_h 512 \
    --test_w 512 \
    --num_validation_images=1