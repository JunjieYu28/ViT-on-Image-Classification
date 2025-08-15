python3 train.py \
--name cifar10_lr_2Eng3_Weight_Decay_5Eng5_Res_New \
--dataset cifar10 \
--model_type ViT-Ours_Res \
--pretrained_dir checkpoint/ViT-B_16.npz \
--train_batch_size 512 \
--eval_batch_size 1024 \
--eval_every 200 \
--learning_rate 2e-3 \
--weight_decay 5e-5 \
--num_steps 20000 \
--warmup_steps 500 \
--seed 42 \
--load_from_pretrained False \
--decay_type "cosine" \
--constant_steps 8000 \
--img_size 32 \
--optimizer Adam \
--min_lr 1e-5 \
--output_dir output

# Attention, please!!! 
# The dropout rate --> configs.py