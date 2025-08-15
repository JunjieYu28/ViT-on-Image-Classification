python3 train_based_pre.py \
--name cifar10_lr_1Eng3_Weight_Decay_Data_augment_pro_base_aug \
--dataset cifar10 \
--model_type ViT-Ours_new \
--finetune_path /home/sichongjie/sichongjie-sub/ViT_torch/output_aug/cifar10_lr_1Eng3_Weight_Decay_Data_augment_checkpoint.bin \
--train_batch_size 512 \
--eval_batch_size 1024 \
--eval_every 200 \
--learning_rate 1e-3 \
--weight_decay 5e-5 \
--num_steps 20000 \
--warmup_steps 500 \
--seed 42 \
--load_from_pretrained True \
--decay_type "cosine" \
--constant_steps 8000 \
--img_size 32 \
--optimizer Adam \
--aug_type "batch_random" \
--random_aug True \
--min_lr 1e-5 \
--output_dir output_loadmodel

# Attention, please!!! 
# The dropout rate --> configs.py