python3 train_aug_pro.py \
--name cifar10_lr_1Eng3_Weight_Decay_Data_augment \
--dataset cifar10 \
--model_type ViT-Ours_new \
--pretrained_dir checkpoint/ViT-B_16.npz \
--train_batch_size 512 \
--eval_batch_size 1024 \
--eval_every 200 \
--learning_rate 1e-3 \
--weight_decay 5e-5 \
--num_steps 20000 \
--warmup_steps 500 \
--seed 42 \
--load_from_pretrained False \
--decay_type "cosine" \
--constant_steps 8000 \
--img_size 32 \
--optimizer Adam \
--aug_type "batch_random" \
--random_aug False \
--mixup_rate 0.2 \
--cutmix_rate 1.0 \
--cut_rate 1.0 \
--flip_p 0.5 \
--min_lr 1e-5 \
--output_dir output_aug

# Attention, please!!! 
# The dropout rate --> configs.py