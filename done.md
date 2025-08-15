# what have done:

1. **aug_utils.py中定义了RandomCropPaste，Cutmix,Mixup函数进行了数据增广**
2. **data_aug.py数据增广下的data_loader**
3. **train_aug.py实现RandomCropPaste,Mixup,Cutmix**
4. **train_aug_pro进一步加入了random_aug**
5. **tmux aug 中run random_aug,RandomCropPaste,Mixup,Cutmix下的训练模型**
6. **tmux lr_1eng3 中run RandomCropPaste,Mixup,Cutmix下训练的模型**
7. **现在可以暂时不使用train_aug.py/.sh了，在train_aug_pro.sh中选择参数 --aug_type "batch_random" --random_aug False即可选择是否使用random_aug，但注意改模型名称**

目前最好模型：/home/sichongjie/sichongjie-sub/ViT_torch/output_aug/cifar10_lr_1Eng3_Weight_Decay_Data_augment_res_checkpoint.bin

test1: random_aug + mixup
test2: random_aug + None