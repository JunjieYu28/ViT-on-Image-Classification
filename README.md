## ViT-torch: Vision Transformer 在 CIFAR-10/100 上的实践（PyTorch）

本项目是一个将 Vision Transformer (ViT) 应用于小规模数据集（尤其是 CIFAR-10）的完整实践工程，包含：
- 模型实现与多种结构配置（原生 ViT、ResNet+ViT 混合、不同 patch/heads/blocks 配置、Stochastic Depth/DropPath 等）
- 训练与评估脚本（含学习率调度 Warmup/Linear/Cosine/Constant-Cosine/Warmup-Constant-Cosine）
- 数据增强（RandomCrop+Paste、MixUp、CutMix、RandAugment 与批次随机增强）
- 可视化与分析（注意力图、注意力距离、梯度 Rollout、特征图、位置编码相似度）

共同作者：
- Xunakun Yang (GitHub：[@xuankunyang](https://github.com/xuankunyang))
- Junjie Yu (Github: [@JunjieYu28] (https://github.com/JunjieYu28))

项目报告与演示文档：
- 报告 PDF：[`ViT.pdf`](ViT.pdf)
- 演示 PPT：[`ViT.pptx`](ViT.pptx)


### 目录结构概览

```
ViT_torch/
  left/                     # 早期/基础训练脚本与工具
    data_utils.py           # CIFAR-10/100 数据加载（基础增广）
    train.py                # 基础训练入口（models/modeling.py）
    train_aug.py            # 基于基础管线的增广训练
    train_aug_pro.py        # 进阶增广训练
    train_sd.py             # 随机深度（Stochastic Depth）实验
    random_aug.py           # 额外 RandAug 相关定义
  models/
    configs.py              # 各类模型超参配置（patch/hidden/heads/layers等）
    modeling.py             # ViT 主实现（含可选 ResNet 混合特征）
    modeling_sd.py          # 随机深度版本
    model_final.py          # 最终版 ViT（小图像友好，img_size=32）
  utils/
    data_aug.py             # 增强版数据加载（MixUp/CutMix/RandomCropPaste/RandAugment）
    aug_utils.py            # MixUp/CutMix/RandomCropPaste 实现
    scheduler.py            # Warmup/Cosine/Constant-Cosine 等学习率调度
    augment_images_all/     # 增强示例图片
  scripts/                  # 训练脚本示例（.sh）
  train_final.py            # 强化版训练入口（默认使用 model_final.py 与 utils/data_aug.py）
  compute_attention_distance_for_all.py  # 注意力距离分析
  grad_rollout.py           # 梯度 Rollout 可视化
  可视化相关 ipynb：
    visualize_attention_map.ipynb
    visualize_attention_distance.ipynb
    visualize_embedding_filters.ipynb
    visualize_feature_map.ipynb
    visualize_grad_rollout.ipynb
  ViT.pdf / ViT.pptx        # 报告与演示文档
```


## 环境与依赖

- Python ≥ 3.8
- PyTorch、Torchvision（建议启用 CUDA）
- 其他：`numpy`、`ml-collections`、`scipy`、`tqdm`、`tensorboard`、`scikit-learn`、`seaborn`、`matplotlib`

示例安装：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # 根据你的 CUDA 版本调整
pip install numpy ml-collections scipy tqdm tensorboard scikit-learn seaborn matplotlib
```


## 数据

无需手动下载，首次运行会自动在 `./data` 下载 CIFAR-10/100。


## 快速开始（推荐：最终版管线）

`train_final.py` 使用 `models/model_final.py` 的小图像友好实现与 `utils/data_aug.py` 的增强数据管线，默认 `img_size=32`，支持更丰富的可视化统计与增强策略。

最小示例（CIFAR-10）：

```bash
python train_final.py --name exp_final_c10 \
  --dataset cifar10 \
  --model_type ViT-Ours_Final \
  --img_size 32 \
  --train_batch_size 512 --eval_batch_size 1024 \
  --learning_rate 1e-3 --weight_decay 5e-5 \
  --num_steps 20000 --decay_type cosine --warmup_steps 500 \
  --aug_type batch_random --random_aug true
```

关键参数（节选）：
- `--model_type`: 见下文“模型与配置”
- `--decay_type`: `cosine` | `linear` | `constant_cosine` | `warmup_constant_cosine`
- `--aug_type`: `None` | `mixup` | `random_crop_paste` | `cutmix` | `batch_random`
- `--random_aug`: 是否启用 `RandAugment`（与上面增广可共存）
- `--mixup_rate`、`--cutmix_rate`、`--cut_rate`、`--flip_p`: 对应 `MixUp`、`CutMix`、`RandomCropPaste` 的超参

输出：
- 模型权重：`output_final/{name}_checkpoint.bin`
- TensorBoard 日志：`logs/{name}`（包含 Top-1/Top-5、混淆矩阵图、配置快照等）

查看训练曲线：

```bash
tensorboard --logdir logs
```


## 基础管线（对照）

`left/train.py` 使用 `models/modeling.py` 与 `left/data_utils.py`（更基础的增广），适合对照实验：

```bash
python left/train.py --name exp_base_c10 \
  --dataset cifar10 \
  --model_type ViT-B_16 \
  --img_size 32 \
  --train_batch_size 512 --eval_batch_size 64 \
  --learning_rate 3e-2 --weight_decay 0 \
  --num_steps 10000 --decay_type warmup_constant_cosine --warmup_steps 500
```

输出：`output/{name}_checkpoint.bin` 与 `logs/{name}`。


## 模型与配置（节选）

在 `models/model_final.py` 与 `models/modeling.py` 中通过 `CONFIGS` 字典选择配置：

- 最终版与变体（来自 `model_final.py` → `configs.py`）：
  - `ViT-Ours_Final`、`ViT-Ours_sd{0..4}`、`ViT-Ours_dp{0..3}`、`ViT-Ours_adp{0..3}`、`ViT-Ours_res{0..2}`
  - `ViT-Ours_ps{2,4,8}`（不同 patch size）、`ViT-Ours_nb{4,12}`（不同层数）、`ViT-Ours_nh{8,16}`（不同 heads）
  - `ViT-Ours_set_288_288/384/768`、`ViT-Ours_set_384_768` 等

- 基础版（来自 `modeling.py` → `configs.py`）：
  - `ViT-Ours_Res`、`ViT-Ours`、`ViT-Ours_new`、`ViT-B_16/B_32`、`ViT-L_16/L_32`、`ViT-H_14`、`R50-ViT-B_16`

说明：
- 若配置中 `ResNet_type != 0` 或存在 `patches.grid`，则使用 ResNet 特征作为混合输入（Hybrid ViT）。
- `transformer.prob_pass`>0 时启用训练时随机跳层（近似 DropPath / Stochastic Depth）。


## 数据增强说明

- `utils/aug_utils.py`：
  - `RandomCropPaste(size, alpha, flip_p)`：在同图随机裁剪-翻转-粘贴，并做局部混合；适合小图像的“结构扰动”。
  - `MixUp(alpha)`、`CutMix(beta)`：标准混合增强；在 `train_final.py` 中按批处理逻辑自动计算损失。
- `utils/data_aug.py`：
  - `--aug_type` 统一切换增强策略；`batch_random` 会在 `random_crop_paste`、`mixup`、`cutmix` 三者中随机选择。
  - `--random_aug true` 会开启 `RandAugment`，与上述增强可叠加。
- 示例可视化：见 `utils/augment_images_all/` 中的示例图片。


## 学习率调度

由 `utils/scheduler.py` 提供：
- `WarmupLinearSchedule`、`WarmupCosineSchedule(min_lr)`
- `ConstantCosineSchedule(constant_steps, min_lr)`
- `WarmupConstantCosineSchedule(warmup_steps, constant_steps, min_lr)`

通过 `--decay_type` 与 `--warmup_steps`、`--constant_steps`、`--min_lr` 组合启用。


## 评估与可视化

- 训练过程中每 `--eval_every` 步在验证集评估；`train_final.py` 记录 Top-1/Top-5、混淆矩阵，并保存最优权重。
- 注意力可视化与分析：
  - `visualize_attention_map.ipynb`、`visualize_grad_rollout.ipynb` 等 Jupyter Notebook
  - `compute_attention_distance_for_all.py`：注意力距离分析。注意：代码内 `checkpoint_path` 有示例硬编码路径，请根据你本地模型路径修改。


## 多卡/分布式

脚本支持 `--local_rank` 形式的分布式；如不使用分布式，保持 `--local_rank -1`（默认）。


## 常见问题（FAQ）

- 内存不足/显存不够：
  - 降低 `--train_batch_size` 或提高 `--gradient_accumulation_steps`
  - 选择更小的模型配置（例如更少的层/更低的 hidden）
- 训练不收敛：
  - 调整 `--learning_rate`、`--weight_decay`、`--warmup_steps`
  - 关闭过强的增强（例如先用 `--aug_type None` 验证基线）
- Windows 运行 `.sh`：
  - `.sh` 为示例命令集合，Windows 下建议直接使用 `python` 命令行运行对应参数。


## 引用与致谢

- 共同作者：Xuankun Yang (GitHub：[@xuankunyang](https://github.com/xuankunyang))与 Junjie Yu。感谢共同完成模型实现、训练与可视化。
- ViT 原始思想来源于 Google 的 Vision Transformer 论文与开源实现（本仓库部分配置与结构命名沿用其风格）。


## 许可证

本项目在参考开源实现基础上进行了大量修改与扩展，采用 MIT 许可证发布。请见仓库根目录的 `LICENSE` 文件。

本项目参考的上游项目许可为 MIT License，已在 `LICENSE` 中保留原版权声明：

- 本项目的增改与新增代码的版权声明：`Copyright (c) 2025 Junjie Yu and Xuankun Yang`

```text
MIT License

Copyright (c) 2020 jeonsworld
Copyright (c) 2025 JunjieYu28 and xuankunyang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

