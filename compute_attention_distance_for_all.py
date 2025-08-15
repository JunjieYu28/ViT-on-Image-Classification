from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import CIFAR10

from models.model_final import VisionTransformer, CONFIGS

raw_dataset = CIFAR10(root='./data', train=False, download=True)
classes = raw_dataset.classes

def load_model(No, config):
    model = VisionTransformer(config, num_classes=10, vis=True)
    checkpoint_path = f"/home/sichongjie/sichongjie-sub/ViT_torch/output_final/cifar10_No_{No}_checkpoint.bin" # 新增参数传入路径
    state_dict = torch.load(checkpoint_path, map_location="cpu")  # 加载权重文件
    model.load_state_dict(state_dict)  # 加载模型参数
    print(f"Loaded fine-tuned model from {checkpoint_path}")
    model.eval()
    return model

def load_img():
    raw_dataset = CIFAR10(root='./data', train=False, download=True)
    classes = raw_dataset.classes

    transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    image_idx = 1

    x = transform(raw_dataset[image_idx][0])

    return x


def compute_distance_matrix(patch_size, num_patches, length):
    """Helper function to compute distance matrix."""

    distance_matrix = np.zeros((num_patches, num_patches))

    for i in range(num_patches):
        for j in range(num_patches):
            if i == j: # zero distance
                continue

            xi, yi = (int(i/length)), (i % length)
            xj, yj = (int(j/length)), (j % length)

            distance_matrix[i, j] = patch_size*np.linalg.norm([xi - xj, yi - yj])
  
    return distance_matrix



def compute_mean_attention_dist(patch_size, attention_weights):
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert (length**2 == num_patches), ("Num patches is not perfect square")

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    attention_weights = attention_weights.detach().numpy()
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(mean_distances, axis=-1) # sum along last axis to get average distance per token
    mean_distances = np.mean(mean_distances, axis=-1) # now average across all the tokes

    return mean_distances

config = CONFIGS["ViT-Ours_Final"]

model = load_model("3", config)
x = load_img()
logits, att_mat_orig = model(x.unsqueeze(0))

num_blocks = 8

results = []

for i in range(num_blocks):
    attn_weights = att_mat_orig[i][0]
    results.append(compute_mean_attention_dist(1, attn_weights[:, 1:, 1:]))

print(results)

flattened_data = [arr.flatten().tolist() for arr in results]

df = pd.DataFrame(flattened_data)

df.to_csv("outputcsv/attn_dis2.csv", index=False, header=False)

print("数据已成功保存为 CSV 文件！")

    