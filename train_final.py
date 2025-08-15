# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import json

from datetime import timedelta

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.model_final import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule, ConstantCosineSchedule, WarmupConstantCosineSchedule
from utils.data_aug import get_loader
from utils.dist_util import get_world_size


logger = logging.getLogger(__name__)


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    # if args.load_from_pretrained:
    #     model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args_dict = vars(args)
    args_dict["device"] = str(args_dict["device"])
    args_json = json.dumps(args_dict, indent=4) 

    writer.add_text("Args/All_arguements", args_json)

    compact_json = config.to_json() 
    config_dict = json.loads(compact_json) 
    config_dict.pop("patches", None)
    config_text = json.dumps(config_dict, indent=4) 

    writer.add_text("Configs/Full_Config", config_text)

    writer.add_text("Configs/Num_Params", str(num_params))

    return args, model, writer


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def top_k_accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  # shape: [batch_size, maxk]
    pred = pred.t()  # shape: [maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # shape: [maxk, batch_size]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append((correct_k / batch_size).item())
    return res

def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    return fig

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()
    all_preds, all_label, all_logits = [], [], []

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    loss_fct = nn.CrossEntropyLoss()
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        all_preds.append(preds.detach().cpu())
        all_label.append(y.detach().cpu())
        all_logits.append(logits.detach().cpu())

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds = torch.cat(all_preds, dim=0)
    all_label = torch.cat(all_label, dim=0)
    all_logits = torch.cat(all_logits, dim=0)

    top1, top5 = top_k_accuracy(all_logits, all_label, topk=(1, 5))
    accuracy = simple_accuracy(all_preds.numpy(), all_label.numpy())


    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Validation Loss: %2.5f" % eval_losses.avg)
    logger.info("Top-1 Accuracy: %2.5f" % top1)
    logger.info("Top-5 Accuracy: %2.5f" % top5)
    writer.add_scalar("loss", eval_losses.avg, global_step)
    writer.add_scalar("top1_accuracy", top1, global_step)
    writer.add_scalar("top5_accuracy", top5, global_step)

    report = classification_report(all_label.numpy(), all_preds.numpy(), output_dict=True)
    for avg_type in ["macro avg", "weighted avg"]:
        for metric_name, value in report[avg_type].items():
            writer.add_scalar(f"{avg_type.replace(' ', '_')}_{metric_name}", value, global_step=global_step)

    return accuracy, all_preds.numpy(), all_label.numpy(), top1, top5

def train(args, model, writer):
    """ Train the model """
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    train_loader, test_loader = get_loader(args)

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.learning_rate, 
                                betas=(args.beta1, args.beta2), 
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total, min_lr=args.min_lr)
    elif args.decay_type == "constant_cosine":
        scheduler = ConstantCosineSchedule(optimizer, constant_steps=args.constant_steps, t_total=t_total, min_lr=args.min_lr)
    elif args.decay_type == "linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    elif args.decay_type == "warmup_constant_cosine":
        scheduler = WarmupConstantCosineSchedule(optimizer, warmup_steps=args.warmup_steps, constant_steps=args.constant_steps, t_total=t_total, min_lr=args.min_lr)
    else:
        raise ValueError(f"Unknown decay_type: {args.decay_type}")

    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  
    losses = AverageMeter()
    global_step, best_acc, best_top1, best_top5 = 0, 0, 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if len(batch) == 2:  # 普通数据或RandomCropPaste
                x, y = batch
                x, y = x.to(args.device), y.to(args.device)
                loss = model(x, y)
            elif len(batch) == 4:  # MixUp或CutMix
                x, y_a, y_b, lam = batch
                x = x.to(args.device)
                if isinstance(y_a, torch.Tensor) and isinstance(y_b, torch.Tensor):  # MixUp
                    y_a, y_b = y_a.to(args.device), y_b.to(args.device)
                    loss = lam * model(x, y_a) + (1 - lam) * model(x, y_b)
                else:  # CutMix
                    y, y_rand = y_a.to(args.device), y_b.to(args.device)
                    loss = lam * model(x, y) + (1 - lam) * model(x, y_rand)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy, all_preds, all_label, top1, top5 = valid(args, model, writer, test_loader, global_step)
                    if best_acc < accuracy:
                        save_model(args, model)
                        best_acc = accuracy

                        cm = confusion_matrix(all_label, all_preds)
                        cm_fig = plot_confusion_matrix(cm, test_loader.dataset.classes)
                        writer.add_figure("Confusion_Matrix/Best", cm_fig, global_step)
                    if best_top1 < top1:
                        best_top1 = top1
                    if best_top5 < top5:
                        best_top5 = top5
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        final_best_result = (
        f"Best Accuracy: {best_acc:.4f}\n"
        f"Best Top-1: {best_top1:.4f}\n"
        f"Best Top-5: {best_top5:.4f}"
        )
        writer.add_text("Final/Best_Results", final_best_result, global_step)
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which task.")
    parser.add_argument("--model_type", 
                        default="ViT-Ours_Final",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output_final", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=32, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1024, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=200, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--aug_type", default="batch_random", type=str, choices=["None", "mixup", "random_crop_paste", "cutmix", "batch_random"], 
                        help="Type of data augmentation.")
    parser.add_argument('--random_aug', type=lambda x: x.lower() == 'true', default=False,
                        help="Enable random augmentation. Use True or False.")
    parser.add_argument("--mixup_rate", default=0.2, type=float,
                    help="mixup hyperparameter alpha, the mix is stronger when alpha is larger, [0,1], suggested 0.1~0.4 \n")
    parser.add_argument("--cutmix_rate", default=0.8, type=float,
                    help="cutmix hyperparameter beta, the interference is stronger when beta is smaller, 0.1 ~ 1.0,suggested larger than 0.5 \n")
    parser.add_argument("--cut_rate", default=1.0, type=float,
                help="random_crop_paste hyperparameter alpha, the smaller the cuted picture the larger alpha, 0.5~2.0 usually\n")
    parser.add_argument("--flip_p", default=0.5, type=float,
                help="random_crop_paste hyperparameter p, flip probability\n")

    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for SGD.\n")
    parser.add_argument("--weight_decay", default=5e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=20000, type=int,
                        help="Total number of training epochs to perform.\n")
    parser.add_argument("--decay_type", choices=["cosine", "linear", "constant_cosine", "warmup_constant_cosine"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.\n")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus.\n")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization.\n")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit.\n")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--load_from_pretrained', type=bool, default=False,
                        help="Loading from a pre-trained model or not.\n")
    parser.add_argument("--constant_steps", default=4000, type=int,
                    help="Number of steps with constant learning rate before cosine decay.\n")
    parser.add_argument("--min_lr", default=1e-5, type=float,
                    help="Minimum lr while using Cosine LRscheduler.\n")
    parser.add_argument("--optimizer", default="Adam", type=str,
                    help="Type of optimizer.\n")
    parser.add_argument("--beta1", default=0.9, type=float,
                    help="Adam beta1.\n")
    parser.add_argument("--beta2", default=0.999, type=float,
                    help="Adam beta2.\n")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    set_seed(args)

    args, model, writer = setup(args)

    train(args, model, writer)


if __name__ == "__main__":
    main()
