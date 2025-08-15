import numpy as np
import torch

class RandomCropPaste(object):
    def __init__(self, size, alpha=1.0, flip_p=0.5):
        """Randomly flip and paste a cropped image on the same image. """
        self.size = size
        self.alpha = alpha
        self.flip_p = flip_p

    def __call__(self, img):
        # 确保输入图像是3D张量 [C, H, W]
        if len(img.shape) != 3:
            return img
            
        lam = np.random.beta(self.alpha, self.alpha)
        front_bbx1, front_bby1, front_bbx2, front_bby2 = self._rand_bbox(lam)
        
        # 确保裁剪区域至少有一个像素
        if front_bbx2 <= front_bbx1 or front_bby2 <= front_bby1:
            return img
            
        # 确保裁剪区域在图像范围内
        if front_bbx1 >= self.size or front_bby1 >= self.size or front_bbx2 <= 0 or front_bby2 <= 0:
            return img
            
        img_front = img[:, front_bby1:front_bby2, front_bbx1:front_bbx2].clone()
        front_w = front_bbx2 - front_bbx1
        front_h = front_bby2 - front_bby1

        # 确保粘贴位置不会超出图像边界
        max_x = max(1, self.size - front_w)
        max_y = max(1, self.size - front_h)
        
        if max_x <= 0 or max_y <= 0:
            return img
            
        img_x1 = np.random.randint(0, max_x)
        img_y1 = np.random.randint(0, max_y)
        img_x2 = img_x1 + front_w
        img_y2 = img_y1 + front_h

        # 确保粘贴区域在图像范围内
        if img_x2 > self.size or img_y2 > self.size:
            return img

        if np.random.rand(1) <= self.flip_p:
            img_front = img_front.flip((-1,))
        if np.random.rand(1) <= self.flip_p:
            img = img.flip((-1,))

        mixup_alpha = np.random.rand(1)
        
        # 创建新的图像副本
        new_img = img.clone()
        # 执行混合操作
        new_img[:, img_y1:img_y2, img_x1:img_x2] = (
            img[:, img_y1:img_y2, img_x1:img_x2] * mixup_alpha + 
            img_front * (1 - mixup_alpha)
        )
        
        return new_img

    def _rand_bbox(self, lam):
        W = self.size
        H = self.size
        
        # 确保裁剪比例在合理范围内
        cut_rat = np.sqrt(1. - lam)
        cut_rat = np.clip(cut_rat, 0.1, 0.9)  # 限制裁剪比例在0.1到0.9之间
        
        cut_w = max(1, np.int32(W * cut_rat))
        cut_h = max(1, np.int32(H * cut_rat))

        # 确保裁剪区域不会太大
        cut_w = min(cut_w, W-1)
        cut_h = min(cut_h, H-1)

        # 计算中心点
        cx = np.random.randint(0, W)
        cy = np.random.randint(0, H)

        # 计算边界框
        bbx1 = np.clip(cx - cut_w // 2, 0, W-1)
        bby1 = np.clip(cy - cut_h // 2, 0, H-1)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # 确保边界框至少有一个像素
        if bbx2 <= bbx1:
            bbx2 = bbx1 + 1
        if bby2 <= bby1:
            bby2 = bby1 + 1

        return bbx1, bby1, bbx2, bby2

class CutMix(object):
  def __init__(self, size, beta):
    self.size = size
    self.beta = beta

  def __call__(self, batch):
    img, label = batch
    rand_img, rand_label = self._shuffle_minibatch(batch)
    lambda_ = np.random.beta(self.beta,self.beta)
    r_x = np.random.uniform(0, self.size)
    r_y = np.random.uniform(0, self.size)
    r_w = self.size * np.sqrt(1-lambda_)
    r_h = self.size * np.sqrt(1-lambda_)
    x1 = int(np.clip(r_x - r_w // 2, a_min=0, a_max=self.size))
    x2 = int(np.clip(r_x + r_w // 2, a_min=0, a_max=self.size))
    y1 = int(np.clip(r_y - r_h // 2, a_min=0, a_max=self.size))
    y2 = int(np.clip(r_y + r_h // 2, a_min=0, a_max=self.size))
    img[:, :, x1:x2, y1:y2] = rand_img[:, :, x1:x2, y1:y2]
    
    lambda_ = 1 - (x2-x1)*(y2-y1)/(self.size*self.size)
    return img, label, rand_label, lambda_

  def _shuffle_minibatch(self, batch):
    img, label = batch
    rand_img, rand_label = img.clone(), label.clone()
    rand_idx = torch.randperm(img.size(0))
    rand_img, rand_label = rand_img[rand_idx], rand_label[rand_idx]
    return rand_img, rand_label

class MixUp(object):
  def __init__(self, alpha=0.1):
    self.alpha = alpha

  def __call__(self, batch):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    x, y = batch
    lam = np.random.beta(self.alpha, self.alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam