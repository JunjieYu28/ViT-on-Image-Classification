import torch
import matplotlib.pyplot as plt
from torch.optim import SGD
from scheduler import ConstantCosineSchedule, WarmupCosineSchedule

model = torch.nn.Linear(10, 2)
optimizer = SGD(model.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20000, eta_min=1e-5)
# scheduler = WarmupCosineSchedule(optimizer, warmup_steps=1000, t_total=20000, min_lr=1e-5)
scheduler = ConstantCosineSchedule(optimizer, 2000, 20000, min_lr=1e-5)

lrs = []
for step in range(20000):
    lrs.append(scheduler.get_last_lr()[0])
    optimizer.step()
    scheduler.step()

plt.plot(lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('ConstantCosineSchedule (constant_steps=2000, t_total=20000)')
plt.savefig('constant_cosine_lr.png')
plt.show()