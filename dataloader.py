import torch
import torch.utils.data as Data
torch.manual_seed(1)

BATCH_SIZE = 5
x = torch.linspace(1, 10, 10) 
y = torch.linspace(10, 1, 10) 
####
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=2,              # 多线程来读数据
    )

for epoch in range(3):   # 训练所有!整套!数据 3 次
    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
    # 假设这里就是你训练的地方...
    # 打出来一些数据
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                batch_x.numpy(), '| batch y: ', batch_y.numpy())
        """
        Epoch:  0 | Step:  0 | batch x:  [ 6.  7.  2.  3.  1. ] | batch y:  [  5.   4.   9.   8.  10. ]
        Epoch:  0 | Step:  1 | batch x:  [  9.  10.   4.   8.   5. ] | batch y:  [ 2.  1.  7.  3.  6. ]
        """
