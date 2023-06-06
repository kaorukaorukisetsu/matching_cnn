import torch
from torch.utils.data import DataLoader
#from d2net import D2Net, D2NetLoss

from lib.model import D2Net
from d2loss_jisaku import D2NetLoss
from my_dataset import MyDataset

# 1. データセットを準備する
train_dataset = MyDataset("/home/natori21_u/jpg8k_me/400/")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. D2-Netモデルと損失関数を定義する
d2net = D2Net()
loss_fn = D2NetLoss()

# 3. トレーニングパラメータを設定する
optimizer = torch.optim.Adam(d2net.parameters(), lr=0.001)
num_epochs = 50

# 4. モデルをトレーニングする
for epoch in range(num_epochs):
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        images, gt_keypoints = batch
        pred_keypoints = d2net(images)
        loss = loss_fn(pred_keypoints, gt_keypoints)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {i + 1}, Loss: {loss.item()}")

# 5. モデルを評価する
test_dataset = MyDataset(is_train=False)
test_dataloader = DataLoader(test_dataset, batch_size=32)
d2net.eval()
with torch.no_grad():
    for batch in test_dataloader:
        images, gt_keypoints = batch
        pred_keypoints = d2net(images)
        # mAPなどの指標を計算する

# 6. 学習済みモデルを保存する
torch.save(d2net.state_dict(), "d2net_trained.pth")
