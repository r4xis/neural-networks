import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import time

def get_num_correct(preds, labels):
  return float(preds.argmax(dim=1).eq(labels).sum())


def draw_line(train_values, test_values, epoch_value, y_label):

  plt.plot(epoch_value, train_values, linewidth=1, 
           color="blue", label='train')
  plt.plot(epoch_value, test_values, linewidth=1, 
           color="red", label='test')
    
  plt.legend()
  
  plt.xlabel("Epoch", fontsize=15)
  plt.ylabel(y_label, fontsize=15)

  plt.show()

class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    
    self.conv1 = nn.Conv2d(in_channels=3, 
                           out_channels=32, 
                           kernel_size=3,
                           padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, 
                           out_channels=64, 
                           kernel_size=3,
                           padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, 
                           out_channels=128, 
                           kernel_size=3,
                           padding=1)
    self.conv4 = nn.Conv2d(in_channels=128, 
                           out_channels=256, 
                           kernel_size=3,
                           padding=1)
    self.conv5 = nn.Conv2d(in_channels=256,
                           out_channels=256,
                           kernel_size=3,
                           padding=1)
    self.conv6 = nn.Conv2d(in_channels=256,
                           out_channels=128,
                           kernel_size=3,
                           padding=1)


    self.fc1 = nn.Linear(in_features=128*4*4, 
                         out_features=64)
    self.fc2 = nn.Linear(in_features=64, 
                         out_features=10)


    self.dropout1 = nn.Dropout(0.2)
    self.dropout2 = nn.Dropout(0.4)
    self.dropout3 = nn.Dropout(0.6)

    self.batchnorm1 = nn.BatchNorm2d(32)
    self.batchnorm2 = nn.BatchNorm2d(64)
    self.batchnorm3 = nn.BatchNorm2d(128)
    self.batchnorm4 = nn.BatchNorm2d(256)
    self.batchnorm5 = nn.BatchNorm2d(256)
    self.batchnorm6 = nn.BatchNorm2d(128)
    
  def forward(self, x):
    x = self.conv1(x)
    x = self.batchnorm1(x)
    x = F.leaky_relu(x)    
            
    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = F.leaky_relu(x)
    
    x = F.max_pool2d(x, 2, 2)
    x = self.dropout1(x)
    
    x = self.conv3(x)
    x = self.batchnorm3(x)
    x = F.leaky_relu(x)
              
    x = self.conv4(x)
    x = self.batchnorm4(x)
    x = F.leaky_relu(x)
    
    x = F.max_pool2d(x, 2, 2)
    x = self.dropout2(x)
    
    x = self.conv5(x)
    x = self.batchnorm5(x)
    x = F.leaky_relu(x)
              
    x = self.conv6(x)
    x = self.batchnorm6(x)
    x = F.leaky_relu(x)
    
    x = F.max_pool2d(x, 2, 2)
    x = self.dropout3(x)

    x = x.view(-1, 128*4*4)
    x = self.fc1(x)
    x = F.leaky_relu(x)
    x = self.fc2(x)
    return x
  


train_set = torchvision.datasets.CIFAR10(
    root="/content/cifar-10",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

test_set = torchvision.datasets.CIFAR10(
    root="/content/cifar-10",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
) 

train_loader = torch.utils.data.DataLoader(train_set, 
                                           batch_size=64, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_set, 
                                           batch_size=256,
                                           shuffle=True)

model = NN()

CUDA = torch.cuda.is_available()
if CUDA:
    model = model.cuda() 
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), 
                             lr=0.003, 
                             weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=128, 
                                            gamma=0.98)

train_loss = []
train_acc = []
test_loss = []
test_acc = []
epoch_val = []

start = time.time()

for epoch in range(100):
  
  total_loss = 0
  total_correct = 0
  iterations = 0
  model.train()

  for batch_train in train_loader:
      scheduler.step()
      images, labels = batch_train 
      if CUDA:
        images = images.cuda()
        labels = labels.cuda()

      optimizer.zero_grad()
      preds = model(images)
      loss = F.cross_entropy(preds, labels)

      loss.backward()
      optimizer.step()

      total_loss += loss.item()
      total_correct += get_num_correct(preds, labels)
      iterations += 1
      
  print(
      "epoch:", epoch+1, 
      "training accuracy:", total_correct/len(train_set), 
      "training loss:", total_loss/iterations
  )
  train_loss.append(total_loss/iterations)
  train_acc.append(total_correct/len(train_set))
  
#   if (epoch+1)%1 == 0:
  total_loss = 0
  total_correct = 0
  iterations = 0
  model.eval()

  for batch_test in test_loader:
    images, labels = batch_test
    if CUDA:
      images = images.cuda()
      labels = labels.cuda()

    preds = model(images)
    loss = F.cross_entropy(preds, labels)
    total_loss += loss.item()
    total_correct += get_num_correct(preds, labels)
    iterations += 1

  print(
    "epoch:", epoch+1, 
    "test accuracy:", total_correct/len(test_set), 
    "test loss:", total_loss/iterations
  )
  test_loss.append(total_loss/iterations)
  test_acc.append(total_correct/len(test_set))
  epoch_val.append(epoch+1)

  print("Learning Rate:", scheduler.get_lr()[0])
  
stop = time.time()
total_time = stop-start
print("Time:", time.strftime('%H:%M:%S', time.gmtime(total_time)))
torch.save(model.state_dict(), "/content/cifar-10.pth")

print()
draw_line(train_loss, test_loss, epoch_val, "Loss")
print()
draw_line(train_acc, test_acc, epoch_val, "Accuracy")
# model.load_state_dict(torch.load(filepath))
# model.eval()
