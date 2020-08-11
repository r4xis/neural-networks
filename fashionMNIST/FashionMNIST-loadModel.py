import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

class NN(nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    
    self.conv1 = nn.Conv2d(in_channels=1, 
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
                           padding=2)
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
  
test_set = torchvision.datasets.FashionMNIST(
    root="/content/fashion-mnist-test",
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)

batch_size = 1

test_loader = torch.utils.data.DataLoader(test_set, 
                                           batch_size=batch_size,
                                           shuffle=True)

classes = ("t-shirt/top", "trouser", "pullover", "dress", 
          "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot")

CUDA = torch.cuda.is_available()

for batch_train in test_loader: # Get Batch
      model.load_state_dict(torch.load("/content/fashionMNIST.pth"))
      model.eval()
      if CUDA:
        model = model.cuda()
#       image = cv2.imread("/content/cat.jpg")
#       trans1 = transforms.ToTensor()
#       image = trans1(image)
      images, labels = batch_train 
      if CUDA:
        images = images.cuda()
        labels = labels.cuda()
#         image = image.cuda()
#       image = image.unsqueeze(0)
      preds = model(images) # Pass Batch
      images, labels = batch_train 
      grid = torchvision.utils.make_grid(images, nrow=1)
      plt.figure(figsize=(3, 3))
      plt.imshow(np.transpose(grid, (1,2,0)))
      print("Pred:", classes[preds.argmax(dim=1)], 
            "-", "Label:", classes[labels])
      break
