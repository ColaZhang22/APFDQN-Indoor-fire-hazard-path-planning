import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, output):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(4 , 8)
        self.fc2 = nn.Linear(8 , 16)
        self.fc3 = nn.Linear(16 , 4)
        self.fc4 = nn.Linear(4 , output)
        self.d1=nn.Dropout(p=0.1)
        self.d2=nn.Dropout(p=0.1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.d1(x)
        x = F.relu(self.fc2(x))
        x = self.d2(x)
        x = F.relu(self.fc3(x))
        y = self.fc4(x)
        return y