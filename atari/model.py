import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

ch = 64
ksize = 4
class DQN(nn.Module):

    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, ch, kernel_size=ksize, stride=2, padding = 0)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=ksize, stride=2, padding = 0)
        self.conv3 = nn.Conv2d(ch, ch, kernel_size=ksize, stride=2, padding = 0)
        #self.dense1 = nn.Linear(2592, 1024)
        self.head = nn.Linear(4096, num_actions)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        # x = F.relu(self.dense1(x))
        return 2*F.tanh(self.head(x))