import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd
import numpy as np
import random
filters = Variable(torch.Tensor([0.1,0.2,0.3]).view(1,1,3))
inputs = Variable(torch.arange(0,16).view(4,1,4))
F.conv1d(inputs, filters)
'''
Variable containing:
(0 ,.,.) = 
  0.8000  1.4000

(1 ,.,.) = 
  3.2000  3.8000

(2 ,.,.) = 
  5.6000  6.2000

(3 ,.,.) = 
  8.0000  8.6000
[torch.FloatTensor of size 4x1x2]
'''
