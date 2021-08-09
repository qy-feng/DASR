import torch
import torch.nn as nn
import torch.nn.functional as F

# class Discriminator_domain(nn.Module):
#     def __init__(self, input_dim):
#         super(Discriminator_domain, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128,1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)
        
#     def forward(self, input):
#         input = input.view(input.size(0), -1)
#         out = F.relu(self.bn1(self.fc1(input)))
#         out = F.relu(self.bn2(self.fc2(out)))
#         out = F.relu(self.bn3(self.fc3(out)))
#         return torch.sigmoid(self.fc4(out))
    
    
# class Discriminator_shape(nn.Module):
#     def __init__(self, input_dim):
#         super(Discriminator_shape, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128,1)
#         self.bn1 = nn.BatchNorm1d(512)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(128)
        
#     def forward(self, input):
#         input = input.view(input.size(0), -1)
#         out = F.relu(self.bn1(self.fc1(input)))
#         out = F.relu(self.bn2(self.fc2(out)))
#         out = F.relu(self.bn3(self.fc3(out)))
#         return torch.sigmoid(self.fc4(out))

    
class Discriminator_domain(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator_domain, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        
    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return torch.sigmoid(out)
    
    
class Discriminator_shape(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator_shape, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        
    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return torch.sigmoid(out)