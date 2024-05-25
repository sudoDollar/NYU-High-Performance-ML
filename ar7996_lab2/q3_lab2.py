from model import Resnet18
from params import Config
import torch
import torch.nn as nn
import torch.optim as optim

#Q3: Number of Parameters and Gradients

#Read parameters from user input
config = Config().parse()

#model
resnet = Resnet18(num_blocks=[2,2,2,2], strides=[1,2,2,2], batch=True)

# Count trainable parameters
total_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
print("Total trainable parameters in ResNet-18:", total_params)

# Define a random input tensor
input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, RGB channels, input size 32x32
target_tensor = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

# Perform forward pass to compute gradients
resnet.train()
output = resnet(input_tensor)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(resnet.parameters(), lr=0.1, weight_decay=5e-4)


loss = loss_fn(output, target_tensor) 
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Count gradients
total_gradients = sum(p.grad.numel() for p in resnet.parameters() if p.grad is not None)
print("Total gradients in ResNet-18:", total_gradients)

