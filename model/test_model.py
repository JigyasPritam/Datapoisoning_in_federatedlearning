import torch
import sys
sys.path.append('.')
from model.net import MLP, CNN

# Test MLP with MNIST (10 classes)
mlp = MLP(num_classes=10)
sample = torch.randn(4, 1, 28, 28)  # batch of 4 images
output = mlp(sample)
print(f"MLP output shape (MNIST): {output.shape}")

# Test MLP with EMNIST (62 classes)
mlp_emnist = MLP(num_classes=62)
output = mlp_emnist(sample)
print(f"MLP output shape (EMNIST): {output.shape}")

# Test CNN with MNIST (10 classes)
cnn = CNN(num_classes=10)
output = cnn(sample)
print(f"CNN output shape (MNIST): {output.shape}")

# Test CNN with EMNIST (62 classes)
cnn_emnist = CNN(num_classes=62)
output = cnn_emnist(sample)
print(f"CNN output shape (EMNIST): {output.shape}")

print("\nAll models working correctly!")