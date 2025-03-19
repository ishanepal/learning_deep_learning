import torch
import torch.nn as nn

class MyDenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim) :
        super (MyDenseLayer, self).__init__()

        # Initialize weights and bias
        self.W = nn.Parameter(torch.randn(input_dim, output_dim, requires_grad = True))
        self.b = nn.Parameter(torch.randn(1, output_dim, requires_grad = True))

    def forward(self, inputs):
        # Forward propagate the inputs
        z = torch.matmul(inputs, self.W) + self.b

        # Feed through a non-linear activation
        output = torch.sigmoid(z)
        return output
    
# Define the layer
layer = MyDenseLayer(input_dim=3, output_dim=2)

# Create a random input tensor (batch_size=4, input_dim=3)
inputs = torch.randn(4, 3)

# Forward pass
output = layer(inputs)

print(output)