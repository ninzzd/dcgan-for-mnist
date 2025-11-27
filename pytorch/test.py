import torch

x = torch.tensor([0,0,0], dtype=torch.float32, requires_grad=True)
b = torch.tensor([1,1,1], dtype=torch.float32)
y = torch.dot(b,x)
y.backward()
print(x.grad)