import torch
# query=torch.rand(10,5,256)
# key=torch.rand(10,5,256)
#
# t=torch.matmul(query,key)
x = torch.randn(3,2)

y = x.repeat(1,1)
print(x)
print(y)
