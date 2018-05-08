import torch

x = torch.Tensor(3,4)
#print (x)

y = torch.randn(3,4)
#print (y)

#if torch.cuda.is_available():
x = x.cuda()
y = y.cuda()
    
	
print(str(x+y))
