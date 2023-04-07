import torch.nn

# -------------------- test 1--------------------------
cov = torch.nn.Conv2d(3, 8, 3)
cov.weight.data = cov.weight.data * 0 + 3.14
cov.bias.data = cov.bias.data * 0 + 3.14
input = torch.ones([16, 3, 256, 256])
output = cov(input)
# print(output)

# -------------------- test 2--------------------------
cov = torch.nn.Conv2d(3, 8, 3)
cov.weight.data = cov.weight.data * 0 + 3.14
cov.bias.data = cov.bias.data * 0 + 3.14
input = torch.ones([16, 3, 256, 256], requires_grad=True)
output = cov(input)
output.backward(torch.ones_like(output))
print(input.grad)

