import torch.nn

# -------------------- test 1--------------------------
linear = torch.nn.Linear(5, 2)
linear.weight.data = linear.weight.data * 0 + 3.14
linear.bias.data = linear.bias.data * 0 + 3.14
input = torch.Tensor([2., 2., 2., 2., 2.])
output = linear(input)
print(output)

# -------------------- test 2--------------------------
linear = torch.nn.Linear(5, 2)
linear.weight.data = linear.weight.data * 0 + 3.14
linear.bias.data = linear.bias.data * 0 + 3.14
input = torch.Tensor([[2., 2., 2., 2., 2.], [2., 2., 2., 2., 2.]])
input.requires_grad = True
output = linear(input)
output = output ** 2
output.backward(torch.ones([3, 2]))
print(input.grad)
