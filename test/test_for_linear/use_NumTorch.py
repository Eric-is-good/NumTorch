import NumTorch.nn

# -------------------- test 1--------------------------
linear = NumTorch.nn.Linear(5, 2)
linear.weight.data = linear.weight.data * 0 + 3.14
linear.bias.data = linear.bias.data * 0 + 3.14
input = NumTorch.Tensor([2., 2., 2., 2., 2.])
output = linear(input)
print(output)

# -------------------- test 2--------------------------
linear = NumTorch.nn.Linear(5, 2)
linear.weight.data = linear.weight.data * 0 + 3.14
linear.bias.data = linear.bias.data * 0 + 3.14
input = NumTorch.Tensor([[2., 2., 2., 2., 2.], [2., 2., 2., 2., 2.]]
                        , requires_grad=True)
output = linear(input)
output = output ** 2
output.backward()
print(input.grad)
