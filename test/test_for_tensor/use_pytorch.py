import torch as nt

# -------------------- test 1--------------------------
a = nt.Tensor([1, 2, 3, 4])
print(a)
print(a.is_leaf)

b = 2 * a
print(b)
print(b.is_leaf)

# -------------------- test 2--------------------------
a = nt.Tensor([1., 2., 3., 4.])
a.requires_grad = True
print(a)
print(a.is_leaf)

b = 2 * a
print(b)
print(b.is_leaf)

c = b.sum(0)
print(c.is_leaf)
c.backward()
print(a.grad)