# NumTorch
用 numpy 实现一个简易 pytorch，用于学习 计算图 和 自动求导。



## 1. 缘由

在读大学时上 [**人工智能课**](https://github.com/Eric-is-good/2022_AI_lesson)，老师叫我们模仿写一个梯度回传的神经网络，当时是基于网络层 layer 写的，不具有通用性，这次打算基于张量 tensor 来写一个框架。**[学习地址](https://github.com/Kaslanarian/PyDyNet)**。


<!-- TOC -->

- [NumTorch](#numtorch)
  - [1. 缘由](#1-缘由)
  - [2. API](#2-api)
    - [2.1. tensor 基类](#21-tensor-基类)
  - [3. 开发文档](#3-开发文档)
    - [3.1. 基本框架](#31-基本框架)
    - [3.2. 计算图前向传播](#32-计算图前向传播)
    - [3.3. 计算图反向传播](#33-计算图反向传播)
    - [3.4. 自动微分](#34-自动微分)
    - [3.5. 静态图](#35-静态图)
      - [3.5.1. 基本图](#351-基本图)
      - [3.5.2. node节点](#352-node节点)
      - [3.5.3. 静态图的运算功能](#353-静态图的运算功能)
      - [3.5.4. 运算代码优化为数学表达式（利用运算符重载）](#354-运算代码优化为数学表达式利用运算符重载)
      - [3.5.5. 拓扑排序](#355-拓扑排序)
      - [3.5.6. 前向传播](#356-前向传播)
      - [3.5.7. 反向传播](#357-反向传播)
    - [3.6. 动态图](#36-动态图)
      - [3.6.1. 与静态图的区别](#361-与静态图的区别)
      - [3.6.2. 大体思路](#362-大体思路)
      - [3.6.3. 动态图的实现](#363-动态图的实现)
    - [3.7. 网络层](#37-网络层)
      - [3.7.1. 基本布局](#371-基本布局)
      - [3.7.2. Module 类](#372-module-类)
      - [3.7.3. 权重类 parameter.py](#373-权重类-parameterpy)
      - [3.7.4. 以 linear 为例](#374-以-linear-为例)

<!-- /TOC -->


## 3. 开发文档

### 3.1. 基本框架

![](https://github.com/Eric-is-good/NumTorch/blob/main/imgs/1.png)



### 3.2. 计算图前向传播

![](https://github.com/Eric-is-good/NumTorch/blob/main/imgs/2.png)



### 3.3. 计算图反向传播

![](https://github.com/Eric-is-good/NumTorch/blob/main/imgs/3.png)

### 3.4. 自动微分

![](https://img-blog.csdnimg.cn/20190312135917103.png)

上图写作 （v，dv/dx）最合适



### 3.5. 静态图

静态图我们仿照 tensorflow

#### 3.5.1. 基本图

从基本的数据结构开始，定义基本计算图和计算图节点：

```python
from random import randint

class NaiveGraph:
    
    node_list = [] # 图节点列表
    id_list = []   # 节点ID列表
    
    class Node:
        def __init__(self) -> None:
            # 生成唯一的节点id
            while True:
                new_id = randint(0, 1000)
                if new_id not in NaiveGraph.id_list:
                    break
            self.id: int = new_id

            self.next = list() # 节点指向的节点列表
            self.last = list() # 指向节点的节点列表
            self.in_deg, self.in_deg_com = 0, 0 # 节点入度
            self.out_deg, self.out_deg_com = 0, 0 # 节点出度
            NaiveGraph.add_node(self)
            
        def build_edge(self, node):
            # 构建self节点与node节点的有向边
            self.out_deg += 1
            node.in_deg += 1
            self.next.append(node)
            node.last.append(self)

    @classmethod
    def add_node(cls, node):
        # 在计算图中加入节点
        cls.node_list.append(node)
        cls.id_list.append(node.id)

    @classmethod
    def clear(cls):
        # 刷新计算图
        cls.node_list.clear()
        cls.id_list.clear()
```

#### 3.5.2. node节点

Tensorflow中有三种变量：

1. 常量（Constant）；
2. 变量（Variable）；
3. 占位符（Placeholder）。

常量不存在导数，求导通常是对变量和占位符去求，而占位符通常是神经网络的数据输入口，我们借鉴TensorFlow，使用`feed_dict`方法对占位符进行赋值。我们模仿这样的设计方法，设计三个`Node`类的派生类：

```python
class Constant(Node):
    def __init__(self, value) -> None:
        super().__init__()
        # 注意到Constant节点的值是私有变量，表示不可更改，且没有梯度变量。
        self.__value = float(value)

    def get_value(self):
        return self.__value

    def __repr__(self) -> str:
        return str(self.__value)

class Variable(Node):
    def __init__(self, value) -> None:
        super().__init__()
        self.value = float(value)
        self.grad = 0.

    def get_value(self):
        return self.value

    def __repr__(self) -> str:
        return str(self.value)

class PlaceHolder(Node):
    def __init__(self) -> None:
        super().__init__()
        self.value = None
        self.grad = 0.

    def get_value(self):
        return self.value

    def __repr__(self) -> str:
        return str(self.value)
```



#### 3.5.3. 静态图的运算功能

上面的三种节点都是独立的，需要通过运算进行连接，所以我们定义Operator类：

**运算不仅是连接，同时也还是一个 *节点* ，所以Operator类可以继承自Variable。**

```python
class Operator(Variable):
    def __init__(self, operator: str) -> None:
        super().__init__(0)
        self.operator = operator
        self.calculate = NaiveGraph.operator_calculate_table[operator]

    def __repr__(self) -> str:
        return self.operator
```

注意到Operator节点，比如加法节点，乘法节点，也是有值有梯度的，所以Operator类可以继承自Variable。节点的运算符，我们用`self.operator`字符串存储，而`self.calculate`是`self.operator`对应的函数，比如，如果`self.operator`为`"add"`，那么`self.calculate`是一个将`self.last`中节点值类加的函数。`operator_calculate_table`是一个字典，存储运算符字符串：运算函数的键值对：

```python
from math import prod
from math import exp as math_exp, log as math_log
from math import sin as math_sin, cos as math_cos

operator_calculate_table = {
    "add": lambda node: sum([last.get_value() for last in node.last]),
    "mul": lambda node: prod([last.get_value() for last in node.last]),
    "div":
    lambda node: node.last[0].get_value() / node.last[1].get_value(),
    "sub":
    lambda node: node.last[0].get_value() - node.last[1].get_value(),
    "exp": lambda node: math_exp(node.last[0].get_value()),
    "log": lambda node: math_log(node.last[0].get_value()),
    "sin": lambda node: math_sin(node.last[0].get_value()),
    "cos": lambda node: math_cos(node.last[0].get_value()),
}
```

示例

```python
constant = NaiveGraph.Constant
Variable = NaiveGraph.Variable

x = Constant(1, name='x')
y = Varialbe(2, name='y')
```

我们想计算加法，我们会新建一个加法节点，然后构建分别从`x`和`y`指向该加法节点的有向边：

```python
add = Operator("add")
x.build_edge(add)
y.build_edge(add)
```

我们这里只建图，不计算，这是基于TensorFlow1的想法。计算步骤会在建图完成后交给前向传播来进行。至于一元运算，则是新建特定的运算节点，然后构建有向边：

```python
exp = Operator("exp")
x.build_edge(exp)
```



#### 3.5.4. 运算代码优化为数学表达式（利用运算符重载）

如果对每一个运算都要重复地写上面的代码，未免过于麻烦，因此我们设计了三个函数模板：

1. 一元函数模板；
2. 二元函数模板；
3. 可结合的二元函数模板。

先看一元函数：

```python
@classmethod
def unary_function_frame(cls, node, operator):
    if not isinstance(node, NaiveGraph.Node):
        node = NaiveGraph.Constant(node)
    node_operator = NaiveGraph.Operator(operator)
    node.build_edge(node_operator)
    return node_operator
```

这里我们添加了一个前置操作，如果操作数的类型不是节点，而是一般的数字，比如

```python
exp = NaiveGraph.exp(3.)
```

我们会将其转化为Constant节点。后面的操作我们在前面提到了，这里不再赘述。接着是一般的二元函数，类似的，我们会先将不是计算图节点的操作数进行类型转换：

```python
@classmethod
def binary_function_frame(cls, node1, node2, operator):
    '''
    一般的二元函数框架
    '''
    if not isinstance(node1, NaiveGraph.Node):
        node1 = NaiveGraph.Constant(node1)
    if not isinstance(node2, NaiveGraph.Node):
        node2 = NaiveGraph.Constant(node2)
    node_operator = NaiveGraph.Operator(operator)
    node1.build_edge(node_operator)
    node2.build_edge(node_operator)
    return node_operator
```

最后是可结合的二元函数，针对的是加法和乘法（也就是连加和连乘）：

利用结合律，把 2n-1 个节点变成汇聚的 n+1 个节点

```python
@classmethod
def commutable_binary_function_frame(cls, node1, node2, operator):
    if not isinstance(node1, NaiveGraph.Node):
        node1 = NaiveGraph.Constant(node1)
    if not isinstance(node2, NaiveGraph.Node):
        node2 = NaiveGraph.Constant(node2)

    if isinstance(
            node1,
            NaiveGraph.Operator,
    ) and node1.operator == operator:
        node2.build_edge(node1)
        return node1
    elif isinstance(
            node2,
            NaiveGraph.Operator,
    ) and node2.operator == operator:
        node1.build_edge(node2)
        return node2
    else:
        node_operator = NaiveGraph.Operator(operator)
        node1.build_edge(node_operator)
        node2.build_edge(node_operator)
        return node_operator
```

在写好框架以后，每当新增一个一元or二元运算，我们只需要在`NaiveGraph`中加一个类方法，同时在函数表中注册对应的计算方法即可（实际上还有一项任务，即在导函数表中注册对应的求导方法），比如加法，减法和指数运算：

```python
@classmethod
def add(cls, node1, node2):
    return NaiveGraph.associative_binary_function_frame(
        node1, node2, "add")

@classmethod
def sub(cls, node1, node2):
    return NaiveGraph.binary_function_frame(node1, node2, "sub")

@classmethod
def exp(cls, node1):
    return NaiveGraph.unary_function_frame(node1, "exp")
```

函数表中加上对应项即可。

为了让代码更简洁，我们实现了**节点类的运算符重载**，比如加法：

```python
class Node:
    def __add__(self, node):
        return NaiveGraph.add(self, node)
    
    def __radd__(self, node):
        return NaiveGraph.add(node, self)
```

这样，下面的代码会直接创建上图的加法节点：

```python
x = Variable(1, 'x')
y = Variable(2, 'y')
z = Variable(3, 'z')
s = x + y + z
```



#### 3.5.5. 拓扑排序

拓扑排序要解决的问题是给一个图的所有节点排序。

1. 找一个入度为零（不需其他关卡通关就能解锁的）的端点，如果有多个，则从编号小的开始找；
2. 将该端点的编号输出；
3. 将该端点删除，同时将所有由该点出发的有向边删除；
4. 循环进行 2 和 3 ，直到图中的图中所有点的入度都为零；
5. 拓扑排序结束；



#### 3.5.6. 前向传播

在构建好计算图之后，我们就可以求值。因为一个节点求值仅当它的操作数节点求值完成，因此使用 [**拓扑排序**](https://oi-wiki.org/graph/topo/) （广度优先）进行这个过程。

```python
@classmethod
def forward(cls):
    node_queue = [] # 节点队列
    
    for node in cls.node_list:
        # 入度为0的节点入队
        if node.in_deg == 0:
            node_queue.append(node)

    while len(node_queue) > 0:
        node = node_queue.pop() # 弹出最后一个
        for next_node in node.next:
            next_node.in_deg -= 1
            next_node.in_deg_com += 1
            if next_node.in_deg == 0:
                next_node.value = next_node.calculate(next_node)
                node_queue.insert(0, next_node)  # 插入到第一个

    for node in cls.node_list:
        node.in_deg += node.in_deg_com
        node.in_deg_com = 0
```

拓扑排序中，我们会不断删去入边，如果节点的入度为0，那么它就必须是求好值的。我们对入度是就地更改的，所以我们设计`in_deg_com`变量对修改的入度进行记录，在排序完成后再还原入度。



#### 3.5.7. 反向传播

大体框架和前向传播相同，区别有：

- 我们设置出度为0的节点，即输出节点的梯度为1，其他节点的梯度都设为0；
- 如果计算图的节点有多个输出，那么求导时应当将这些**输出对应的梯度相加**，这也是为什么PyTorch等框架在每次反向传播前都需要将梯度清零的原因。

```python
@classmethod
def backward(cls):
    node_queue = []
    for node in cls.node_list:
        if node.out_deg == 0 and not isinstance(
                node,
                NaiveGraph.Constant,
        ):
            node.grad = 1.
            node_queue.append(node)
    if len(node_queue) > 1:
        print('''
            计算图中的函数是多元输出，自动微分会计算梯度的和，
            如果要求指定输出的导数，应该是用backward_from_node。
            ''')

    while len(node_queue) > 0:
        node = node_queue.pop()
        for last_node in node.last:
            last_node.out_deg -= 1
            last_node.out_deg_com += 1
            if last_node.out_deg == 0 and not isinstance(
                    last_node,
                    NaiveGraph.Constant,
            ):  # 找到上一层的节点求导
                for n in last_node.next:
                    assert n.operator != None
                    # df1/dl = df1/dn * dn/dl
                    last_node.grad += n.grad * cls.__deriv(n, last_node)
                node_queue.insert(0, last_node)

    for node in cls.node_list:
        node.out_deg += node.out_deg_com
        node.out_deg_com = 0
```

`cls.__deriv`是一个查表函数，表中是对应函数的导数：

```python
@classmethod
def __deriv(cls, child: Operator, parent: Node):
    return {
        "add": cls.__deriv_add,
        "sub": cls.__deriv_sub,
        "mul": cls.__deriv_mul,
        "div": cls.__deriv_div,
        "exp": cls.__deriv_exp,
        "log": cls.__deriv_log,
    }[child.operator](child, parent)
```

所以我们还设计了一个机制，即**对某一个特定节点进行求导**（即 pytorch 里面的 x.backward() ），它只有的入队代码和上面的反向传播有区别：

```python
 @classmethod
def backward_from_node(cls, y):
    assert type(y) != cls.Constant, "常量无法求导"

    node_queue = []
    for node in cls.node_list:
        if node.out_deg == 0 and not isinstance(
                node,
                cls.Constant,
        ):
            if node == y:
                node.grad = 1.
            else:
                node.grad = 0.
            node_queue.append(node)
    
    while len(node_queue) > 0:
        ...
```

我们为Variable类等提供这样的接口，这样的写法就挺像Pytorch了：

```python
class Variable(Node):
    def backward(self):
        NaiveGraph.backward_from_node(self)
```

由此，我们可以实现 **Jacobi矩阵** 的求解函数：

```python
@classmethod
def jacobi(cls, y_list: list, x_list: list):
    j = []
    for y in y_list:
        NaiveGraph.zero_grad()
        y.backward()
        j.append([
            x.grad if type(x) != NaiveGraph.Constant else 0.
            for x in x_list
        ])
    return j
```



#### 非标量的反向传播

![](https://img-blog.csdnimg.cn/20200509110539884.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0hhcHBpbmVzc1NvdXJjZUw=,size_16,color_FFFFFF,t_70)





### 3.6. 动态图

#### 3.6.1. 与静态图的区别

静态计算图的宗旨是先建图再计算。这里的计算包括前向传播和反向传播。深度学习框架，比如TensorFlow1和Theano都采用的是静态图。静态图的一个很反直觉的设定就是，调用计算函数后，用户无法得到计算的结果，因为**这种计算函数的目的是建图，而不是计算**。所有的计算必须要等到最后的前向传播才能进行，以我们在前面设计的计算图为例：

```python
a = NaiveGraph.Variable(1., name='a')
b = NaiveGraph.Variable(2., name='b')
c = a + b
print(c.get_value()) # 此处c不是3，因为没有前向传播
NaiveGraph.forward()
print(c.get_value()) # 此处c才是3，因为已经前向传播
```

我们希望`c`在计算后能够马上得到结果。这就是动态计算图的思想。

我们能够在PyTorch看到这种操作：

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        ...

    def forward(self, x):
        x = F.relu(self.conv1(x))
        ...

net = Net()
```

接着在训练中多次调用`net.forward`方法，如果是静态计算图，`net.forward`实际上只是建图函数（因为只有计算过程），**重复调用会创建多个重复的节点**。但在动态计算图中，`net.forward`会在**构建计算图后马上求值**。剩下的问题是，按我们前面说的，动态计算图的求值仍逃不过“建图”这个步骤，而一旦多次求值，计算图中仍然会出现重复的冗余节点。这就涉及到动态计算图所谓“动态”的第二个特性： ***反向传播之后销毁计算图***。所以对于训练神经网络而言，静态计算图的步骤是：

- 一次建图+n次前向传播+n次反向传播；

而动态计算图则是：

- n次 建图 / 前向传播+n次反向传播。

所以PyTorch比TensorFlow更灵活，更Pythonic；而TensorFlow难懂，但效率高。这也解释了“学术用PyTorch，企业用TensorFlow”这句话的由来。



#### 3.6.2. 大体思路

下面就是动态计算图的实现方面的探究。

![dynamic](https://github.com/Eric-is-good/NumTorch/blob/main/imgs/4.gif)

上图是PyTorch建图并求值，然后反向传播的代码与流程。动图的循环播放可以视作 “前向传播——反向传播” 的循环。可以发现有两种变量一直存在，一种是用户自定义的输入变量，一种是我们要更新的参数。除此之外，运算过程中产生的节点，我们都会进行删除。

我们考虑PyTorch中Tensor的`requires_grad`属性，表明该张量是否需要进行求导，默认是False。**我们可以将需要求导的张量视为变量，不需要求导的张量视为常量**。由此，PyTorch中的一个很好理解的规定是：一个运算，它的参数（操作数）只要有一个的`requires_grad`为True，那么运算结果也是需要求导的张量；如果参数（操作数）全部不需要求导，那么它的运算结果也不需要求导。

在反向传播的过程中，梯度流不需要流到不需要求导的节点。从这点考虑，PyTorch不会将这些节点放到计算图中。于是，在我们日常使用PyTorch时，类似下面的语句对计算图无影响：

```python
import torch

x = torch.zeros((1, 2))
y = torch.ones((1, 2))
z = x + y
```

所以，PyTorch在反向传播后，对计算图进行销毁，实际上是将满足下列条件的节点删除

1. `requires_grad`标签为True（实际上是节点在计算图中的必要条件）；
2. 由运算产生，即存在指向该节点的其他节点。

这里的**删除**包括：

1. 将该节点踢出计算图节点集合；
2. 删除该节点的前驱和后继信息。

这样它成为**孤立节点，只是一个存储值的节点。如果我们对它的值感兴趣，完全可以将它的值用在后面的运算中，甚至以新节点的身份重返计算图**。

最后再提一下PyTorch反向传播的一个细节。默认的反向传播后即销毁计算图的机制，在多轮训练神经网络等场景下显得累赘。PyTorch为节点的`backward`方法提供`retain_graph`参数，当它为True时，反向传播后会保留计算图，而不去销毁。比如下面代码

```python
import torch

x = torch.randn(3, 4, requires_grad=True)
y = x**2
output = y.mean()
output.backward()
output.backward()
```

在执行第二次反向传播时会报错：

```
RuntimeError: Trying to backward through the graph a second time, 
but the saved intermediate results have already been freed.
Specify retain_graph=True when calling .backward() 
or autograd.grad() the first time.
```

这就是因为计算图已经被销毁了，而一旦指定`retain_graph`参数为True：

```python
import torch

x = torch.randn(3, 4, requires_grad=True)
y = x**2
output = y.mean()
output.backward(retain_graph=True)
output.backward()
```

程序则不会报错。我们在后面会尝试模仿PyTorch的该机制。



#### 3.6.3. 动态图的实现

在静态计算图的基础上，我们打算实现一个基于标量的动态计算图，语法上更靠近PyTorch。首先是节点的设计，这里我们不再将节点区分为常量，变量和占位符了，而是统一的Node类：

```python
class Node:
    def __init__(self, value, requires_grad=False) -> None:
        # 生成唯一的id
        while True:
            new_id = randint(0, 1000)
            if new_id not in NaiveGraph.id_list:
                break
        self.id: int = new_id
        self.value = float(value)
        self.requires_grad = requires_grad
        # grad和grad_fn分别是节点梯度和节点对应的求导函数
        # 借鉴PyTorch
        self.grad = 0. if self.requires_grad else None
        self.grad_fn = None
        # 默认是操作符名，该属性为绘图需要
        self.name = None
        self.next = list()
        self.last = list()
        # 由于不需要前向传播，所以入度属性被淘汰
        self.out_deg, self.out_deg_com = 0, 0
        if self.requires_grad:
            # 不需要求梯度的节点不出现在动态计算图中
            NaiveGraph.add_node(self)
```

接着是计算函数的改变，我们坚持了模块化的思想，继续沿用一元函数和二元函数框架：

```python
@classmethod
def unary_function_frame(cls, node, operator: str):
    if type(node) != cls.Node:
        node = cls.Node(node)

    # grad_fn_table是一个字符串——函数元组字典，元组中是求值函数和求导函数
    fn, grad_fn = cls.grad_fn_table.get(operator)
    # 这里fn(node)说明我们直接计算输出，即动态计算图的特征
    operator_node = cls.Node(fn(node), node.requires_grad)
    operator_node.name = operator
    if operator_node.requires_grad:
        # 可求导的节点才可有grad_fn成员
        operator_node.grad_fn = grad_fn
        # 只有可求导的变量间才会用有向边联系
        node.build_edge(operator_node)
    return operator_node

@classmethod
def binary_function_frame(cls, node1, node2, operator: str):
    if type(node1) != cls.Node:
        node1 = cls.Node(node1)
    if type(node2) != cls.Node:
        node2 = cls.Node(node2)

    fn, grad_fn = cls.grad_fn_table.get(operator)
    # 两个输入只要有一个是变量，输出就是变量
    requires_grad = node1.requires_grad or node2.requires_grad
    operator_node = cls.Node(
        fn(node1, node2), # 直接计算
        requires_grad=requires_grad,
    )
    operator_node.name = operator
    if requires_grad:
        operator_node.grad_fn = grad_fn
        node1.build_edge(operator_node)
        node2.build_edge(operator_node)
    return operator_node
```

最后是反向传播步骤，我们抛弃了之前那种基于整个图的反向传播，而是只留出针对特定节点的接口，所以`backward`成为了Node类的成员方法：

```python
def backward(self, retain_graph=False):
    if self not in NaiveGraph.node_list:
        print("AD failed because the node is not in graph")
        return

    node_queue = []
    self.grad = 1.

    for node in NaiveGraph.node_list:
        if node.requires_grad:
            if node.out_deg == 0:
                node_queue.append(node)

    while len(node_queue) > 0:
        node = node_queue.pop()
        for last_node in node.last:
            last_node.out_deg -= 1
            last_node.out_deg_com += 1
            if last_node.out_deg == 0 and last_node.requires_grad:
                # 加入节点是需要求导的这一条件
                for n in last_node.next:
                    last_node.grad += n.grad * n.grad_fn(n, last_node)
                node_queue.insert(0, last_node)

    if retain_graph:
        # 保留图
        for node in NaiveGraph.node_list:
            node.out_deg += node.out_deg_com
            node.out_deg_com = 0
    else:
        # 释放计算图：删除所有非叶子节点
        new_list = [] # 新的计算图节点集
        for node in NaiveGraph.node_list:
            if len(node.last) == 0:
                # is leaf
                new_list.append(node)
            else:
                # 清除节点信息
                node.next.clear()
                node.last.clear()
                node.in_deg = 0
            node.next.clear()
            node.out_deg = 0
            node.out_deg_com = 0
        NaiveGraph.node_list = new_list
```



### 3.7. 网络层

#### 3.7.1. 基本布局

在 NumTorch.nn 包里面实现，具体如下

```powershell
C:.
│   functional.py
│   init.py
│   parameter.py
│   __init__.py
│
└───modules
        activation.py
        batchnorm.py
        conv.py
        dropout.py
        linear.py
        loss.py
        module.py
        rnn.py
        __init__.py

```

在原版 pytorch 中，所有 网络层继承 Module 类，该类整理了权重为列表，并使得网络可以成树状组建，**提供了神经网络的统一连接和搭建方法**

```python
class Module:
    r"""Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.

    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child.

    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    """
```

pytorch中，functional.py 中存放着调用 c 的接口，而各个函数的 python 部分在 modules 下完成，下面为原版 pytorch

```python
# linear.py 调用 functional.py 

import functional as F
def forward(self, input: Tensor) -> Tensor:
    return F.linear(input, self.weight, self.bias)


# functional.py 中的 F.linear

linear = _add_docstr(
    torch._C._nn.linear,
    r"""
linear(input, weight, bias=None) -> Tensor 
...  """
    

```

pytorch 在 parameter.py 中将 tensor 封装成 **网络权重**

```python
# parameter.py
class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.
    """

   
# 在 linear.py 中初始化调用    
self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
```

pytorch 在 init.py 里面提供了工具类，例如初始化权重的算法

```python
def kaiming_uniform_(
    tensor: Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
):
```

我们将模仿这个过程



#### 3.7.2. Module 类

```python
class Module:
    def __init__(self) -> None:
        self._train = True
        self._parameters = OrderedDict()   # 有序字典装权重表

    def __call__(self, *x) -> Tensor:
        return self.forward(*x)

    def __setattr__(self, __name: str, __value) -> None:   # 属性赋值，重载是为了将权重同步到有序字典里
        self.__dict__[__name] = __value
        if isinstance(__value, Parameter):
            self._parameters[__name] = __value
        if isinstance(__value, Module):
            for key in __value._parameters:
                self._parameters[__name + "." + key] = __value._parameters[key]

    def __repr__(self) -> str:
        module_list = [
            module for module in self.__dict__.items()
            if isinstance(module[1], Module)
        ]
        return "{}(\n{}\n)".format(
            self.__class__.__name__,
            "\n".join([
                "{:>10} : {}".format(module_name, module)
                for module_name, module in module_list
            ]),
        )

    def parameters(self):
        for param in self._parameters.values():
            yield param

    def train(self, mode: bool = True):
        set_grad_enabled(mode)
        self.set_module_state(mode)

    def set_module_state(self, mode: bool):
        self._train = mode
        for module in self.__dict__.values():
            if isinstance(module, Module):
                module.set_module_state(mode)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def eval(self):
        return self.train(False)
```



#### 3.7.3. 权重类 parameter.py

在 parameter.py 里面 创建 **权重类**，其实就是一个带梯度的 tensor

```python
class Parameter(Tensor):
    def __init__(self, data: Tensor) -> None:
        super().__init__(data.data, True, float)

    def __repr__(self) -> str:
        return "Parameter : {}".format(self.data)
```



#### 3.7.4. 以 linear 为例

在 linear.py 里初始化，并调用 parameter.py 初始化权重，调用 functional.py 进行前向传播

```python
class Linear(Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(empty((self.in_features, self.out_features)))   # 调用 parameter.py
        self.bias = Parameter(empty(self.out_features)) if bias else None
        self.reset_paramters()

    def reset_paramters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight, self.bias)           # 调用 functional.py 

    def __repr__(self) -> str:
        return "Linear(in_features={}, out_features={}, bias={})".format(
            self.in_features, self.out_features, self.bias is not None)
```



functional.py 中前向传播~~（我们没有 c 语言库）~~

```python
def linear(x: tensor.Tensor, weight: tensor.Tensor, bias: tensor.Tensor):
    affine = x @ weight
    if bias is not None:
        affine = affine + bias
    return affine
```



### 使用框架开发

我们已经完成了所有基本框架，我们再次基础上二次开发，实现一些应用层面的东西，例如一些常用神经网络，一些常用优化器，一些常用批处理 trick。

#### 

#### 卷积神经网络

![](https://pic3.zhimg.com/v2-7fce29335f9b43bce1b373daa40cccba_b.webp)

卷积本身很简单，最简单的方法就是遍历移动，就两个 for 循环，但这显然不符合并行化。

卷积并行化的常用方法是 **im2col+GEMM**。使用 im2col 转化为矩阵乘法，再用 GEMM 优化。

常规的卷积操作为：

<img src="https://img-blog.csdnimg.cn/20190109205748603.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyOTk4NTkz,size_16,color_FFFFFF,t_70" style="zoom: 67%;" />

<img src="https://img-blog.csdnimg.cn/20190109205814109.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyOTk4NTkz,size_16,color_FFFFFF,t_70" style="zoom:67%;" />

转化

<img src="https://img-blog.csdnimg.cn/2019010920593847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzMyOTk4NTkz,size_16,color_FFFFFF,t_70" style="zoom:67%;" />

此时的卷积操作就可转化为矩阵乘法：

<img src="https://img-blog.csdnimg.cn/20200215213604226.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzIwODgwNDE1,size_16,color_FFFFFF,t_70" style="zoom:67%;" />



举个例子来说明 GEMM 矩阵乘法优化算法

1.直接暴力矩阵乘法：

```c++
for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            C[m][n]+= A[m][k] * B[k][n];
        }
    }
}
```

上述公式总计算量为2MNK (其中 𝑀、𝑁、𝐾 分别指代三层循环执行的次数，2 指代循环最内层的一次乘法和加法） ，内存访问操作总数为 4MNK（其中 4 指代对c取存和取a，b）。GEMM 的优化均以此为基点。



2.将输出的计算拆分为 1×4 的小块，即将 𝑁 维度拆分为两部分。计算该块输出时，需要使用 𝐴 矩阵的 1 行，和 𝐵 矩阵的 4 列。

每次入第 i 个（列）黄和第 i 行绿（看起来和正常矩阵运算刚刚相反）

![](https://pic2.zhimg.com/80/v2-a4a3ba5a21012600872e1f3f7e15bc59_720w.webp)

```c++
for (int m = 0; m < M; m++) {
  for (int n = 0; n < N; n += 4) {
    C[m][n + 0] = 0;
    C[m][n + 1] = 0;
    C[m][n + 2] = 0;
    C[m][n + 3] = 0;
    for (int k = 0; k < K; k++) {
      C[m][n + 0] += A[m][k] * B[k][n + 0];
      C[m][n + 1] += A[m][k] * B[k][n + 1];
      C[m][n + 2] += A[m][k] * B[k][n + 2];
      C[m][n + 3] += A[m][k] * B[k][n + 3];
    }
  }
}
```

简单的观察即可发现，上述伪代码的最内侧计算使用的矩阵 𝐴 的元素是一致的。因此可以将 𝐴[𝑚] [𝑘] 读取到寄存器中，从而实现 4 次数据复用。一般将最内侧循环称作计算核（micro kernel）。进行这样的优化后，内存访问操作数量变为 （3 + 1/4）MNK。（取 a 变成了原来的 1/4）



3.同理操作b, 每次入第 i 列黄和第 i 行绿，一共k次

![](https://pic3.zhimg.com/80/v2-73afa75e91d9a66a3d194db36f994232_720w.webp)

```c++
for (int m = 0; m < M; m += 4) {
  for (int n = 0; n < N; n += 4) {
    C[m + 0][n + 0..3] = 0;
    C[m + 1][n + 0..3] = 0;
    C[m + 2][n + 0..3] = 0;
    C[m + 3][n + 0..3] = 0;
    for (int k = 0; k < K; k++) {
      C[m + 0][n + 0..3] += A[m + 0][k] * B[k][n + 0..3];
      C[m + 1][n + 0..3] += A[m + 1][k] * B[k][n + 0..3];
      C[m + 2][n + 0..3] += A[m + 2][k] * B[k][n + 0..3];
      C[m + 3][n + 0..3] += A[m + 3][k] * B[k][n + 0..3];
    }
  }
}
```

访存变为 2 +1/4 +1/4 =(2+1/2) MNK

访存可以无限接近于 2 MNK

又可以把 c 移出去 ，所以可以优化到接近 2MN + 1/2 MNK = 1/2 MNK



