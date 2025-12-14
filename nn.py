# from micrograd.engine import Value

# a = Value(-4.0)
# b = Value(2.0)
# c = a + b
# d = a * b + b**3
# c += c + 1
# c += 1 + c + (-a)
# d += d * 2 + (b + a).relu()
# d += 3 * d + (b - a).relu()
# e = c - d
# f = e**2
# g = f / 2.0
# g += 10.0 / f
# print(f'{g:.4f}') # prints 24.7041, the outcome of this forward pass
# g.backward()
# print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
# print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db



import math 
import numpy as np 
import matplotlib.pyplot as plt

# def f(x):
#     return 3*x**2 -4*x +5
# print(f(3.0))

# xs = np.arange(-5,5,0.25)
# print(xs)
# ys = f(xs)
# print(ys)


# plt.plot(xs, ys)
# plt.show()

# h = 0.00001
# x = 2/3
# print((f(x+h)-f(x))/h)

# let get more complex
# a = 2.0
# b = -3.0
# c = 10.0
# d1 = a*b + c
# c += h 
# d2 = a*b + c
# print('d1', d1)
# print('d2', d2)
# print('slope', (d2 - d1) / h)


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.label = label

    def __repr__(self):
        return f"Value(data {self.data})"
    def __add__(self, other):
        out = Value(self.data + other.data ,  (self, other), _op='+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), _op='*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = ((math.exp(2*x) - 1) / (math.exp(2*x) + 1)) 
        out = Value(t, (self,) , 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


       
# h=0.0001
# a = Value(2.0, label='a')
# b = Value(-3.0, label='b')
# c = Value(10.0, label='c')
# e = a*b
# e.label='e'
# d = e + c
# d.label='d'
# f = Value(-2.0 +h,label='f')
# L = d * f
# L.label='L'
# # print(L)
# L.grad=1.0
# f.grad= 4.0
# d.grad=-2.0
# c.grad= -2.0
# e.grad = -2.0
# a.grad = -2.0 * -3.0
# b.grad = -2.0 * 2

# a.data += 0.01 * a.grad
# b.data += 0.01 * b.grad
# c.data += 0.01 * c.grad
# f.data += 0.01 * f.grad
# e = a*b
# d = e + c
# L = d * f
# print(L.data)






from graphviz import Digraph

def trace(root):
    # builds a set of all notes in the graph
    notes,edges = set(), set()
    def build(v):
        if v not in notes:
            notes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return notes, edges
def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'})

    notes, edges = trace(root)
    for n in notes:
        uid = str(id(n))
        #for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape="record")
        if n._op:
            #is this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label=n._op)
            #and connect this node to it
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        #connect n1 to the op of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

# dot = draw_dot(L)
# dot.render('graph', view=True)  


def lol():
    h=0.001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b
    e.label='e'
    d = e + c
    d.label='d'
    f = Value(-2.0,label='f')
    L = d * f
    L.label='L'    
    L1=L.data

    a = Value(2.0 , label='a')
    b = Value(-3.0 + h, label='b')
    c = Value(10.0, label='c')
    e = a*b
    e.label='e'
    d = e + c 
    d.label='d'
    f = Value(-2.0,label='f')
    L = d * f
    L.label='L' 
    L2=L.data 
    print((L2-L1) / h)


# lol()





#inputs 
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
#weights
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
#bias
b = Value(6.73564758609708 , label='b')
#x1*w1 + x2*w2 + b
x1w1 = x1*w1
x1w1.label = 'x1w1'
x2w2 = x2*w2
x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b
n.label= 'n'

o = n.tanh()
o.label='o'

# o.grad = 1.0
## backward manually
# o._backward()
# n._backward()
# b._backward()
# x1w1x2w2._backward()
# x1w1._backward()
# x2w2._backward()


# n.grad = o.data ** 2
# x1w1x2w2.grad = n.grad * 1
# b.grad = n.grad * 1
# x1w1.grad = b.grad * 1
# x2w2.grad = b.grad * 1
# x1.grad = x1w1.grad * w1.data
# w1.grad = x1w1.grad * x1.data
# x2.grad = x2w2.grad * w2.data
# w2.grad = x2w2.grad * x2.data


#topological & automatic backward
# topo = []
# visited = set()
# def build_topo(v):
#     if v not in visited:
#         visited.add(v)
#         for child in v._prev:
#             build_topo(child)
#         topo.append(v)
# build_topo(o)
# print(topo)
# for i in reversed(topo):
#     i._backward()
#     # print(i)
o.backward()
dot = draw_dot(o)
dot.render('graph', view=True)  
