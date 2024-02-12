import math


class Value:
    """Stores a single scalar value and its gradient."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # a string rep of the operation that produced this node

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # convert to Value
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # plus is a function of two inputs, so the gradient is 1 in each
            self.grad += out.grad  # dL/dx = dL/dout * dout/dx chain rule
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad  # d/dx x^n = n*x^(n-1)
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data <= 0 else self.data, (self,), 'ReLU')  # define the ReLU operation

        def _backward():
            self.grad += (out.data > 0) * out.grad  # derivative of ReLU is 0 for x<0, else 1
        out._backward = _backward

        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.data**2) * out.grad  # derivative of tanh is (1 - tanh^2)
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topological_lst = []
        visited = set()

        def build_topological(v: Value):
            """Recursive function to build topological ordering of graph nodes."""
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topological(child)
                topological_lst.append(v)
        build_topological(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1  # start with the gradient of the final node which is dL/dL = 1
        for v in reversed(topological_lst):
            v._backward()

    def __neg__(self):  # -self
        return self * -1

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


