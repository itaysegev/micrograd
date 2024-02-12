import random
from micrograd.engine import Value


class Module:
    def zero_grad(self):  # helper function to zero out the gradients of all parameters
        for p in self.parameters():
            p.grad = 0


    @staticmethod
    def parameters():
        return []


class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # learnable weights initialized randomly
        self.b = Value(0)  # learnable bias initialized to 0
        self.nonlin = nonlin  # whether to apply the ReLU nonlinearity

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)  # weight inputs, sum, and add bias
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]  # our parameters consist of the weights and bias

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]  # create nout neurons

    def __call__(self, x):
        out = [n(x) for n in self.neurons]  # apply each neuron to the input
        return out[0] if len(out) == 1 else out  # if we have only one output, return it directly

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  # apply each layer
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"