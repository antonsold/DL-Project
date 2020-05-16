import torch
import math


class Module(object):
    def __init__(self):
        self.parameters = []

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return self.parameters


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        init_range = 1. / math.sqrt(in_features)
        self.weights = torch.Tensor(in_features, out_features).uniform_(-init_range, init_range)
        self.grad_w = torch.zeros((in_features, out_features))
        self.parameters = [(self.weights, self.grad_w)]
        if bias:
            self.bias = torch.Tensor(out_features).uniform_(-init_range, init_range)
            self.grad_b = torch.zeros(out_features)
            self.parameters.append((self.bias, self.grad_b))
        else:
            self.bias = None

    def forward(self, input_):
        self.input = input_
        if self.bias is not None:
            return torch.addmm(self.bias, input_, self.weights)
        else:
            return input_.matmul(self.weights)

    def backward(self, grad_output):
        self.grad_w += self.input.t().matmul(grad_output)
        grad_input = grad_output.matmul(self.weights.t())
        if self.bias is not None:
            self.grad_b += grad_output.sum(dim=0)
        return grad_input


class ReLU(Module):
    def forward(self, input_):
        self.input = input_
        return torch.relu(input_)

    def backward(self, grad_output):
        return torch.mul((self.input > 0).int(), grad_output)


class Tanh(Module):
    def forward(self, input_):
        self.input = input_
        return torch.tanh(input_)

    def backward(self, grad_output):
        return torch.tanh(self.input).pow(2).mul(-1).add(1).mul(grad_output)


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.layers = list(args)
        for module in args:
            self.parameters += module.parameters

    def forward(self, input_):
        x = input_
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        y = loss_grad
        for layer in reversed(self.layers):
            y = layer.backward(y)
        return


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input_, target):
        return self.forward(input_, target)

    def forward(self, input_, target):
        if target.dim() == 1:
            target = target.view(target.size(0), 1)
        return (input_ - target).pow(2).sum().item()

    def backward(self, input_, target):
        if target.dim() == 1:
            target = target.view(target.size(0), 1)

        return (input_ - target).mul(2)
