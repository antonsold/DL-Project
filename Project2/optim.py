import torch


class SGD:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for w, dw in self.parameters:
            w.sub_(dw.mul(self.learning_rate))
