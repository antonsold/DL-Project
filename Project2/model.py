import torch
import optim as opt
import module as md


class Model:
    def __init__(self, layers):
        self.layers = layers
        self.parameters = []
        for layer in layers:
            self.parameters += layer.parameters

    def __call__(self, input_):
        return self.forward(input_)

    def zero_grad(self):
        for w, dw in self.parameters:
            dw.zero_()

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


def train_model(model, train_input, train_target, batch_size=100, n_epochs=250, loss=md.MSELoss(), learning_rate=0.1, print_loss=True):
    sample_size = train_input.size(0)
    sgd = opt.SGD(model.parameters, learning_rate)
    for epoch in range(n_epochs):
        cumulative_loss = 0
        for n_start in range(0, sample_size, batch_size):
            # resetting the gradients
            model.zero_grad()
            output = model(train_input[n_start : n_start + batch_size])
            # accumulating the loss over the mini-batches
            cumulative_loss += loss(output, train_target[n_start : n_start + batch_size]) * batch_size
            # calculating the gradient of the loss wrt final outputs
            loss_grad = loss.backward(output, train_target[n_start : n_start + batch_size])
            # propagating it backward
            model.backward(loss_grad)
            # updating the parameters
            sgd.step()
        if print_loss:
            print("Epoch: %i" % epoch)
            print("Loss: %f" % (cumulative_loss / sample_size))


def accuracy(true_target, predicted):
    return true_target.argmax(dim=1).sub(predicted.argmax(dim=1)).eq(0).float().mean().item()
