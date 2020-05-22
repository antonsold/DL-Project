import torch
import module as md
import model as m
import generate as gen
torch.set_grad_enabled(False)
torch.manual_seed(123)


sample_size = 1000


train_input, train_target = gen.generate_set(sample_size)
test_input, test_target = gen.generate_set(sample_size)


layers = [md.Linear(2, 25), md.ReLU(), md.Linear(25, 25), md.ReLU(), md.Linear(25, 25),
          md.ReLU(), md.Linear(25, 2), md.Tanh()]
model = m.Model(layers)


m.train_model(model, train_input, train_target)
print("Train accuracy: %f" % m.accuracy(train_target, model(train_input)))
print("Test accuracy: %f" % m.accuracy(test_target, model(test_input)))
