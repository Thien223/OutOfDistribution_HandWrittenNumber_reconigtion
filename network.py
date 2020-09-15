import torch
from torch.nn import Module, Softmax, BatchNorm2d, ReLU, MaxPool2d, Linear, Sigmoid


class CNN__(Module):
    def __init__(self, num_classes=10):
        super(CNN__, self).__init__()

        self.linear1 = Linear(in_features=28*28, out_features=128)
        self.relu1 = ReLU()
        self.linear2 = Linear(in_features=128, out_features=128)
        self.relu2 = ReLU()
        self.fc = Linear(in_features=128, out_features=num_classes)
        self.relu3 = ReLU()
        self.softmax = Softmax(dim=1)

    def forward(self, input):
        output = self.linear1(input)
        output = self.relu1(output)
        output = self.linear2(output)
        output = self.relu2(output)
        output1 = self.fc(output)
        output2 = self.softmax(output1)
        return output1, output2

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.sigmoid = Sigmoid()
    def forward(self, x):
        outputs = self.linear(x)
        # outputs = self.sigmoid(outputs)
        return outputs

