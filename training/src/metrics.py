import numpy as np
import torch


class MyDict(dict):
    """ Custom dict to control how metrics are displayed """

    def __init__(self, precision=5):
        self.precision = precision

    def __str__(self):
        text = ""
        for k, v in self.items():
            if isinstance(v, float):
                text += f" {k} {round(v, self.precision)}"
            else:
                text += f" {k} {v}"
        return text


class Metrics(MyDict):
    """ Class to keep track of metrics """

    def __init__(self, metrics, prefix='', precision=5):
        super().__init__(precision)
        self.metrics = metrics
        self.precision = precision
        self.average = MyDict(precision)
        self.prefix = prefix

        self[f'{self.prefix}loss'] = []
        for m in self.metrics:
            self[f'{self.prefix}{m.name}'] = []

    def compute_average(self):
        for k in self:
            self.average[k] = np.mean(self[k])
        return self.average

    def compute(self, loss, y_hat, y):
        self[f'{self.prefix}loss'].append(loss)
        for metric in self.metrics:
            self[f'{self.prefix}{metric.name}'].append(metric(y_hat, y))


class Accuracy():
    def __init__(self):
        self.name = "acc"

    def __call__(self, y_hat, y):
        return (torch.argmax(y_hat, axis=1) == y).sum().item() / len(y)
