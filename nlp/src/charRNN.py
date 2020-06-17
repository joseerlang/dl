from fastprogress import master_bar, progress_bar
import torch
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"


class CharRNNModel():
    def __init__(self, net):
        self.net = net

    def compile(self, loss, optimizer, metrics=None):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def train(self, dataloader):
        self.net.train()
        train_loss, train_metrics = [], None
        if self.metrics:
            train_metrics = [[] for m in self.metrics]
        for X, y in progress_bar(dataloader, parent=self.mb):
            X, y = X.to(device), y.to(device)
            self.optimizer.zero_grad()
            output = self.net(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            comment = f'train_loss {np.mean(train_loss):.5f}'
            if self.metrics:
                for i, metric in enumerate(self.metrics):
                    train_metrics[i].append(metric.call(output, y))
                    comment += f' train_{metric.name} {np.mean(train_metrics[i]):.5f}'
            self.mb.child.comment = comment
        return train_loss, train_metrics

    def eval(self, dataloader):
        self.net.eval()
        val_loss, val_metrics = [], None
        if self.metrics:
            val_metrics = [[] for m in self.metrics]
        with torch.no_grad():
            for X, y in progress_bar(dataloader, parent=self.mb):
                X, y = X.to(device), y.to(device)
                output = self.net(X)
                loss = self.loss(output, y)
                val_loss.append(loss.item())
                comment = f'val_loss {np.mean(val_loss):.5f}'
                if self.metrics:
                    for i, metric in enumerate(self.metrics):
                        val_metrics[i].append(metric.call(output, y))
                        comment += f' val_{metric.name} {np.mean(val_metrics[i]):.5f}'
                self.mb.child.comment = comment
        return val_loss, val_metrics

    def fit(self, dataloader, val_dataloader=None, epochs=100, batch_size=32):

        self.history = {"loss": []}
        if self.metrics:
            for metric in self.metrics:
                self.history[f'{metric.name}'] = []
        if val_dataloader:
            self.history["val_loss"] = []
            if self.metrics:
                for metric in self.metrics:
                    self.history[f'val_{metric.name}'] = []

        # training loop
        self.net.to(device)
        self.mb = master_bar(range(1, epochs+1))
        for epoch in self.mb:
            # train
            train_loss, train_metrics = self.train(dataloader)
            self.history["loss"].append(np.mean(train_loss))
            if self.metrics:
                for i, metric in enumerate(self.metrics):
                    self.history[f'{metric.name}'].append(
                        np.mean(train_metrics[i]))
            self.bar_text = f'Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f}'
            if self.metrics:
                for i, metric in enumerate(self.metrics):
                    self.bar_text += f' {metric.name} {np.mean(train_metrics[i]):.5f}'
            # eval
            if val_dataloader:
                val_loss, val_metrics = self.eval(val_dataloader)
                self.history["val_loss"].append(np.mean(val_loss))
                if self.metrics:
                    for i, metric in enumerate(self.metrics):
                        self.history[f'val_{metric.name}'].append(
                            np.mean(val_metrics[i]))
                self.bar_text += f' val_loss {np.mean(val_loss):.5f}'
                if self.metrics:
                    for i, metric in enumerate(self.metrics):
                        self.bar_text += f' val_{metric.name} {np.mean(val_metrics[i]):.5f}'
            # print stats
            self.mb.write(self.bar_text)

        return self.history

    def predict(self, X):
        # X is a tokenized sentence
        self.net.to(device)
        self.net.eval()
        X = torch.tensor([X])
        with torch.no_grad():
            output = self.net(X.to(device))
            return output
