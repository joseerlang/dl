from .metrics import Metrics
from fastprogress import master_bar, progress_bar
import torch

default_device = "cuda" if torch.cuda.is_available() else "cpu"


class Model():

    def __init__(self, net, device=None):
        self.net = net
        self.device = device or default_device

    def compile(self, optimizer, loss, metrics=[], scheduler=None, precision=5):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.precision = precision
        self.scheduler = scheduler

    def train_bar(self, dataloader, device=None):
        device = device or self.device
        self.net.to(device)
        self.net.train()
        metrics = Metrics(self.metrics, precision=self.precision)
        for batch in progress_bar(dataloader, parent=self.mb):
            X, y = batch
            X, y = X.to(device), y.to(device)
            self.optimizer.zero_grad()
            y_hat = self.net(X)
            loss = self.loss(y_hat, y)
            loss.backward()
            self.optimizer.step()
            metrics.compute(loss.item(), y_hat, y)
            self.mb.child.comment = metrics.compute_average()
        return metrics.compute_average()

    def eval_bar(self, dataloader, device=None):
        device = device or self.device
        self.net.to(device)
        self.net.eval()
        metrics = Metrics(self.metrics, prefix="eval_",
                          precision=self.precision)
        with torch.no_grad():
            for batch in progress_bar(dataloader, parent=self.mb):
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = self.net(X)
                loss = self.loss(y_hat, y)
                metrics.compute(loss.item(), y_hat, y)
                self.mb.child.comment = metrics.compute_average()
            return metrics.compute_average()

    def fit(self, dataloader, eval_dataloader=None, epochs=100, device=None):
        self.init_history(eval=eval_dataloader)
        self.mb = master_bar(range(1, epochs+1))
        eval_metrics = []
        end = False
        for epoch in self.mb:
            train_metrics = self.train_bar(dataloader, device)
            if eval_dataloader:
                eval_metrics = self.eval_bar(eval_dataloader, device)
                end = self.early_stopping(epoch, eval_metrics)
            self.update_history(
                epoch, self.optimizer.param_groups[0]['lr'], train_metrics, eval_metrics)
            self.mb.write(self.get_text(
                epoch, epochs, train_metrics, eval_metrics))
            self.scheduler_step(epoch, eval_metrics)
            if end:
                break
        return self.history

    def init_history(self, eval):
        self.history = {"epochs": [], "lr": [], "metrics": {}}
        train_metrics = Metrics(self.metrics)
        eval_metrics = {}
        if eval:
            eval_metrics = Metrics(self.metrics, prefix="eval_")
        for m in train_metrics:
            self.history["metrics"][m] = []
        for m in eval_metrics:
            self.history["metrics"][m] = []

    def update_history(self, epoch, lr, train_metrics, eval_metrics=[]):
        self.history["epochs"].append(epoch)
        self.history["lr"].append(lr)
        for m in train_metrics:
            self.history["metrics"][m].append(train_metrics[m])
        for m in eval_metrics:
            self.history["metrics"][m].append(eval_metrics[m])

    def get_text(self, epoch, epochs, train_metrics, eval_metrics=None):
        text = f"Epoch {epoch}/{epochs}"
        text += f" {train_metrics}"
        if eval_metrics:
            text += f" {eval_metrics}"
        return text

    def evaluate(self, dataloader, device=None):
        self.mb = master_bar(range(0, 1))
        for _ in self.mb:
            eval_metrics = self.eval_bar(dataloader, device)
            self.mb.write(str(eval_metrics))

    def predict(self, dataloader, device=None, train=False):
        device = device or self.device
        self.net.to(device)
        if train:
            self.net.train()
        else:
            self.net.eval()
        with torch.no_grad():
            preds = torch.tensor([]).to(device)
            for batch in progress_bar(dataloader):
                X = batch
                X = X.to(device)
                pred = self.net(X)
                preds = torch.cat([preds, pred])
            return preds

    def early_stopping(self, epoch, metrics):
        return False

    def scheduler_step(self, epoch, metrics):
        return
