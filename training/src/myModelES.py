from ..metrics import Metrics
from .myModel import MyModel
from fastprogress import master_bar, progress_bar
import torch

default_device = "cuda" if torch.cuda.is_available() else "cpu"


class MyModelES(MyModel):
    """ Model with early stopping. """

    def __init__(self, net, device=None, patience=5, es_from=0, es_metric="eval_loss", mode="min"):
        super().__init__(net, device)
        self.patience = patience
        self.es_from = es_from
        self.es_metric = es_metric
        self.mode = mode

    def fit(self, dataloader, eval_dataloader=None, epochs=100, device=None):
        self.best_metric = 1e10 if self.mode == "min" else 0
        self.step = 0
        self.best_e = 0
        return super().fit(dataloader, eval_dataloader, epochs, device)

    def early_stopping(self, epoch, metrics):
        if epoch >= self.es_from:
            self.step += 1
            metric = metrics[self.es_metric]
            if self.mode == "min" and metric < self.best_metric:
                self.save_best(metric, epoch)
            elif self.mode == "max" and metric > self.best_metric:
                self.save_best(metric, epoch)
            if self.step >= self.patience:
                self.net.load_state_dict(torch.load('best_dict.pth'))
                print(f"training stopped at epoch {epoch}")
                return True
            return False

    def save_best(self, metric, epoch):
        self.best_metric = metric
        torch.save(self.net.state_dict(), 'best_dict.pth')
        self.best_e = epoch
        self.step = 0
        print(
            f"best model found at epoch {self.best_e} with {self.es_metric} {self.best_metric:.5f}")
