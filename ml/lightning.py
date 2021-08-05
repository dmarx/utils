import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pl_bolts
from pl_bolts.datamodules import FashionMNISTDataModule
from torchmetrics import Accuracy, MetricCollection, Precision, Recall

# Modified from @tchaton's style demonstrated here: https://github.com/tchaton/pytorch2lightning 
class BaseModel(pl.LightningModule):
    """General purpose base class with some useful defaults I can override"""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._metrics = MetricCollection([Accuracy(), Precision(), Recall()])
    def _post_init(self):
        #self.save_hyperparameters()
        self._stage_metrics = {
            stage: self._metrics.clone(prefix=f'{stage}_')
            for stage in ['train','val', 'test']}
    def forward(self, x):
        return self.model(x)
    def _loss_func(self, y_hat, y):
        return F.nll_loss(y_hat, y, reduction='sum')
    def _step(self, stage, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self._loss_func(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, logger=True)
        self.log_dict(self._stage_metrics[stage](y_hat, y), prog_bar=True, logger=True)
        return loss
    def training_step(self, batch, batch_idx):
        return self._step('train', batch, batch_idx)
    def validation_step(self, batch, batch_idx):
        return self._step('val', batch, batch_idx)
    def test_step(self, batch, batch_idx):
        return self._step('val', batch, batch_idx)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

if __name__ == '__main__':

    class SimpleClassifier(BaseModel):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.save_hyperparameters() # looks like this doesn't work in a parent class. PR?

            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

            self._post_init()

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output
