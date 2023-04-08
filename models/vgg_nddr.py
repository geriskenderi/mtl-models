import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_absolute_error, median_absolute_error
from torchvision.models import vgg16, vgg16_bn, VGG16_Weights, VGG16_BN_Weights

class NDDRLayer(nn.Module):
    def __init__(self, hidden_dim, task_ids, init_weights=[0.9, 0.1]):
        super().__init__()
        self.task_ids = task_ids
        alpha, beta = init_weights[0], init_weights[1]

        self.layer = nn.ModuleList(
            [nn.Sequential(
                    nn.Conv2d(len(task_ids) * hidden_dim, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim, momentum=0.05),
                    nn.ReLU()
                )
            for _ in self.task_ids])

        # Weight init as in paper
        for i, task in enumerate(self.task_ids):
            layer = self.layer[task]
            t_alpha = torch.diag(torch.FloatTensor([alpha for _ in range(hidden_dim)])) # C x C
            t_beta = torch.diag(torch.FloatTensor([beta for _ in range(hidden_dim)])).repeat(1, len(self.task_ids)) # C x (C x T)
            t_alpha = t_alpha.view(hidden_dim, hidden_dim, 1, 1)
            t_beta = t_beta.view(hidden_dim, hidden_dim * len(self.task_ids), 1, 1)

            # Conv init
            layer[0].weight.data.copy_(t_beta)
            layer[0].weight.data[:,int(i*hidden_dim):int((i+1)*hidden_dim)].copy_(t_alpha)

            # Batchnorm init
            layer[1].weight.data.fill_(1.0)
            layer[1].bias.data.fill_(0.0)

    def forward(self, features):
        features = torch.cat(features, dim=1) # Concat features for all tasks along one dimension
        outputs = [self.layer[task](features) for task in self.task_ids]
        ft_t1, ft_t2 = outputs[0], outputs[1]

        return ft_t1, ft_t2

class NDDRVGG(pl.LightningModule):
    def __init__(self, hidden_dim, output_sizes, dataset_name, feature_extractor, learning_rate=1e-2):
        super(NDDRVGG, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.task_ids = [0,1]

        pretrained_weights = None
        if 'woof' in dataset_name:
            if feature_extractor == 'vgg':
                pretrained_weights = VGG16_Weights.IMAGENET1K_V1
            elif feature_extractor == 'vgg_bn':
                pretrained_weights = VGG16_BN_Weights.IMAGENET1K_V1

        # Build one VGG model for each task.
        vgg_t1, vgg_t2 = None, None
        if feature_extractor == 'vgg':
            vgg_t1 = vgg16(weights=pretrained_weights).features
            vgg_t2 = vgg16(weights=pretrained_weights).features
        elif feature_extractor == 'vgg_bn':
            vgg_t1 = vgg16_bn(weights=pretrained_weights).features
            vgg_t2 = vgg16_bn(weights=pretrained_weights).features

        # We need an nddr layer after each pooling operation. This means we must separate the feature extractors into pieces
        self.vgg_t1 = nn.ModuleList()
        self.vgg_t2 = nn.ModuleList()
        self.nddrs = nn.ModuleList()

        for i, task_net in enumerate([vgg_t1, vgg_t2]):
            temp = []
            children = list(task_net.children())
            for j, module in enumerate(children):
                if module._get_name() == 'MaxPool2d':
                    temp.append(module)
                    task_modulelist = [self.vgg_t1, self.vgg_t2][i]
                    task_modulelist.append(nn.Sequential(*temp))
                    temp = []

                    if i == 0:
                        nddr = NDDRLayer(hidden_dim=children[j-2].weight.shape[0], task_ids=[0,1])
                        self.nddrs.append(nddr)
                else:
                    temp.append(module)

        self.dataset_name = dataset_name
        self.classification_heads = nn.ModuleList()
        for out_dim in output_sizes:
            self.classification_heads.append(
                nn.Sequential(
                    # nn.Linear(512*8*8, hidden_dim),
                    nn.Linear(512*7*7, hidden_dim),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, out_dim)
                )
            )

    def forward(self, x):
        task_logits = []
        for i in range(len(self.nddrs)):
            if i == 0:
                ft1, ft2 = self.vgg_t1[i](x), self.vgg_t2[i](x)
            else:
                ft1, ft2 = self.vgg_t1[i](ft1), self.vgg_t2[i](ft2)

            ft1, ft2 = self.nddrs[i]([ft1, ft2])


        ft1, ft2 = torch.flatten(ft1, start_dim=1), torch.flatten(ft2, start_dim=1)
        logits_t1, logits_t2 = self.classification_heads[0](ft1), self.classification_heads[1](ft2)
        task_logits.append(logits_t1)
        task_logits.append(logits_t2)

        return task_logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {'params': self.vgg_t1.parameters()},
                {'params': self.vgg_t2.parameters()},
                {'params': self.classification_heads.parameters()},
                {'params': self.nddrs.parameters(), 'lr': self.learning_rate*100}
            ],
            lr=self.learning_rate,
            weight_decay=self.learning_rate*1e-2
        )

        return optimizer

    # Learning rate warm-up and decay (override automatic optimization in PyL)
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # update params
        optimizer.step(closure=optimizer_closure)

        # LR decay after 'epoch_decay_start' epochs
        epoch_decay_start = 10
        if epoch > epoch_decay_start:
            lr_scale = ((self.trainer.max_epochs-epoch)+epoch_decay_start)/self.trainer.max_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.learning_rate

    def on_train_epoch_start(self):
        self.train_pred = []
        self.train_gt = []

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        losses = []
        for i in range(2):
            lgs, gt = logits[i], y[:, i]
            loss = F.cross_entropy(lgs, gt)
            self.log(f'train_task{i}_loss', loss, sync_dist=True)
            losses.append(loss)
        mtl_loss = sum(losses)
        self.log('train_loss', mtl_loss, sync_dist=True)

        # Save for evaluation
        self.train_pred.append(logits)
        self.train_gt.append(y)

        # Log learning rate for monitoring
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=True, sync_dist=True)

        return mtl_loss

    def on_train_epoch_end(self):
        self.mtl_evaluation(self.train_pred, self.train_gt, self.training)


    def on_validation_epoch_start(self):
        self.val_pred = []
        self.val_gt = []

    def validation_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        losses = []
        for i in range(2):
            lgs, gt = logits[i], y[:, i]
            loss = F.cross_entropy(lgs, gt)
            self.log(f'val_task{i}_loss', loss, sync_dist=True)
            losses.append(loss)
        mtl_loss = sum(losses)
        self.log('val_loss', mtl_loss, sync_dist=True)

        # Save for evaluation
        self.val_pred.append(logits)
        self.val_gt.append(y)

    def on_validation_epoch_end(self):
        self.mtl_evaluation(self.val_pred, self.val_gt, self.training)

    def mtl_evaluation(self, pred_list, gt_list, training):
        train_str = 'train' if training else 'val'
        for i, task in enumerate(self.task_ids):
            pred, gt = torch.vstack([x[i] for x in pred_list]), torch.hstack([y[:, task] for y in gt_list])

            if self.dataset_name == 'imdb' and task == 0:
                pred = F.softmax(pred, -1)
                age_values = torch.arange(0, 100, 1).to(pred.device)
                pred = (pred * age_values).sum(-1)
                pred, gt = pred.detach().cpu().numpy().flatten(), gt.detach().cpu().numpy().flatten()
                mae = mean_absolute_error(gt, pred)
                made = median_absolute_error(gt, pred)
                self.log(f'{train_str}_task{i}_mae', mae, sync_dist=True)
                self.log(f'{train_str}_task{i}_made', made, sync_dist=True)

                if train_str == 'val':
                    print(f'Task: {task}, MAE: {mae}, MedAE: {made}')
            else:
                pred, gt = pred.argmax(-1).detach().cpu().numpy().flatten(), gt.detach().cpu().numpy().flatten()
                acc = accuracy_score(gt, pred)
                mcc = matthews_corrcoef(gt, pred)
                self.log(f'{train_str}_task{i}_accuracy', acc, sync_dist=True)
                self.log(f'{train_str}_task{i}_mcc', mcc, sync_dist=True)

                if train_str == 'val':
                    print(f'Task: {i}, Acc: {acc}, MCC: {mcc}')