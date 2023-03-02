import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, median_absolute_error
from torchvision.models import vgg16

class MTLVGG(pl.LightningModule):
    def __init__(self, hidden_dim, num_tasks, task_ids, output_sizes, dataset_name, learning_rate=1e-3):
        super(MTLVGG, self).__init__()
        self.save_hyperparameters() 

        self.feature_extractor = vgg16(pretrained=False).features
        self.dataset_name = dataset_name
        self.num_tasks = num_tasks
        self.task_ids = task_ids
        self.learning_rate = learning_rate

        self.classification_heads = nn.ModuleList()
        for task_id in task_ids:
            self.classification_heads.append(
                nn.Sequential(
                    nn.Linear(512*7*7, hidden_dim),
                    nn.Dropout(0.1),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_sizes[task_id])
                )
            )

    def forward(self, x):
        # common features from CNN
        features = self.feature_extractor(x)
        features = torch.flatten(features, start_dim=1)

        task_logits = []
        for i in range(self.num_tasks):
            logits_ti = self.classification_heads[i](features)
            task_logits.append(logits_ti)

        return task_logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.learning_rate*1e-2
        )

        return optimizer

    # learning rate warm-up (override automatic optimization in PyL)
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
        for i, task in enumerate(self.task_ids):
            lgs, gt = logits[i], y[:, task]
            loss = F.cross_entropy(lgs, gt)
            self.log(f'train_task{task}_loss', loss, sync_dist=True)
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
        for i, task in enumerate(self.task_ids):
            lgs, gt = logits[i], y[:, task]
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
                self.log(f'{train_str}_task{task}_mae', mae, sync_dist=True)
                self.log(f'{train_str}_task{task}_made', made, sync_dist=True)

                if train_str == 'val':
                    print(f'Task: {task}, MAE: {mae}, MedAE: {made}')
            else:
                pred, gt = pred.argmax(-1).detach().cpu().numpy().flatten(), gt.detach().cpu().numpy().flatten()        
                acc = accuracy_score(gt, pred)
                f1 = f1_score(gt, pred, average="micro")
                self.log(f'{train_str}_task{task}_accuracy', acc, sync_dist=True)
                self.log(f'{train_str}_task{task}_f1', f1, sync_dist=True)

                if train_str == 'val':
                    print(f'Task: {task}, Acc: {acc}, F1: {f1}')
