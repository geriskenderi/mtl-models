import argparse
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tqdm import tqdm
from pytorch_lightning import seed_everything
from torchvision import transforms as T
from data.imdb import IMDBDataset
from torch.utils.data import DataLoader
from models.vgg import MTLVGG
from models.vgg_nddr import NDDRVGG
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_absolute_error, median_absolute_error

def main(args):
    # Seed 
    seed_everything(args.seed)

    # Define dataset and dataloaders
    data_path = Path(args.data_path)
    transforms = T.Compose([
        T.Resize((256,256)), 
        T.ToTensor()
    ])

    print('Loading datasets and dataloaders...')
    trainset = IMDBDataset(data_path, transforms, partition_idx_path=args.train_file_path)
    testset = IMDBDataset(data_path, transforms, partition_idx_path=args.test_file_path)
    
    # # Debug only
    # trainset = torch.utils.data.Subset(trainset, list(range(100)))
    # testset = torch.utils.data.Subset(testset, list(range(100)))

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print('Done!')

    # Build model
    num_tasks = len(args.task_ids)
    assert len(args.task_output_sizes) >= num_tasks, 'Please provide one or more task ids and their corresponding output sizes (in order)'
    
    model = None
    if args.use_nddr:
        model = NDDRVGG(
            hidden_dim=args.hidden_dim,
            output_sizes=args.task_output_sizes,
            dataset_name=args.dataset_name,
            learning_rate=args.learning_rate
        )
    else:
        model = MTLVGG(
            hidden_dim=args.hidden_dim,
            num_tasks=num_tasks,
            task_ids=args.task_ids,
            output_sizes=args.task_output_sizes,
            dataset_name=args.dataset_name,
            learning_rate=args.learning_rate
        )
    
    model.to('cpu')
    model.load_state_dict(torch.load(Path(args.ckpt_path + args.model_weights_file),  map_location='cpu')['state_dict'], strict=True)
    model.eval()

    # Run model on testset
    pred, gt = [], []
    for testbatch in tqdm(testloader):
        with torch.no_grad():
            x, y = testbatch
            logits = model(x)
            pred.append(logits)
            gt.append(y)
    pred, gt = torch.vstack([torch.hstack(yy) for yy in pred]), torch.vstack(gt)
    
    # Get predictions for both tasks
    pred_t1, pred_t2 = pred[:, :100], pred[:, -2:]
    pred_t1 = F.softmax(pred_t1, -1)
    pred_t2 = pred_t2.argmax(1)
    age_values = torch.arange(0, 100, 1).to(pred.device)
    pred_t1 = (pred_t1 * age_values).sum(-1)

    # Calculate metrics
    pred_t1 = pred_t1.detach().cpu().numpy().flatten()
    pred_t2 = pred_t2.detach().cpu().numpy().flatten()
    gt = gt.detach().cpu().numpy()
    mae = mean_absolute_error(gt[:, 0], pred_t1)
    made = median_absolute_error(gt[:, 0], pred_t1)
    acc = accuracy_score(gt[:, 1], pred_t2)
    mcc = matthews_corrcoef(gt[:, 1], pred_t2)
    print(f'MAE: {mae}, MedAE: {made}, Acc: {acc}, MCC: {mcc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/media/data/gskenderi/imdb_crop/')
    parser.add_argument('--ckpt_path', type=str, default='/media/data/gskenderi/nddr_ckpt/')
    # parser.add_argument('--model_weights_file', type=str, default='VGG-multitask_True-usesNDDR_True-tasks[0, 1]-epoch=5-11-01-2023-21-24.ckpt')
    parser.add_argument('--model_weights_file', type=str, default='VGG-multitask_True-usesNDDR_True-tasks[0, 1]-epochepoch=5-25-12-2022-15-34.ckpt')
    parser.add_argument('--train_file_path', type=str, default='data/train_idx.npy')
    parser.add_argument('--test_file_path', type=str, default='data/test_idx.npy')
    parser.add_argument('--dataset_name', type=str, default='imdb')
    parser.add_argument('--use_nddr', action='store_true')
    parser.add_argument('--task_ids', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--task_output_sizes', type=int, nargs='+', default=[100, 2])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.0001)     
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=21) 
    parser.add_argument('--gpu_num', type=int, default=0) 

    args = parser.parse_args()
    main(args)
