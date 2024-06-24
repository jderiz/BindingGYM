from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--root', type=str, default='../')
parser.add_argument('--train_dms_mapping', type=str, default='')
parser.add_argument('--test_dms_mapping', type=str, default='')
parser.add_argument('--train_number', type=int, default=None)

parser.add_argument('--dms_input', type=str, default='')
parser.add_argument('--dms_index', type=int, default=0)

parser.add_argument('--structure_path', type=str, default='../input/structures')

parser.add_argument('--model_type', type=str, default='structure')
parser.add_argument('--lora', action='store_true', default=True, help='')

parser.add_argument('--mode', type=str, default='intra')
parser.add_argument('--split', type=str, default='random')
parser.add_argument('--use_weight', type=str, default='pretrained')

parser.add_argument('--remark', type=str, default='')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--tmp_path', type=str, default='tmp')

parser.add_argument('--top_test_frac', type=float, default=0.1)

parser.add_argument('--batch_size', type=int, default=48)

parser.add_argument('--seed', type=int, default=42, help='')

args, unknown = parser.parse_known_args()

import os,gc,sys
import math
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import copy
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from sklearn.metrics import roc_auc_score
import scipy
import DEMEmodel
from loss import myloss,listMLE
from dataset import *
import datetime
from torch_geometric.loader import DataLoader, DataListLoader

from torch_geometric.nn.data_parallel import DataParallel
import esm
from esm import ESM2, Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
# from SaProt import get_struc_seq, load_esm_saprot
from peft import get_peft_model, LoraConfig
from protein_mpnn_utils import parse_PDB,tied_featurize,ProteinMPNN
from utils import DMS_file_for_LLM
# torch.multiprocessing.set_sharing_strategy('file_system')

import random
def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


import hashlib

def generate_unique_id(string):
    # 创建一个哈希对象
    hash_object = hashlib.sha1()
    
    # 将字符串编码为字节流并更新哈希对象
    hash_object.update(string.encode('utf-8'))
    
    # 获取哈希值的十六进制表示
    hash_value = hash_object.hexdigest()
    
    # 返回唯一的ID
    return hash_value

Seed_everything(args.seed)

lora_config = {
                # "task_type": "SEQ_CLS",
                "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj", "lm_head.dense"],
                "modules_to_save": [],
                "inference_mode": False,
                "lora_dropout": 0.1,
                "lora_alpha": 8,
            }
peft_config = LoraConfig(**lora_config)

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', \
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def onehot(x):
    seqnpy=np.zeros((len(x),len(amino_acids)))
    seq1=np.array(list(x))
    for i,aa in enumerate(amino_acids):
        seqnpy[seq1==aa,i] = 1
    return seqnpy

training = True

if training:
    train_df = pd.read_csv(args.train_dms_mapping)
    train = []
    valid = []
    test = None
    if args.mode == 'intra':
        name = train_df.loc[args.dms_index,'DMS_id']
        train = pd.read_csv(f'{args.dms_input}/{name}.csv')
        train = DMS_file_for_LLM(train,focus=False if args.model_type=='structure' else True)
        print(train)
        if args.split == 'top_test':
            train = train.sort_values(by='DMS_score').reset_index(drop=True)
            train['rank'] = np.arange(0,train.shape[0])
            train['top_frac'] = train['rank'] / train.shape[0]
            test = train.loc[(train['top_frac']<args.top_test_frac)|(train['top_frac']>=(1-args.top_test_frac))].reset_index(drop=True)
            bottom_n = (test['top_frac']<args.top_test_frac).sum()
            print(test)
            train = train.loc[(train['top_frac']>=args.top_test_frac)&(train['top_frac']<(1-args.top_test_frac))].reset_index(drop=True)
    elif args.mode == 'inter':
        cluster_df = pd.read_csv('./cache/BindingGYM_cluster.tsv',sep='\t',header=None,names=['cluster','DMS_id'])
        cluster_map_dic = dict(cluster_df.set_index('DMS_id')['cluster'])
        clusters = []
        for i in tqdm(train_df.index):
            DMS_id = train_df.loc[i,'DMS_id']
            dms_df = pd.read_csv(f'{args.dms_input}/{DMS_id}.csv')
            # for c in ['mutant','wildtype_sequence','mutated_sequence']:
            #     dms_df[c] = dms_df[c].apply(eval)
            dms_df = DMS_file_for_LLM(dms_df,focus=False if args.model_type=='structure' else True)
            # if len(dms_df['wildtype_sequence'].values[0]) > 1000:continue
            print(DMS_id)
            dms_df['DMS_id'] = DMS_id
            train.append(dms_df)
            clusters.append(cluster_map_dic[DMS_id])        
        print(clusters)
    if args.mode == 'intra':
        args.tmp_path = f'train_on_{name}_{args.mode}_{args.split}'
    else:
        args.tmp_path = f'train_on_BindingGYM_{args.mode}_{args.split}'

    if args.use_weight == 'native':
        args.tmp_path += f'_native_{args.model_type}'
    else:
        args.tmp_path += f'_{args.model_type}'
    if args.lora and args.model_type == 'sequence':
        args.tmp_path += f'_lora'
    args.tmp_path += f'_seed{args.seed}'
    print(args.tmp_path)


def Metric(preds,labels,bottom_preds=None,bottom_labels=None,top_preds=None,top_labels=None,eps=1e-6):
    pearson = scipy.stats.pearsonr(preds,labels)[0]
    spearman = scipy.stats.spearmanr(preds,labels)[0]
    rmse = np.mean((preds-labels)**2)**0.5
    ms = {'pearson':pearson,
            'spearman':spearman,
            'rmse':rmse}
    top_metrics = []
    bottom_metrics = []
    unbias_metrics = []
    
    if top_preds is not None:
        all_preds = np.concatenate([bottom_preds,preds,top_preds])
        # n = int(len(top_labels)*0.1/args.top_test_frac)
        n = len(top_preds)
        m = len(bottom_preds)
        top_pred_idxs = np.argsort(all_preds)[-(len(all_preds)-n):]
        top_idxs = np.arange(n)
        for k in [10,20,50,100]:
            precision = (top_pred_idxs[-min(k,n):]>=(len(all_preds)-n)).mean()
            recall = (top_pred_idxs[-min(k,n):]>=(len(all_preds)-n)).sum() / n
            f1 = 2*precision*recall / (precision+recall+eps)
            top_metrics.append([precision,recall,f1])

            precision = (top_pred_idxs[:min(k,m)]<m).mean()
            recall = (top_pred_idxs[:min(k,m)]<m).sum() / n
            f1 = 2*precision*recall / (precision+recall+eps)
            bottom_metrics.append([precision,recall,f1])

        precision = (top_pred_idxs>=(len(all_preds)-n)).mean()
        recall = (top_pred_idxs>=(len(all_preds)-n)).sum() / n
        f1 = 2*precision*recall / (precision+recall+eps)
        top_metrics.append([precision,recall,f1])
        
        precision = (top_pred_idxs<m).mean()
        recall = (top_pred_idxs<m).sum() / n
        f1 = 2*precision*recall / (precision+recall+eps)
        bottom_metrics.append([precision,recall,f1])

        for i,k in enumerate([10,20,50,100,'top_frac']):
            unbias_metrics.append(list(np.array(top_metrics[i])-np.array(bottom_metrics[i])))

        ms.update({'top10_precision_recall_f1':top_metrics[0],
                'top20_precision_recall_f1':top_metrics[1],
                'top50_precision_recall_f1':top_metrics[2],
                'top100_precision_recall_f1':top_metrics[3],
                'top_frac_precision_recall_f1':top_metrics[4],
                'bottom10_precision_recall_f1':bottom_metrics[0],
                'bottom20_precision_recall_f1':bottom_metrics[1],
                'bottom50_precision_recall_f1':bottom_metrics[2],
                'bottom100_precision_recall_f1':bottom_metrics[3],
                'bottom_frac_precision_recall_f1':bottom_metrics[4],
                'unbias10_precision_recall_f1':unbias_metrics[0],
                'unbias20_precision_recall_f1':unbias_metrics[1],
                'unbias50_precision_recall_f1':unbias_metrics[2],
                'unbias100_precision_recall_f1':unbias_metrics[3],
                'unbias_frac_precision_recall_f1':unbias_metrics[4],})
    return ms

bs = args.batch_size

def Write_log(logFile,text,isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    return None

gpus = list(range(torch.cuda.device_count()))
print("Gpus:",len(gpus))
obj_max = 1
folds = 5
epochs = 100
lr = 1e-3
patience = 3
run_id = None
output_root = './output/'

if not run_id:
    run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    while os.path.exists(output_root+run_id+'/'):
        time.sleep(1)
        run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_root + f'{args.tmp_path}/'
else:
    output_path = output_root + run_id + '/'


if training:    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for file in ['*.py','*.sh']:
        os.system(f'cp ./{file} {output_path}')

    log = open(output_path + 'train.log','w',buffering=1)

    if len(gpus) > 1:
        parallel_running = True
    else:
        parallel_running = False
    if parallel_running:
        loader_class = DataListLoader
    else:
        loader_class = DataLoader
    print('parallel_running: ', parallel_running)


    all_oof = []
    all_metric_df = []

    if args.mode == 'intra':
        if args.split == 'contig':
            split = [[[],[]] for _ in range(folds)]
            train = train.loc[train['mutant'].apply(lambda x:len(x.split(':'))<2)].reset_index(drop=True)
            if train.shape[0] < 100:
                sys.exit()
            print(train)
            all_muts = {}
            for i in train.index:
                for m in train.loc[i,'mutant'].split(':'):
                    if m == '':
                        pos = 0
                    else:
                        pos = int(m[1:-1])
                    if pos not in all_muts:
                        all_muts[pos] = [i]
                    else:
                        all_muts[pos].append(i)
            print(all_muts.keys())
            fold_count = [0 for _ in range(folds)]
            used_idxs = set()
            sorted_muts = sorted(all_muts.items(),key=lambda x:x[0])
            at_fold = 0
            used_count = 0
            for i in range(len(sorted_muts)):
                pos,idxs = sorted_muts[i]
                idxs = set(idxs)
                split[at_fold][1].extend(list(idxs-used_idxs))
                fold_count[at_fold] = len(split[at_fold][1])
                for fold in range(folds):
                    if fold != at_fold:
                        split[fold][0].extend(list(idxs-used_idxs))
                used_idxs |= idxs
                if fold_count[at_fold] >= (train.shape[0]-used_count)/(folds-at_fold) and at_fold < folds-1:
                    used_count += fold_count[at_fold]
                    at_fold += 1
                    
            print([split[fold][1][:5] for fold in range(folds)])
            print([len(split[fold][0]) for fold in range(folds)])
            print([len(split[fold][1]) for fold in range(folds)])
        elif args.split == 'mod':
            split = [[[],[]] for _ in range(folds)]
            train = train.loc[train['mutant'].apply(lambda x:len(x.split(':'))<2)].reset_index(drop=True)
            if train.shape[0] < 100:
                sys.exit()
            print(train)
            all_muts = {}
            for i in train.index:
                for m in train.loc[i,'mutant'].split(':'):
                    if m == '':
                        pos = 0
                    else:
                        pos = int(m[1:-1])
                    if pos not in all_muts:
                        all_muts[pos] = [i]
                    else:
                        all_muts[pos].append(i)
            print(all_muts.keys())
            used_idxs = set()
            for pos in all_muts:
                at_fold = pos % folds
                idxs = all_muts[pos]
                idxs = set(idxs)
                split[at_fold][1].extend(list(idxs-used_idxs))
                for fold in range(folds):
                    if fold != at_fold:
                        split[fold][0].extend(list(idxs-used_idxs))
                used_idxs |= idxs
            print([split[fold][1][:5] for fold in range(folds)])
            print([len(split[fold][0]) for fold in range(folds)])
            print([len(split[fold][1]) for fold in range(folds)])
        else:
            idxs = np.array(train.index.tolist())
            split = list(KFold(n_splits=folds, random_state=args.seed, 
                shuffle=True).split(idxs))
    elif args.mode == 'inter':
        split = list(GroupKFold(n_splits=folds).split(clusters,groups=clusters))
    if args.model_type == 'structure':
        TaskDataset = StructureDataset
    elif args.model_type == 'sequence':
        TaskDataset = SequenceDataset
    
    make_test_dataset = False
    all_valid = []
    for fold in range(folds):
        Write_log(log,f'fold{fold} training start')
        if obj_max == 1:
            best_valid_metric = -1e9
        else:
            best_valid_metric = 1e9
        hidden_dim = 128
        num_layers = 3
        if args.model_type == 'structure':
            model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=0.2, k_neighbors=48)
            if args.use_weight == 'pretrained':
                state_dict = torch.load("./cache/v_48_020.pt", torch.device('cpu'))
                model.load_state_dict(state_dict['model_state_dict'])
            esm_alphabet = None
        elif args.model_type == 'sequence':
            if args.use_weight == 'pretrained':
                esm_pretrain_model, esm_alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
            else:
                esm_pretrain_model = ESM2()
                esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
            if args.lora:
                esm_pretrain_model = get_peft_model(esm_pretrain_model, peft_config)
        
            model = DEMEmodel.DEME(esm_pretrain_model, None)
        if parallel_running:
            model = DataParallel(model)
        model.cuda()
        if test is not None and not make_test_dataset:
            test_dataset = TaskDataset(test,test.index.tolist(),structure_path=args.structure_path,batch_size=args.batch_size,esm_alphabet=esm_alphabet)
            test_dataloader = loader_class(dataset=test_dataset,batch_size=bs,shuffle=False, drop_last=False,num_workers=8)
            all_test_pred = []
            make_test_dataset = True
            
        if args.mode == 'intra':
            fold_train = [train.loc[split[fold][0]].reset_index(drop=True)]
            fold_valid = train.loc[split[fold][1]].reset_index(drop=True)
        elif args.mode == 'inter':
            fold_train = [train[i] for i in split[fold][0]]
            fold_valid = pd.concat([train[i] for i in split[fold][1]]).reset_index(drop=True)
            print(fold_valid['DMS_id'].unique())
        train_dataset = TaskDataset(fold_train,list(range(len(fold_train))),structure_path=args.structure_path,batch_size=args.batch_size,esm_alphabet=esm_alphabet,evaluation=False)
        train_dataloader = loader_class(dataset=train_dataset,batch_size=bs,shuffle=False, drop_last=True,num_workers=8)
        valid_dataset = TaskDataset(fold_valid,fold_valid.index.tolist(),structure_path=args.structure_path,batch_size=args.batch_size,esm_alphabet=esm_alphabet)
        valid_dataloader = loader_class(dataset=valid_dataset,batch_size=bs,shuffle=False, drop_last=False,num_workers=8)
        loss_tr = listMLE

        optimizer = torch.optim.AdamW(model.parameters(),betas=(0.9, 0.99), lr=lr, weight_decay=0.05,eps=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=100)
        
        not_improve_epochs = 0
        for epoch in range(epochs):
            np.random.seed(666*epoch)
            train_dataset.seed_bias = epoch
            train_loss = 0.0
            train_num = 0
            
            model.train()
            if args.model_type == 'structure':
                model.features.augment_eps = 0.2
            
            bar = tqdm(train_dataloader)

            if epoch > 0:
                for i,data in enumerate(bar):
                    optimizer.zero_grad()
                    try:
                        if not parallel_running:
                            data = data.cuda()
                        outputs = model(data)
                        if parallel_running:
                            y = torch.cat([d.reg_labels for d in data],dim=0).to(outputs.device)
                        else:
                            y = data.reg_labels
    
                        loss = loss_tr(-outputs,-y)
                        train_num += y.shape[0]
                        train_loss += y.shape[0] * loss.item()

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)

                        optimizer.step()
                        scheduler.step()
                        bar.set_description('loss: %.4f' % (loss.item()))
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print('| WARNING: ran out of memory, skipping batch')
                            for p in model.parameters():
                                if p.grad is not None:
                                    del p.grad  # free some memory
                            del data
                            torch.cuda.empty_cache()
                            continue
                        else:
                            raise(e)
                
                train_loss /= train_num

            model.eval()
            if args.model_type == 'structure':
                model.features.augment_eps = 0.
                model.randn = None
            valid_loss = 0.0
            valid_num = 0
            valid_pred = []
            valid_y = []
            bar = tqdm(valid_dataloader)
            for k,data in enumerate(bar):
                if not parallel_running:
                    data = data.cuda()
                with torch.no_grad():
                    outputs = model(data)
                    valid_pred.append(outputs.detach().cpu().numpy())
                    if parallel_running:
                        y = torch.cat([d.reg_labels for d in data],dim=0).to(outputs.device)
                    else:
                        y = data.reg_labels
                    valid_y.append(y.detach().cpu().numpy())
                    loss = loss_tr(-outputs,-y)
                    valid_num += y.shape[0]
                    valid_loss += y.shape[0] * loss.item()
                    bar.set_description('loss: %.4f' % (loss.item()))
                
            valid_pred = np.concatenate(valid_pred).reshape(-1)
            valid_y = np.concatenate(valid_y).reshape(-1)
            valid_metric = valid_loss / valid_num

            
            if args.model_type == 'structure':
                all_randn = [model.randn.clone()]
                for _ in range(4):
                    valid_pred1 = []
                    model.randn = None
                    bar = tqdm(valid_dataloader)
                    for k,data in enumerate(bar):
                        if not parallel_running:
                            data = data.cuda()
                        with torch.no_grad():
                            outputs = model(data)
                            valid_pred1.append(outputs.detach().cpu().numpy())
                            if parallel_running:
                                y = torch.cat([d.reg_labels for d in data],dim=0).to(outputs.device)
                            else:
                                y = data.reg_labels
                            loss = loss_tr(-outputs,-y)
                            bar.set_description('loss: %.4f' % (loss.item()))
                    valid_pred1 = np.concatenate(valid_pred1).reshape(-1)
                    valid_pred += valid_pred1
                    all_randn.append(model.randn.clone())
                valid_pred /= 5
            
            if test is None:
                if args.mode == 'intra':
                    valid_metrics = Metric(valid_pred,valid_y)
                elif args.mode == 'inter':
                    print(fold_valid.groupby('DMS_id',sort=False)['DMS_score'].count().cumsum())
                    count_cumsum = fold_valid.groupby('DMS_id',sort=False)['DMS_score'].count().cumsum().tolist()
                    for i in range(len(count_cumsum)):
                        if i == 0:
                            valid_metrics = Metric(valid_pred[:count_cumsum[i]],valid_y[:count_cumsum[i]])
                            print(valid_metrics)
                        else:
                            valid_metrics1 = Metric(valid_pred[count_cumsum[i-1]:count_cumsum[i]],valid_y[count_cumsum[i-1]:count_cumsum[i]])
                            print(valid_metrics1)
                            for k in valid_metrics:
                                valid_metrics[k] += valid_metrics1[k]
                    for k in valid_metrics:
                        valid_metrics[k] /= len(count_cumsum)
            else:
                test_loss = 0.0
                test_num = 0
                test_pred = []
                test_y = []
                bar = tqdm(test_dataloader)
                if args.model_type == 'structure':
                    model.features.augment_eps = 0.
                    model.randn = all_randn[0]
                for data in bar:
                    if not parallel_running:
                        data = data.cuda()
                    with torch.no_grad():
                        outputs = model(data)
                        test_pred.append(outputs.detach().cpu().numpy())
                        if parallel_running:
                            y = torch.cat([d.reg_labels for d in data],dim=0).to(outputs.device)
                            # mask = torch.cat([d.label_masks for d in data],dim=0).to(outputs.device)
                        else:
                            y = data.reg_labels
                            # mask = data.label_masks
                        test_y.append(y.detach().cpu().numpy())
                        loss = loss_tr(-outputs,-y)#(loss_tr(-outputs,-y)).mean()
                        # loss = loss.sum() / mask.sum()
                        test_num += y.shape[0]
                        test_loss += y.shape[0] * loss.item()
                        bar.set_description('loss: %.4f' % (loss.item()))
                test_pred = np.concatenate(test_pred).reshape(-1)
                test_y = np.concatenate(test_y).reshape(-1)
                test_metric = test_loss / valid_num
                if args.model_type == 'structure':
                    for run_i in range(4):
                        test_pred1 = []
                        model.randn = all_randn[run_i+1]
                        bar = tqdm(test_dataloader)
                        for k,data in enumerate(bar):
                            if not parallel_running:
                                data = data.cuda()
                            with torch.no_grad():
                                outputs = model(data)
                                test_pred1.append(outputs.detach().cpu().numpy())
                                if parallel_running:
                                    y = torch.cat([d.reg_labels for d in data],dim=0).to(outputs.device)
                                else:
                                    y = data.reg_labels
                                loss = loss_tr(-outputs,-y)
                                bar.set_description('loss: %.4f' % (loss.item()))
                        test_pred1 = np.concatenate(test_pred1).reshape(-1)
                        test_pred += test_pred1
                    test_pred /= 5
                if args.split == 'top_test':
                    valid_metrics = Metric(valid_pred,valid_y,bottom_preds=test_pred[:bottom_n],top_preds=test_pred[bottom_n:])
                else:
                    valid_metrics = Metric(valid_pred,valid_y)
                test_metrics = Metric(test_pred,test_y)
            if obj_max*(valid_metrics['spearman']) > obj_max*best_valid_metric:
                if len(gpus) > 1:
                    best_model = copy.deepcopy(model.module.state_dict())
                    # torch.save(model.module.state_dict(),output_path + f'model{fold}.ckpt')
                else:
                    best_model = copy.deepcopy(model.state_dict())
                    # torch.save(model.state_dict(),output_path + f'model{fold}.ckpt')
                not_improve_epochs = 0
                best_valid_metric = valid_metrics['spearman']
                best_metric = valid_metrics
                best_valid_pred = valid_pred
                if test is not None:
                    best_test_metric = test_metrics
                    best_test_pred = test_pred
                    Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, valid_loss: %.6f, valid_mean:%.6f, valid_pearson:%.6f, valid_spearman: %.6f, valid_rmse: %.6f'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_metric,np.mean(valid_pred),valid_metrics['pearson'],valid_metrics['spearman'],valid_metrics['rmse']))
                    Write_log(log,'test_mean:%.6f, test_pearson: %.6f, test_spearman: %.6f, test_rmse: %.6f'%(np.mean(test_pred),test_metrics['pearson'],test_metrics['spearman'],test_metrics['rmse']))
                else:
                    Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, valid_loss: %.6f, valid_mean:%.6f, valid_pearson:%.6f, valid_spearman: %.6f, valid_rmse: %.6f'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_metric,np.mean(valid_pred),valid_metrics['pearson'],valid_metrics['spearman'],valid_metrics['rmse']))
            else:
                not_improve_epochs += 1
                if test is not None:
                    Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, valid_loss: %.6f, valid_mean:%.6f, valid_pearson:%.6f, valid_spearman: %.6f, valid_rmse: %.6f NIE +1 ---> %s'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_metric,np.mean(valid_pred),valid_metrics['pearson'],valid_metrics['spearman'],valid_metrics['rmse'],not_improve_epochs))
                    Write_log(log,'test_mean:%.6f, test_pearson: %.6f, test_spearman: %.6f, test_rmse: %.6f'%(np.mean(test_pred),test_metrics['pearson'],test_metrics['spearman'],test_metrics['rmse']))
                else:
                    Write_log(log,'[epoch %s] lr: %.6f, train_loss: %.6f, valid_loss: %.6f, valid_mean:%.6f, valid_pearson:%.6f, valid_spearman: %.6f, valid_rmse: %.6f NIE +1 ---> %s'%(epoch,optimizer.param_groups[0]['lr'],train_loss,valid_metric,np.mean(valid_pred),valid_metrics['pearson'],valid_metrics['spearman'],valid_metrics['rmse'],not_improve_epochs))
                    
            if args.split == 'top_test':
                Write_log(log,'top10_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['top10_precision_recall_f1'][0],valid_metrics['top10_precision_recall_f1'][1],valid_metrics['top10_precision_recall_f1'][2]))
                Write_log(log,'top20_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['top20_precision_recall_f1'][0],valid_metrics['top20_precision_recall_f1'][1],valid_metrics['top20_precision_recall_f1'][2]))
                Write_log(log,'top50_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['top50_precision_recall_f1'][0],valid_metrics['top50_precision_recall_f1'][1],valid_metrics['top50_precision_recall_f1'][2]))
                Write_log(log,'top100_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['top100_precision_recall_f1'][0],valid_metrics['top100_precision_recall_f1'][1],valid_metrics['top100_precision_recall_f1'][2]))
                Write_log(log,'bottom10_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['bottom10_precision_recall_f1'][0],valid_metrics['bottom10_precision_recall_f1'][1],valid_metrics['bottom10_precision_recall_f1'][2]))
                Write_log(log,'bottom20_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['bottom20_precision_recall_f1'][0],valid_metrics['bottom20_precision_recall_f1'][1],valid_metrics['bottom20_precision_recall_f1'][2]))
                Write_log(log,'bottom50_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['bottom50_precision_recall_f1'][0],valid_metrics['bottom50_precision_recall_f1'][1],valid_metrics['bottom50_precision_recall_f1'][2]))
                Write_log(log,'bottom100_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['bottom100_precision_recall_f1'][0],valid_metrics['bottom100_precision_recall_f1'][1],valid_metrics['bottom100_precision_recall_f1'][2]))
                Write_log(log,'unbias10_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['unbias10_precision_recall_f1'][0],valid_metrics['unbias10_precision_recall_f1'][1],valid_metrics['unbias10_precision_recall_f1'][2]))
                Write_log(log,'unbias20_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['unbias20_precision_recall_f1'][0],valid_metrics['unbias20_precision_recall_f1'][1],valid_metrics['unbias20_precision_recall_f1'][2]))
                Write_log(log,'unbias50_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['unbias50_precision_recall_f1'][0],valid_metrics['unbias50_precision_recall_f1'][1],valid_metrics['unbias50_precision_recall_f1'][2]))
                Write_log(log,'unbias100_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['unbias100_precision_recall_f1'][0],valid_metrics['unbias100_precision_recall_f1'][1],valid_metrics['unbias100_precision_recall_f1'][2]))
            if not_improve_epochs >= patience:
                break
        if args.mode == 'intra':
            train.loc[split[fold][1],'fold'] = fold
            train.loc[split[fold][1],'pred'] = best_valid_pred
        else:
            fold_valid['pred'] = best_valid_pred
            all_valid.append(fold_valid)
        if test is None:
            all_metrics = best_metric
            Write_log(log,'oof_spearman: %.6f, oof_rmse: %.6f'%(all_metrics['pearson'],all_metrics['spearman']))
        else:
            test['pred'] = best_test_pred
            all_metrics = best_metric
            Write_log(log, 'oof_pearson:%.6f, oof_spearman: %.6f, oof_rmse: %.6f'%(all_metrics['pearson'],all_metrics['spearman'],all_metrics['rmse']))
            Write_log(log, 'test_pearson:%.6f, test_mean:%.6f, test_spearman: %.6f, test_rmse: %.6f'%(np.mean(test_pred),best_test_metric['pearson'],best_test_metric['spearman'],best_test_metric['rmse']))
            test['fold'] = fold
            print(test)    
            all_test_pred.append(test.copy()) 
        
        del optimizer,scheduler,model,data,outputs,loss,y
        gc.collect()
        torch.cuda.empty_cache()

        torch.save(best_model,output_path + f'model{fold}.ckpt')
    
    if args.mode == 'intra':
        train.to_csv(output_path+'oof.csv',index=False)
    elif args.mode == 'inter':
        for valid in all_valid:
            for DMS_id,g in valid.groupby('DMS_id'):
                g.to_csv(output_path+f'{DMS_id}_oof.csv',index=False)
    if test is not None:
        all_test_pred = pd.concat(all_test_pred)
        all_test_pred.to_csv(output_path+'pred.csv',index=False)
        
    log.close()
    