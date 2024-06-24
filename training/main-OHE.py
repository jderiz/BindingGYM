from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--root', type=str, default='../')
parser.add_argument('--train_dms_mapping', type=str, default='')
parser.add_argument('--test_dms_mapping', type=str, default='')
parser.add_argument('--train_number', type=int, default=None)

parser.add_argument('--dms_input', type=str, default='')
parser.add_argument('--dms_index', type=int, default=0)

parser.add_argument('--structure_path', type=str, default='')

parser.add_argument('--model_type', type=str, default='OHE')
parser.add_argument('--lora', action='store_true', default=True, help='')

parser.add_argument('--mode', type=str, default='intra')
parser.add_argument('--split', type=str, default='random')
parser.add_argument('--use_weight', type=str, default='pretrained')

parser.add_argument('--remark', type=str, default='')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--tmp_path', type=str, default='tmp')

parser.add_argument('--first_fold', action='store_true', default=False, help='')
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
import copy,datetime
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression, Ridge
import networkx as nx

import scipy
from utils import DMS_file_for_LLM

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

training = True

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', \
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def onehot(seqs):
    seqs = np.array([list(seq) for seq in seqs])
    X = np.zeros((seqs.shape[0],seqs.shape[1]*20))
    for i in range(seqs.shape[1]):
        for j,aa in enumerate(amino_acids):
            X[:,i*20+j] = seqs[:,i] == aa
    return X

def onehot_aa(mutants):
    X = np.zeros((len(mutants),400))
    for i,ms in enumerate(mutants):
        if ms == '':continue
        for m in ms.split(':'):
            wt_aa = m[0]
            mt_aa = m[-1]
            j = amino_acids.index(wt_aa)
            k = amino_acids.index(mt_aa)
            X[i,j*20+k] = 1
    return X 

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
        cluster_df = pd.read_csv('./data/BindingGYM_cluster.tsv',sep='\t',header=None,names=['cluster','DMS_id'])
        cluster_map_dic = dict(cluster_df.set_index('DMS_id')['cluster'])
        clusters = []
        for i in tqdm(train_df.index):
            DMS_id = train_df.loc[i,'DMS_id']
            dms_df = pd.read_csv(f'{args.dms_input}/{DMS_id}.csv')
            dms_df = DMS_file_for_LLM(dms_df,focus=False if args.model_type=='structure' else True)
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
    if np.isnan(pearson):
        pearson = 0
    if np.isnan(spearman):
        spearman = 0
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
    if test is not None:
        if args.model_type == 'OHE':
            test_X = onehot(test['mutated_sequence'].values)
        elif args.model_type == 'OHE_AA': 
            test_X = onehot(test['mutant'].values)
        test_y = test['DMS_score'].values
    all_valid = []
    all_test_pred = []
    for fold in range(folds):
        Write_log(log,f'fold{fold} training start')
        if obj_max == 1:
            best_valid_metric = -1e9
        else:
            best_valid_metric = 1e9

        fold_train = train.loc[split[fold][0]].reset_index(drop=True)
        fold_valid = train.loc[split[fold][1]].reset_index(drop=True)
        train_X = onehot(fold_train['mutated_sequence'].values) if args.model_type == 'OHE' else onehot_aa(fold_train['mutant'].values)
        train_y = fold_train['DMS_score'].values
        valid_X = onehot(fold_valid['mutated_sequence'].values) if args.model_type == 'OHE' else onehot_aa(fold_valid['mutant'].values)
        valid_y = fold_valid['DMS_score'].values

        for reg_coef in [0.01]:
            model = Ridge(alpha=reg_coef)
            model.fit(train_X, train_y)
            valid_pred = model.predict(valid_X)
        
            if test is None:
                valid_metrics = Metric(valid_pred,valid_y)
                
            if test is not None:
                test_pred = model.predict(test_X)
                if args.split == 'top_test':
                    valid_metrics = Metric(valid_pred,valid_y,bottom_preds=test_pred[:bottom_n],top_preds=test_pred[bottom_n:])
                else:
                    valid_metrics = Metric(valid_pred,valid_y)
                test_metrics = Metric(test_pred,test_y)
            if obj_max*(valid_metrics['spearman']) > obj_max*best_valid_metric:
                not_improve_epochs = 0
                best_valid_metric = valid_metrics['spearman']
                best_metric = valid_metrics
                best_valid_pred = valid_pred
                if test is not None:
                    best_test_metric = test_metrics
                    best_test_pred = test_pred
                    Write_log(log,'[reg_coef %s] valid_mean:%.6f, valid_pearson:%.6f, valid_spearman: %.6f, valid_rmse: %.6f'%(reg_coef,np.mean(valid_pred),valid_metrics['pearson'],valid_metrics['spearman'],valid_metrics['rmse']))
                    Write_log(log,'test_mean:%.6f, test_pearson: %.6f, test_spearman: %.6f, test_rmse: %.6f'%(np.mean(test_pred),test_metrics['pearson'],test_metrics['spearman'],test_metrics['rmse']))
                else:
                    Write_log(log,'[reg_coef %s] valid_mean:%.6f, valid_pearson:%.6f, valid_spearman: %.6f, valid_rmse: %.6f'%(reg_coef,np.mean(valid_pred),valid_metrics['pearson'],valid_metrics['spearman'],valid_metrics['rmse']))
            else:
                not_improve_epochs += 1
                if test is not None:
                    Write_log(log,'[reg_coef %s] valid_mean:%.6f, valid_pearson:%.6f, valid_spearman: %.6f, valid_rmse: %.6f NIE +1 ---> %s'%(reg_coef,np.mean(valid_pred),valid_metrics['pearson'],valid_metrics['spearman'],valid_metrics['rmse'],not_improve_epochs))
                    Write_log(log,'test_mean:%.6f, test_pearson: %.6f, test_spearman: %.6f, test_rmse: %.6f'%(np.mean(test_pred),test_metrics['pearson'],test_metrics['spearman'],test_metrics['rmse']))
                else:
                    Write_log(log,'[reg_coef %s] valid_mean:%.6f, valid_pearson:%.6f, valid_spearman: %.6f, valid_rmse: %.6f NIE +1 ---> %s'%(reg_coef,np.mean(valid_pred),valid_metrics['pearson'],valid_metrics['spearman'],valid_metrics['rmse'],not_improve_epochs))
                    
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
                
        if test is not None:
            test['fold'] = fold
            print(test)    
            all_test_pred.append(test.copy()) 

    if args.mode == 'intra':
        train.to_csv(output_path+'oof.csv',index=False)
    elif args.mode == 'inter':
        for valid in all_valid:
            for DMS_id,g in valid.groupby('DMS_id'):
                g.to_csv(output_path+f'{DMS_id}_oof.csv',index=False)
    if test is not None:
        all_test_pred = pd.concat(all_test_pred)
        print(all_test_pred)
        all_test_pred.to_csv(output_path+'pred.csv',index=False)
   
    log.close()