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
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--feat', type=str, default='esm')

parser.add_argument('--first_fold', action='store_true', default=False, help='')
parser.add_argument('--top_test_frac', type=float, default=None)

parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--include_flu', action='store_true', default=False, help='')

parser.add_argument('--aug', action='store_true', default=False, help='')
parser.add_argument('--struc3D', action='store_true', default=False, help='')
# parser.add_argument('--RNAformer', action='store_true', default=False, help='')
# parser.add_argument('--EternaFold', action='store_true', default=False, help='')
# parser.add_argument('--vienna_2', action='store_true', default=False, help='')
# parser.add_argument('--rnastructure', action='store_true', default=False, help='')
# parser.add_argument('--contrafold_2', action='store_true', default=False, help='')

parser.add_argument('--mas_opm', action='store_true', default=False, help='')
parser.add_argument('--pair_triio', action='store_true', default=False, help='')
parser.add_argument('--pair_trise', action='store_true', default=False, help='')
parser.add_argument('--pair_trans', action='store_true', default=False, help='')

parser.add_argument('--do_train', action='store_true', default=False, help='')
parser.add_argument('--only_valid', action='store_true', default=False, help='')
parser.add_argument('--predict', action='store_true', default=False, help='')
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
# from SaProt import get_struc_seq, load_esm_saprot
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

training = True if not args.predict or args.do_train else False

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', \
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

def onehot(seqs):
    seqs = np.array([list(seq) for seq in seqs])
    X = np.zeros((seqs.shape[0],seqs.shape[1]*20))
    for i in range(seqs.shape[1]):
        for j,aa in enumerate(amino_acids):
            X[:,i*20+j] = seqs[:,i] == aa
    return X

def aastat(mutants):
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
    if args.mode == 'intra':
        name = train_df.loc[args.dms_index,'DMS_id']
        train = pd.read_csv(f'{args.dms_input}/{name}.csv')
        train = DMS_file_for_LLM(train,focus=False if args.model_type=='structure' else True)
        print(train)
        test = None
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
            if not args.include_flu:
                if 'FluAH' in DMS_id:
                    continue
            dms_df = pd.read_csv(f'{args.dms_input}/{DMS_id}.csv')
            # for c in ['mutant','wildtype_sequence','mutated_sequence']:
            #     dms_df[c] = dms_df[c].apply(eval)
            dms_df = DMS_file_for_LLM(dms_df,focus=False if args.model_type=='structure' else True)
            # if len(dms_df['wildtype_sequence'].values[0]) > 1000:continue
            print(DMS_id)
            dms_df['DMS_id'] = DMS_id
            train.append(dms_df)
            clusters.append(cluster_map_dic[DMS_id])

        test = None
        print(clusters)
        # test = pd.read_csv(args.test_dms_mapping)
        # all_test = []
        # for POI,g in test.groupby('POI'):
        #     g = DMS_file_for_LLM(g,focus=False if args.model_type=='structure' else True)
        #     if len(g['wildtype_sequence'].values[0]) > 1000:continue
        #     all_test.append(g)
        # test = pd.concat(all_test).sort_index()
        # test['POI'] = test['POI'].apply(lambda x:x.split('_')[0])
        # test = test.loc[test['mutant']!=''].reset_index(drop=True)
    if args.mode == 'intra':
        args.tmp_path = f'train_on_{name}_{args.mode}_{args.split}'
    else:
        args.tmp_path = f'train_on_BindingGYM_{args.mode}_{args.split}'

    if args.include_flu and args.mode == 'inter':
        args.tmp_path += '_include_flu'
    if args.use_weight == 'native':
        args.tmp_path += f'_native_{args.model_type}'
    else:
        args.tmp_path += f'_{args.model_type}'
    if args.lora and args.model_type == 'sequence':
        args.tmp_path += f'_lora'
    args.tmp_path += f'_seed{args.seed}'
    # args.tmp_path = f'tmp{args.dms_index}'
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
        all_preds = np.concatenate([top_preds[::-1],preds,bottom_preds])
        n = len(top_preds)
        m = len(bottom_preds)
        top_pred_idxs = np.argsort(-all_preds)[:n]
        top_idxs = np.arange(n)
        for k in [10,20,50,100]:
            precision = (top_pred_idxs[:min(k,n)]<(n/args.top_test_frac*0.1)).mean()
            recall = (top_pred_idxs[:min(k,n)]<(n/args.top_test_frac*0.1)).sum() / n
            f1 = 2*precision*recall / (precision+recall+eps)
            top_metrics.append([precision,recall,f1])

            precision = (top_pred_idxs[:min(k,n)]>=(len(all_preds)-m)).mean()
            recall = (top_pred_idxs[:min(k,n)]>=(len(all_preds)-m)).sum() / n
            f1 = 2*precision*recall / (precision+recall+eps)
            bottom_metrics.append([precision,recall,f1])
 
        for i,k in enumerate([10,20,50,100]):
            unbias_metrics.append(list(np.array(top_metrics[i])-np.array(bottom_metrics[i])))
    #     # print(bottom_preds)
    #     # print(preds)
    #     # print(top_preds)
    #     for k in [10,20,50,100]:
    #         bottom_precision = (bottom_preds > np.percentile(preds,p)).mean()
    #         top_precision = (top_preds > np.percentile(preds,p)).mean()
    #         top_metrics2.append([bottom_precision,top_precision])
    # # return pearson, spearman, rmse,  top_metrics1, top_metrics2
    
        ms.update({'top10_precision_recall_f1':top_metrics[0],
                'top20_precision_recall_f1':top_metrics[1],
                'top50_precision_recall_f1':top_metrics[2],
                'top100_precision_recall_f1':top_metrics[3],
                'top_frac_precision_recall_f1':0,
                'bottom10_precision_recall_f1':bottom_metrics[0],
                'bottom20_precision_recall_f1':bottom_metrics[1],
                'bottom50_precision_recall_f1':bottom_metrics[2],
                'bottom100_precision_recall_f1':bottom_metrics[3],
                'bottom_frac_precision_recall_f1':0,
                'unbias10_precision_recall_f1':unbias_metrics[0],
                'unbias20_precision_recall_f1':unbias_metrics[1],
                'unbias50_precision_recall_f1':unbias_metrics[2],
                'unbias100_precision_recall_f1':unbias_metrics[3],
                'unbias_frac_precision_recall_f1':0,})
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

# if os.path.exists(f'{output_root}/{args.tmp_path}/oof.csv'):
#     sys.exit()

if not run_id:
    run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    while os.path.exists(output_root+run_id+'/'):
        time.sleep(1)
        run_id = 'run_nn_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_root + f'{args.tmp_path}/'
else:
    output_path = output_root + run_id + '/'

# if os.path.exists(output_path+'metric.csv'):
#     raise
if training:    
    if args.do_train:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for file in ['*.py','*.sh']:
            # if os.path.exists(f'./{file}'):
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
                # contig_len = int(np.ceil(len(train.loc[0,'wildtype_sequence'])/folds))
                # split = [[[],[]] for _ in range(folds)]
                # for i in train.index:
                #     count = [0] * folds
                #     for m in train.loc[i,'mutant'].split(':'):
                #         if m == '':
                #             count[0] += 1
                #         else:
                #             count[(int(m[1:-1])-1)//contig_len] += 1
                #     # print(count)
                #     count_max_fold = count.index(max(count))
                #     for fold in range(folds):
                #         if count[fold] > 0:
                #             split[fold][1].append(i)
                #         for fold1 in range(folds):
                #             if fold1 != fold:
                #                 split[fold1][0].append(i)
                        # if fold == count_max_fold:
                        #     split[fold][1].append(i)
                        # else:
                        #     split[fold][0].append(i)
                # print([len(split[fold][1]) for fold in range(folds)])
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
            elif args.split == 'pos':
                split = [[[],[]] for _ in range(folds)]
                all_muts = {}
                for i in train.index:
                    for m in train.loc[i,'mutant'].split(':'):
                        if m == '':
                            if m not in all_muts:
                                all_muts[m] = [i]
                            else:
                                all_muts[m].append(i)
                        else:
                            pos = m
                            if pos not in all_muts:
                                all_muts[pos] = [i]
                            else:
                                all_muts[pos].append(i)
                print(all_muts.keys())
                fold_count = [0 for _ in range(folds)]
                used_idxs = set()
                reverse_muts = sorted(all_muts.items(),key=lambda x:len(x[1]))
                print({x[0]:len(x[1]) for x in reverse_muts})
                ascending_muts = sorted(all_muts.items(),key=lambda x:-len(x[1]))
                min_count_fold = 0
                for i in range(len(all_muts)):
                    if i % 10 == 0 and len(split[min_count_fold][1]) > 1:
                        min_count_fold = fold_count.index(min(fold_count))
                    # if i % 1 == 0:
                    pos,idxs = reverse_muts[i]
                    # else:
                    #     pos,idxs = ascending_muts[i//2]
                    idxs = set(idxs)
                    split[min_count_fold][1].extend(list(idxs-used_idxs))
                    fold_count[min_count_fold] = len(split[min_count_fold][1])
                    for fold in range(folds):
                        if fold != min_count_fold:
                            split[fold][0].extend(list(idxs-used_idxs))
                    used_idxs |= idxs
                print([len(split[fold][0]) for fold in range(folds)])
                print([len(split[fold][1]) for fold in range(folds)])
            else:
                idxs = np.array(train.index.tolist())
                split = list(KFold(n_splits=folds, random_state=args.seed, 
                    shuffle=True).split(idxs))
        elif args.mode == 'inter':
            split = list(GroupKFold(n_splits=folds).split(clusters,groups=clusters))
        if test is not None:
            test_X = aastat(test['mutant'].values)
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
            print(fold_train[['mutant']])
            fold_valid = train.loc[split[fold][1]].reset_index(drop=True)
            print(fold_valid[['mutant']])
            train_X = aastat(fold_train['mutant'].values)
            train_y = fold_train['DMS_score'].values
            valid_X = aastat(fold_valid['mutant'].values)
            valid_y = fold_valid['DMS_score'].values
            if not args.only_valid:
                for reg_coef in [0.01,0.05,0.1,1]:
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
                        # metric_df = pd.DataFrame({'DMS_id':name,
                        #                         'split':[args.split],
                        #                         'oof_spearman':[all_metrics['pearson']],
                        #                         'oof_rmse':[all_metrics['spearman']],
                        #                         'fold':[fold]}) 
                    else:
                        test['pred'] = best_test_pred
                        all_metrics = best_metric
                        Write_log(log, 'oof_pearson:%.6f, oof_spearman: %.6f, oof_rmse: %.6f'%(all_metrics['pearson'],all_metrics['spearman'],all_metrics['rmse']))
                        Write_log(log, 'test_pearson:%.6f, test_mean:%.6f, test_spearman: %.6f, test_rmse: %.6f'%(np.mean(test_pred),best_test_metric['pearson'],best_test_metric['spearman'],best_test_metric['rmse']))
                                
                        # Write_log(log,'top10_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][0][0],all_metrics['rmse'][0][1],all_metrics['rmse'][0][2]))
                        # Write_log(log,'top20_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][1][0],all_metrics['rmse'][1][1],all_metrics['rmse'][1][2]))
                        # Write_log(log,'top50_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][2][0],all_metrics['rmse'][2][1],all_metrics['rmse'][2][2]))
                        # Write_log(log,'top100_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][3][0],all_metrics['rmse'][3][1],all_metrics['rmse'][3][2]))
                        # Write_log(log,'top_test_frac precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][4][0],all_metrics['rmse'][4][1],all_metrics['rmse'][4][2]))
                        # Write_log(log,'bottom_test_beyond_95: %.6f, bottom_test_beyond_99: %.6f'%(all_metrics[3][2][0],all_metrics[3][3][0]))
                        # Write_log(log,'top_test_beyond_95: %.6f, top_test_beyond_99: %.6f'%(all_metrics[3][2][1],all_metrics[3][3][1]))
                        # metric_df = pd.DataFrame({'DMS_id':name,
                        #                         'split':[args.split],
                        #                         'oof_spearman':[all_metrics['pearson']],
                        #                         'oof_rmse':[all_metrics['spearman']],
                        #                         'top10_test_precision':all_metrics['rmse'][0][0],
                        #                         'top10_test_recal':all_metrics['rmse'][0][1],
                        #                         'top10_test_f1':all_metrics['rmse'][0][2],
                        #                         'top20_test_precision':all_metrics['rmse'][1][0],
                        #                         'top20_test_recal':all_metrics['rmse'][1][1],
                        #                         'top20_test_f1':all_metrics['rmse'][1][2],
                        #                         'top50_test_precision':all_metrics['rmse'][2][0],
                        #                         'top50_test_recal':all_metrics['rmse'][2][1],
                        #                         'top50_test_f1':all_metrics['rmse'][2][2],
                        #                         'top100_test_precision':all_metrics['rmse'][3][0],
                        #                         'top100_test_recal':all_metrics['rmse'][3][1],
                        #                         'top100_test_f1':all_metrics['rmse'][3][2],
                        #                         'top_test_frac_precision':all_metrics['rmse'][4][0],
                        #                         'top_test_frac_recal':all_metrics['rmse'][4][1],
                        #                         'top_test_frac_f1':all_metrics['rmse'][4][2],
                        #                         'bottom_test_beyond_80':all_metrics[3][0][0],
                        #                         'bottom_test_beyond_85':all_metrics[3][1][0],
                        #                         'bottom_test_beyond_90':all_metrics[3][2][0],
                        #                         'bottom_test_beyond_95':all_metrics[3][3][0],
                        #                         'bottom_test_beyond_99':all_metrics[3][4][0],
                        #                         'top_test_beyond_80':all_metrics[3][0][1],
                        #                         'top_test_beyond_85':all_metrics[3][1][1],
                        #                         'top_test_beyond_90':all_metrics[3][2][1],
                        #                         'top_test_beyond_95':all_metrics[3][3][1],
                        #                         'top_test_beyond_99':all_metrics[3][4][1],
                        #                         'fold':[fold],
                        #                         'method':['esm2_finetune']})
                if test is not None:
                    test['fold'] = fold
                    print(test)    
                    all_test_pred.append(test.copy()) 



                if args.first_fold:
                    break
            else:
                epoch = 0
                model_path = output_path + f'model{fold}.ckpt'
                state_dict = torch.load(model_path, torch.device('cuda'))
                model.load_state_dict(state_dict)
                model.eval()
                valid_loss = 0.0
                valid_num = 0
                valid_pred = []
                valid_y = []
                bar = tqdm(valid_dataloader)
                for data in bar:
                    data = data.cuda()
                    with torch.no_grad():
                        outputs = model(data)
                        valid_pred.append(outputs.detach().cpu().numpy())
                        if parallel_running:
                            y = torch.cat([d.reg_labels for d in data],dim=0).to(outputs.device)
                            # mask = torch.cat([d.label_masks for d in data],dim=0).to(outputs.device)
                        else:
                            y = data.reg_labels
                            # mask = data.label_masks
                        valid_y.append(y.detach().cpu().numpy())
                        loss = loss_tr(-outputs,-y)#(loss_tr(-outputs,-y)).mean()
                        # loss = loss.sum() / mask.sum()
                        valid_num += y.shape[0]
                        valid_loss += y.shape[0] * loss.item()
                        bar.set_description('loss: %.4f' % (loss.item()))
                valid_pred = np.concatenate(valid_pred).reshape(-1)
                valid_y = np.concatenate(valid_y).reshape(-1)
                valid_metric = valid_loss / valid_num
                if test is None:
                    valid_metrics = Metric(valid_pred,valid_y)
                else:
                    test_loss = 0.0
                    test_num = 0
                    test_pred = []
                    test_y = []
                    bar = tqdm(test_dataloader)
                    for data in bar:
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
                    valid_metrics = Metric(valid_pred,valid_y,test_pred,test_y)
                
                if obj_max*(valid_metrics['spearman']) > obj_max*best_valid_metric:
                    not_improve_epochs = 0
                    best_valid_metric = valid_metrics['spearman']
                    best_metric = valid_metrics
                    best_valid_pred = valid_pred
                    if test is not None:
                        best_test_pred = test_pred
                        Write_log(log,'[epoch %s] valid_loss: %.6f, valid_mean:%.6f, valid_spearman: %.6f, valid_rmse: %.6f'%(epoch,valid_metric,np.mean(valid_pred),valid_metrics['spearman'],valid_metrics['rmse']))
                        Write_log(log,'top10_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['rmse'][0][0],valid_metrics['rmse'][0][1],valid_metrics['rmse'][0][2]))
                        Write_log(log,'top20_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['rmse'][1][0],valid_metrics['rmse'][1][1],valid_metrics['rmse'][1][2]))
                        Write_log(log,'top50_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['rmse'][2][0],valid_metrics['rmse'][2][1],valid_metrics['rmse'][2][2]))
                        Write_log(log,'top100_test precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['rmse'][3][0],valid_metrics['rmse'][3][1],valid_metrics['rmse'][3][2]))
                        Write_log(log,'top_test_frac precision: %.6f, recall: %.6f, f1: %.6f'%(valid_metrics['rmse'][4][0],valid_metrics['rmse'][4][1],valid_metrics['rmse'][4][2]))
 
                # oof = train.loc[val_idxs]
                fold_valid['pred'] = best_valid_pred
                all_oof.append(oof.copy())
                if test is None:
                    all_metrics = best_metric
                    Write_log(log,'oof_spearman: %.6f, oof_rmse: %.6f'%(all_metrics['pearson'],all_metrics['spearman']))
                    metric_df = pd.DataFrame({'DMS_id':name,
                                            'split':[args.split],
                                            'oof_spearman':[all_metrics['pearson']],
                                            'oof_rmse':[all_metrics['spearman']],
                                            'fold':[fold],
                                            'method':['esm2_add_head_supervised']}) 
                else:
                    test['pred'] = best_test_pred
                    all_metrics = best_metric
                    Write_log(log,'oof_spearman: %.6f, oof_rmse: %.6f'%(all_metrics['pearson'],all_metrics['spearman']))
                    Write_log(log,'top10_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][0][0],all_metrics['rmse'][0][1],all_metrics['rmse'][0][2]))
                    Write_log(log,'top20_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][1][0],all_metrics['rmse'][1][1],all_metrics['rmse'][1][2]))
                    Write_log(log,'top50_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][2][0],all_metrics['rmse'][2][1],all_metrics['rmse'][2][2]))
                    Write_log(log,'top100_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][3][0],all_metrics['rmse'][3][1],all_metrics['rmse'][3][2]))
                    Write_log(log,'top_test_frac precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][4][0],all_metrics['rmse'][4][1],all_metrics['rmse'][4][2]))
                    metric_df = pd.DataFrame({'DMS_id':name,
                                            'split':[args.split],
                                            'oof_spearman':[all_metrics['pearson']],
                                            'oof_rmse':[all_metrics['spearman']],
                                            'top10_test_precision':all_metrics['rmse'][0][0],
                                            'top10_test_recal':all_metrics['rmse'][0][1],
                                            'top10_test_f1':all_metrics['rmse'][0][2],
                                            'top20_test_precision':all_metrics['rmse'][1][0],
                                            'top20_test_recal':all_metrics['rmse'][1][1],
                                            'top20_test_f1':all_metrics['rmse'][1][2],
                                            'top50_test_precision':all_metrics['rmse'][2][0],
                                            'top50_test_recal':all_metrics['rmse'][2][1],
                                            'top50_test_f1':all_metrics['rmse'][2][2],
                                            'top100_test_precision':all_metrics['rmse'][3][0],
                                            'top100_test_recal':all_metrics['rmse'][3][1],
                                            'top100_test_f1':all_metrics['rmse'][3][2],
                                            'top_test_frac_precision':all_metrics['rmse'][4][0],
                                            'top_test_frac_recal':all_metrics['rmse'][4][1],
                                            'top_test_frac_f1':all_metrics['rmse'][4][2],
                                            'fold':[fold],
                                            'method':['esm2_add_head_supervised']})
                test['fold'] = fold
                print(test)    
                all_test_pred.append(test.copy()) 
                all_metric_df.append(metric_df) 
                if args.first_fold:
                    break
            # print(best_valid_pred.mean(),best_test_pred.mean())
            # oof[val_idxs] = best_valid_pred
            # oof_fold[val_idxs] = fold
            # pred += best_test_pred / folds
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
        # all_metric_df = pd.concat(all_metric_df)
        # all_metric_df.to_csv(output_path+'metric.csv',index=False)
            
            

        '''
        train['pred'] = oof
        train['fold'] = oof_fold
        if test is None:
            all_metrics = Metric(train['pred'].values,train['DMS_score'].values)
            Write_log(log,'oof_spearman: %.6f, oof_rmse: %.6f'%(all_metrics['pearson'],all_metrics['spearman']))
            metric_df = pd.DataFrame({'DMS_id':name,
                                    'split':[args.split],
                                    'oof_spearman':[all_metrics['pearson']],
                                    'oof_rmse':[all_metrics['spearman']],
                                    'method':['esm2_add_head_supervised']})
        else:
            test['pred'] = pred
            all_metrics = Metric(train['pred'].values,train['DMS_score'].values,test['pred'].values,test['DMS_score'].values)
            Write_log(log,'oof_spearman: %.6f, oof_rmse: %.6f'%(all_metrics['pearson'],all_metrics['spearman']))
            Write_log(log,'top10_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][0][0],all_metrics['rmse'][0][1],all_metrics['rmse'][0][2]))
            Write_log(log,'top20_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][1][0],all_metrics['rmse'][1][1],all_metrics['rmse'][1][2]))
            Write_log(log,'top50_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][2][0],all_metrics['rmse'][2][1],all_metrics['rmse'][2][2]))
            Write_log(log,'top100_test precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][3][0],all_metrics['rmse'][3][1],all_metrics['rmse'][3][2]))
            Write_log(log,'top_test_frac precision: %.6f, recall: %.6f, f1: %.6f'%(all_metrics['rmse'][4][0],all_metrics['rmse'][4][1],all_metrics['rmse'][4][2]))
            metric_df = pd.DataFrame({'DMS_id':name,
                                    'split':[args.split],
                                    'oof_spearman':[all_metrics['pearson']],
                                    'oof_rmse':[all_metrics['spearman']],
                                    'top10_test_precision':all_metrics['rmse'][0][0],
                                    'top10_test_recal':all_metrics['rmse'][0][1],
                                    'top10_test_f1':all_metrics['rmse'][0][2],
                                    'top20_test_precision':all_metrics['rmse'][1][0],
                                    'top20_test_recal':all_metrics['rmse'][1][1],
                                    'top20_test_f1':all_metrics['rmse'][1][2],
                                    'top50_test_precision':all_metrics['rmse'][2][0],
                                    'top50_test_recal':all_metrics['rmse'][2][1],
                                    'top50_test_f1':all_metrics['rmse'][2][2],
                                    'top100_test_precision':all_metrics['rmse'][3][0],
                                    'top100_test_recal':all_metrics['rmse'][3][1],
                                    'top100_test_f1':all_metrics['rmse'][3][2],
                                    'top_test_frac_precision':all_metrics['rmse'][4][0],
                                    'top_test_frac_recal':all_metrics['rmse'][4][1],
                                    'top_test_frac_f1':all_metrics['rmse'][4][2],
                                    'method':['esm2_add_head_supervised']})
            test.to_csv(output_path+'pred.csv',index=False)
        train.to_csv(output_path+'oof.csv',index=False)
        metric_df.to_csv(output_path+'metric.csv',index=False)
        '''
        log.close()
        # log_df = pd.DataFrame({'run_id':[run_id],'loss':[best_valid_metric],'remark':args.remark})

        # log_df.to_csv('./output/experiment.csv',index=False,mode='a')

'''
if args.predict:
    test = pd.read_feather(f'{args.root}/test.feather')
    models = []
    for fold in range(folds):
        model = DEMEmodel.DEME(si_dim,ss_dim,N_cycle,m_dim,s_dim,z_dim,n_head,c,n_layer,docheck,block_per_check,drop_rate=0.)
        model.cuda()
        if os.path.exists(output_path + f'model{fold}.ckpt'):
            model_path = output_path + f'model{fold}.ckpt'
            state_dict = torch.load(model_path, torch.device('cuda'))
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
    # model = DEMEmodel.DEME(si_dim,ss_dim,N_cycle,m_dim,s_dim,z_dim,n_head,c,n_layer,docheck,block_per_check,drop_rate=0.)
    # model.cuda()
    # model_path = output_path + f'model{fold}.ckpt'
    # state_dict = torch.load(model_path, torch.device('cuda'))
    # model.load_state_dict(state_dict)
    # model.eval()
    # models.append(model)

    all_idx = []
    all_pred = []
    for length,group in test.groupby('length',sort=False):
        print(f'length: {length} inference start')
        test_dataset = TaskDataset(group,None,list(group.index),predict=True)
        test_dataloader = DataLoader(dataset=test_dataset,batch_size=bs,shuffle=False, drop_last=False,num_workers=8)
        for data in tqdm(test_dataloader):
            data = data.cuda()
            with torch.no_grad():
                outputs = torch.stack([m(data,args) for m in models],dim=0).mean(0)
            # pred = torch.cat([data.idx[:,None],outputs.view(-1,2)],dim=1)
            all_idx.extend(data.idx.detach().cpu().numpy())
            all_pred.append(outputs.view(-1,2).detach().cpu().numpy())
        print(f'length: {length} inference end')
    all_pred = np.concatenate(all_pred,axis=0)
    sub = pd.DataFrame(data=all_pred,columns=['reactivity_DMS_MaP','reactivity_2A3_MaP'])
    sub.insert(0,'id',all_idx)
    print(sub)
    # sub['id'] = sub['id'].astype(int)
    # print(sub)
    # if not os.path.exists(output_path+'submission.csv.zip'):
    sub.to_csv(output_path+'submission.csv.zip',index=False, compression='zip')

        
    # if 'tmp' in output_path:
    #     os.rename(output_path,output_root+run_id+'/')
'''