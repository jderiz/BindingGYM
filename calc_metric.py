import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from sklearn.metrics import roc_auc_score,matthews_corrcoef,ndcg_score,average_precision_score

folds = 5
seed = 42
top_test_frac = 0.1

bindingGYM = pd.read_csv('./input/BindingGYM.csv')



def calc_two_extreme_metric(train,bottom_test,top_test,pred_col):
    ms = {}
    train = train.loc[(train['top_frac']>=top_test_frac)&(train['top_frac']<(1-top_test_frac))].reset_index(drop=True)
    valid = train
    preds = valid[pred_col].values
    bottom_preds = bottom_test[pred_col].values
    top_preds = top_test[pred_col].values
    all_preds = np.concatenate([bottom_preds,preds,top_preds])
    # n = int(len(top_labels)*0.1/args.top_test_frac)
    n = len(top_preds)
    m = len(bottom_preds)
    top_pred_idxs = np.argsort(all_preds)[-(len(all_preds)-n):]
    top_idxs = np.arange(n)
    for k in [10,20,50,100]:
        hit = (top_pred_idxs[-min(k,n):]>=(len(all_preds)-n)).mean()
#         print(k,hit)
        if f'TopHit@{k}' not in ms:
            ms[f'TopHit@{k}'] = hit 
        else:
            ms[f'TopHit@{k}'] += hit

        hit = (top_pred_idxs[:min(k,n)][:min(k,m)]<m).mean()
        if f'BottomHit@{k}' not in ms:
            ms[f'BottomHit@{k}'] = hit 
        else:
            ms[f'BottomHit@{k}'] += hit 
    for k in [10,20,50,100]:
        ms[f'UnbiasHit@{k}'] = ms[f'TopHit@{k}'] - ms[f'BottomHit@{k}']
    return ms

def calc_zero_shot_metric(df,pred_col,label_col='DMS_score',top_test=True):
    label_bin = (df[label_col]>np.percentile(df[label_col].values,90))+0
    pred_bin = (df[pred_col]>np.percentile(df[pred_col].values,90))+0
    Spearman = df[label_col].rank().corr(df[pred_col].rank())
    AUC = roc_auc_score(label_bin,df[pred_col])
    MCC = matthews_corrcoef(label_bin,pred_bin)
    NDCG = ndcg_score(df[label_col].rank().values.reshape(1,-1),df[pred_col].values.reshape(1,-1),k=df.shape[0]//10)
    AP = average_precision_score(label_bin,df[pred_col])
    ms = {'Spearman':Spearman,
            'AUC':AUC,
            'MCC':MCC,
            'NDCG':NDCG,
           'AP':AP}
    if top_test:
        train = df.sort_values(by=label_col)
        train['rank'] = np.arange(0,train.shape[0])
        train['top_frac'] = train['rank'] / train.shape[0]
        bottom_test = train.loc[(train['top_frac']<(top_test_frac))].reset_index(drop=True)
        top_test = train.loc[(train['top_frac']>=(1-top_test_frac))].reset_index(drop=True)
        ms.update(calc_two_extreme_metric(train,bottom_test,top_test,pred_col))
    return ms

def get_mutant_count(x):
    n = 0
    for c in x:
        if x[c] != '':
            n += len(x[c].split(':'))
    return n

def get_zero_shot_metric_df(path,pred_col,contig=False,mod=False):
    zero_shot_metric = {}
    for DMS_id in bindingGYM['DMS_id'].values:
        orig_df = pd.read_csv(f'./input/Binding_substitutions_DMS/{DMS_id}.csv')
        print(DMS_id)
        df = pd.read_csv(f'{path}/{DMS_id}.csv')
        
        assert df.shape[0] == orig_df.shape[0]
        if contig or mod:
            df['mutant'] = df['mutant'].apply(eval)
            df = df.loc[df['mutant'].apply(get_mutant_count)<2].reset_index(drop=True)
            if df.shape[0] < 100:
                continue
        zero_shot_metric[DMS_id] = calc_zero_shot_metric(df,pred_col)
    zero_shot_metric_df = pd.DataFrame(zero_shot_metric.values())
    zero_shot_metric_df.insert(0,'DMS_id',zero_shot_metric.keys())
    return zero_shot_metric_df


ProteinMPNN_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/proteinmpnn/output','global_score')
ProteinMPNN_zero_shot_metric_df.to_csv('./results/ProteinMPNN_zero_shot_metric.csv',index=False)

ProteinMPNN_contig_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/proteinmpnn/output','global_score',contig=True)
ProteinMPNN_contig_zero_shot_metric_df.to_csv('./results/ProteinMPNN_zero_shot_intra_contig_metric.csv',index=False)

ProteinMPNN_mod_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/proteinmpnn/output','global_score',mod=True)
ProteinMPNN_mod_zero_shot_metric_df.to_csv('./results/ProteinMPNN_zero_shot_intra_mod_metric.csv',index=False)

ESM_if1_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/esm-if1/output','esm_if1')
ESM_if1_zero_shot_metric_df.to_csv('./results/ESM-if1_zero_shot_metric.csv',index=False)

PPIformer_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/ppiformer/output','ddG')
PPIformer_zero_shot_metric_df.to_csv('./results/PPIformer_zero_shot_metric.csv',index=False)

SaProt_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/saprot/output','SaProt_650M_AF2')
SaProt_zero_shot_metric_df.to_csv('./results/SaProt_zero_shot_metric.csv',index=False)

ProGen2_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/progen2/output','Progen2_score')
ProGen2_zero_shot_metric_df.to_csv('./results/ProGen2_zero_shot_metric.csv',index=False)

ESM1v_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/esm-1v/output','Ensemble_ESM1v')
ESM1v_zero_shot_metric_df.to_csv('./results/ESM1v_zero_shot_metric.csv',index=False)

ESM2_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/esm2/output','esm2_t33_650M_UR50D')
ESM2_zero_shot_metric_df.to_csv('./results/ESM2_zero_shot_metric.csv',index=False)

EVE_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/eve/output','evol_indices_seed_42')
EVE_zero_shot_metric_df.to_csv('./results/EVE_zero_shot_metric.csv',index=False)

Tranception_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/tranception/output','avg_score')
Tranception_zero_shot_metric_df.to_csv('./results/Tranception_zero_shot_metric.csv',index=False)

TranceptEVE_zero_shot_metric_df = get_zero_shot_metric_df('./modelzoo/trancepteve/output','avg_score')
TranceptEVE_zero_shot_metric_df.to_csv('./results/TranceptEVE_zero_shot_metric.csv',index=False)

def get_finetune_intra_random_metric_df(path,pred_col,model_type):
    zero_shot_metric = {}
    oneORtwo_mut_metric = {}
    multi_mut_metric = {}
    for DMS_id in bindingGYM['DMS_id'].values:
        print(DMS_id)
        train = pd.read_csv(f'./input/Binding_substitutions_DMS/{DMS_id}.csv')
        df = pd.read_csv(f'{path}/train_on_{DMS_id}_intra_random_{model_type}_seed{seed}/oof.csv')
        assert train.shape[0] == df.shape[0]
        assert df['fold'].isna().sum()==0
        zero_shot_metric[DMS_id] = calc_zero_shot_metric(df,pred_col,top_test=False)

        oneORtwo_df = df.loc[df['mutant'].fillna('').apply(lambda x:len(x.split(':'))<3)].reset_index(drop=True)
        print(oneORtwo_df.shape)
        if oneORtwo_df.shape[0] >= 100:
            oneORtwo_mut_metric[DMS_id] = calc_zero_shot_metric(oneORtwo_df,pred_col,top_test=False)

        multi_df = df.loc[df['mutant'].fillna('').apply(lambda x:len(x.split(':'))>=3)].reset_index(drop=True)
        print(multi_df.shape)
        if multi_df.shape[0] >= 100:
            multi_mut_metric[DMS_id] = calc_zero_shot_metric(multi_df,pred_col,top_test=False)

    zero_shot_metric_df = pd.DataFrame(zero_shot_metric.values())
    zero_shot_metric_df.insert(0,'DMS_id',zero_shot_metric.keys())
    
    oneORtwo_mut_metric_df = pd.DataFrame(oneORtwo_mut_metric.values())
    oneORtwo_mut_metric_df.insert(0,'DMS_id',oneORtwo_mut_metric.keys())
    
    multi_mut_metric_df = pd.DataFrame(multi_mut_metric.values())
    multi_mut_metric_df.insert(0,'DMS_id',multi_mut_metric.keys())
    return zero_shot_metric_df,oneORtwo_mut_metric_df,multi_mut_metric_df


ProteinMPNN_intra_random_metric_df,ProteinMPNN_intra_random_metric_oneORtwo_df,ProteinMPNN_intra_random_metric_multi_df = get_finetune_intra_random_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','structure')
ProteinMPNN_R_intra_random_metric_df,ProteinMPNN_R_intra_random_metric_oneORtwo_df,ProteinMPNN_R_intra_random_metric_multi_df = get_finetune_intra_random_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','native_structure')
ProteinMPNN_intra_random_metric_df.to_csv('./results/ProteinMPNN_finetune_intra_random_metric.csv',index=False)
ProteinMPNN_intra_random_metric_oneORtwo_df.to_csv('./results/ProteinMPNN_finetune_intra_random_metric_oneORtwo.csv',index=False)
ProteinMPNN_intra_random_metric_multi_df.to_csv('./results/ProteinMPNN_finetune_intra_random_metric_multi.csv',index=False)
ProteinMPNN_R_intra_random_metric_df.to_csv('./results/ProteinMPNN_R_finetune_intra_random_metric.csv',index=False)
ProteinMPNN_R_intra_random_metric_oneORtwo_df.to_csv('./results/ProteinMPNN_R_finetune_intra_random_metric_oneORtwo.csv',index=False)
ProteinMPNN_R_intra_random_metric_multi_df.to_csv('./results/ProteinMPNN_R_finetune_intra_random_metric_multi.csv',index=False)

ESM2_intra_random_metric_df,ESM2_intra_random_metric_oneORtwo_df,ESM2_intra_random_metric_multi_df = get_finetune_intra_random_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','sequence_lora')
ESM2_R_intra_random_metric_df,ESM2_R_intra_random_metric_oneORtwo_df,ESM2_R_intra_random_metric_multi_df = get_finetune_intra_random_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','native_sequence_lora')
ESM2_intra_random_metric_df.to_csv('./results/ESM2_finetune_intra_random_metric.csv',index=False)
ESM2_intra_random_metric_oneORtwo_df.to_csv('./results/ESM2_finetune_intra_random_metric_oneORtwo.csv',index=False)
ESM2_intra_random_metric_multi_df.to_csv('./results/ESM2_finetune_intra_random_metric_multi.csv',index=False)
ESM2_R_intra_random_metric_df.to_csv('./results/ESM2_R_finetune_intra_random_metric.csv',index=False)
ESM2_R_intra_random_metric_oneORtwo_df.to_csv('./results/ESM2_R_finetune_intra_random_metric_oneORtwo.csv',index=False)
ESM2_R_intra_random_metric_multi_df.to_csv('./results/ESM2_R_finetune_intra_random_metric_multi.csv',index=False)

OHE_intra_random_metric_df,OHE_intra_random_metric_oneORtwo_df,OHE_intra_random_metric_multi_df = get_finetune_intra_random_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','OHE')
OHE_intra_random_metric_df.to_csv('./results/OHE_finetune_intra_random_metric.csv',index=False)
OHE_intra_random_metric_oneORtwo_df.to_csv('./results/OHE_finetune_intra_random_metric_oneORtwo.csv',index=False)
OHE_intra_random_metric_multi_df.to_csv('./results/OHE_finetune_intra_random_metric_multi.csv',index=False)

from training.utils import DMS_file_for_LLM

def get_finetune_intra_contig_metric_df(path,pred_col,model_type):
    zero_shot_metric = {}
    for DMS_id in bindingGYM['DMS_id'].values:
        train = pd.read_csv(f'./input/Binding_substitutions_DMS/{DMS_id}.csv')
        train = DMS_file_for_LLM(train,focus=True)
        train = train.loc[train['mutant'].apply(lambda x:len(x.split(':'))<2)].reset_index(drop=True)
        if train.shape[0] < 100:
            continue
        print(DMS_id)
        df = pd.read_csv(f'{path}/train_on_{DMS_id}_intra_contig_{model_type}_seed{seed}/oof.csv')
        assert train.shape[0] == df.shape[0]
        assert df['fold'].isna().sum()==0
        zero_shot_metric[DMS_id] = calc_zero_shot_metric(df,pred_col,top_test=False)
        
    zero_shot_metric_df = pd.DataFrame(zero_shot_metric.values())
    zero_shot_metric_df.insert(0,'DMS_id',zero_shot_metric.keys())
    return zero_shot_metric_df

def get_finetune_intra_mod_metric_df(path,pred_col,model_type):
    zero_shot_metric = {}
    for DMS_id in bindingGYM['DMS_id'].values:
        train = pd.read_csv(f'./input/Binding_substitutions_DMS/{DMS_id}.csv')
        train = DMS_file_for_LLM(train,focus=True)
        train = train.loc[train['mutant'].apply(lambda x:len(x.split(':'))<2)].reset_index(drop=True)
        if train.shape[0] < 100:
            continue
        print(DMS_id)
        df = pd.read_csv(f'{path}/train_on_{DMS_id}_intra_mod_{model_type}_seed{seed}/oof.csv')
        assert train.shape[0] == df.shape[0]
        assert df['fold'].isna().sum()==0
        zero_shot_metric[DMS_id] = calc_zero_shot_metric(df,pred_col,top_test=False)
        
    zero_shot_metric_df = pd.DataFrame(zero_shot_metric.values())
    zero_shot_metric_df.insert(0,'DMS_id',zero_shot_metric.keys())
    return zero_shot_metric_df

OHE_intra_contig_metric_df = get_finetune_intra_contig_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','OHE')
OHE_intra_contig_metric_df.to_csv('./results/OHE_finetune_intra_contig_metric.csv',index=False)

OHE_intra_mod_metric_df = get_finetune_intra_mod_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','OHE')
OHE_intra_mod_metric_df.to_csv('./results/OHE_finetune_intra_mod_metric.csv',index=False)

ProteinMPNN_intra_contig_metric_df = get_finetune_intra_contig_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','structure')
ProteinMPNN_R_intra_contig_metric_df = get_finetune_intra_contig_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','native_structure')
ProteinMPNN_intra_contig_metric_df.to_csv('./results/ProteinMPNN_finetune_intra_contig_metric.csv',index=False)
ProteinMPNN_R_intra_contig_metric_df.to_csv('./results/ProteinMPNN_R_finetune_intra_contig_metric.csv',index=False)
ProteinMPNN_intra_mod_metric_df = get_finetune_intra_mod_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','structure')
ProteinMPNN_R_intra_mod_metric_df = get_finetune_intra_mod_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','native_structure')
ProteinMPNN_intra_mod_metric_df.to_csv('./results/ProteinMPNN_finetune_intra_mod_metric.csv',index=False)
ProteinMPNN_R_intra_mod_metric_df.to_csv('./results/ProteinMPNN_R_finetune_intra_mod_metric.csv',index=False)

ESM2_intra_contig_metric_df = get_finetune_intra_contig_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','sequence_lora')
ESM2_R_intra_contig_metric_df = get_finetune_intra_contig_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','native_sequence_lora')
ESM2_intra_contig_metric_df.to_csv('./results/ESM2_finetune_intra_contig_metric.csv',index=False)
ESM2_R_intra_contig_metric_df.to_csv('./results/ESM2_R_finetune_intra_contig_metric.csv',index=False)
ESM2_intra_mod_metric_df = get_finetune_intra_mod_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','sequence_lora')
ESM2_R_intra_mod_metric_df = get_finetune_intra_mod_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','native_sequence_lora')
ESM2_intra_mod_metric_df.to_csv('./results/ESM2_finetune_intra_mod_metric.csv',index=False)
ESM2_R_intra_mod_metric_df.to_csv('./results/ESM2_R_finetune_intra_mod_metric.csv',index=False)

aastat_intra_contig_metric_df = get_finetune_intra_contig_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','aastat')
aastat_intra_contig_metric_df.to_csv('./results/aastat_finetune_intra_contig_metric.csv',index=False)

aastat_intra_mod_metric_df = get_finetune_intra_mod_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output/','pred','aastat')
aastat_intra_mod_metric_df.to_csv('./results/aastat_finetune_intra_mod_metric.csv',index=False)

def get_finetune_intra_top_test_metric_df(path,pred_col,model_type):
    zero_shot_metric = {}
    for DMS_id in bindingGYM['DMS_id'].values:
        try:
            df = pd.read_csv(f'./input/Binding_substitutions_DMS/{DMS_id}.csv')
            train = pd.read_csv(f'{path}/train_on_{DMS_id}_intra_top_test_{model_type}_seed{seed}/oof.csv')
            test = pd.read_csv(f'{path}/train_on_{DMS_id}_intra_top_test_{model_type}_seed{seed}/pred.csv')
            print(DMS_id,df.shape[0]==(train.shape[0]+test.query("fold==0").shape[0]),train['fold'].isna().sum(),train.shape,test.shape)
            bottom_test = test.loc[test['top_frac']<0.5]
            bottom_test = bottom_test.groupby('rank')[pred_col].max().reset_index()
            top_test = test.loc[test['top_frac']>0.5]
            top_test = top_test.groupby('rank')[pred_col].max().reset_index()
            
            zero_shot_metric[DMS_id] = calc_two_extreme_metric(train,bottom_test,top_test,pred_col)
        except:
            pass
    zero_shot_metric_df = pd.DataFrame(zero_shot_metric.values())
    zero_shot_metric_df.insert(0,'DMS_id',zero_shot_metric.keys())
    return zero_shot_metric_df


ProteinMPNN_intra_top_test_metric_df = get_finetune_intra_top_test_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output','pred','structure')
ProteinMPNN_intra_top_test_metric_df.to_csv('./results/ProteinMPNN_finetune_intra_top_test_metric.csv',index=False)


ProteinMPNN_R_intra_top_test_metric_df = get_finetune_intra_top_test_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output','pred','native_structure')
ProteinMPNN_R_intra_top_test_metric_df.to_csv('./results/ProteinMPNN_R_finetune_intra_top_test_metric.csv',index=False)


# In[1014]:


ESM2_intra_top_test_metric_df = get_finetune_intra_top_test_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output','pred','sequence_lora')


# In[1015]:


ESM2_intra_top_test_metric_df.mean()


# In[1016]:


ESM2_intra_top_test_metric_df.to_csv('./results/ESM2_finetune_intra_top_test_metric.csv',index=False)


# In[1022]:


ESM2_R_intra_top_test_metric_df = get_finetune_intra_top_test_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output','pred','native_sequence_lora')


# In[1023]:


ESM2_R_intra_top_test_metric_df.mean()


# In[1024]:


ESM2_R_intra_top_test_metric_df.to_csv('./results/ESM2_R_finetune_intra_top_test_metric.csv',index=False)


# In[1018]:


OHE_intra_top_test_metric_df = get_finetune_intra_top_test_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output','pred','OHE')


# In[1019]:


OHE_intra_top_test_metric_df


# In[1020]:


OHE_intra_top_test_metric_df.mean()


# In[1021]:


OHE_intra_top_test_metric_df.to_csv('./results/OHE_finetune_intra_top_test_metric.csv',index=False)


# In[903]:


def get_finetune_inter_metric_df(path,pred_col,model_type):
    zero_shot_metric = {}
    oneORtwo_mut_metric = {}
    multi_mut_metric = {}
    for DMS_id in bindingGYM['DMS_id'].values:
        print(DMS_id)
        train = pd.read_csv(f'./input/Binding_substitutions_DMS/{DMS_id}.csv')
        df = pd.read_csv(f'{path}/train_on_BindingGYM_inter_cluster_{model_type}_seed{seed}/{DMS_id}_oof.csv')
        df['a'] = df['mutant'].astype(str)
        df = df.drop_duplicates('a').reset_index(drop=True)
        assert train.shape[0] == df.shape[0]
        
        zero_shot_metric[DMS_id] = calc_zero_shot_metric(df,pred_col,top_test=False)
        
        oneORtwo_df = df.loc[df['mutant'].fillna('').apply(lambda x:len(x.split(':'))<3)].reset_index(drop=True)
        print(oneORtwo_df.shape)
        if oneORtwo_df.shape[0] >= 100:
            oneORtwo_mut_metric[DMS_id] = calc_zero_shot_metric(oneORtwo_df,pred_col,top_test=False)

        multi_df = df.loc[df['mutant'].fillna('').apply(lambda x:len(x.split(':'))>=3)].reset_index(drop=True)
        print(multi_df.shape)
        if multi_df.shape[0] >= 100:
            multi_mut_metric[DMS_id] = calc_zero_shot_metric(multi_df,pred_col,top_test=False)

    zero_shot_metric_df = pd.DataFrame(zero_shot_metric.values())
    zero_shot_metric_df.insert(0,'DMS_id',zero_shot_metric.keys())
    oneORtwo_mut_metric_df = pd.DataFrame(oneORtwo_mut_metric.values())
    oneORtwo_mut_metric_df.insert(0,'DMS_id',oneORtwo_mut_metric.keys())
    
    multi_mut_metric_df = pd.DataFrame(multi_mut_metric.values())
    multi_mut_metric_df.insert(0,'DMS_id',multi_mut_metric.keys())
    return zero_shot_metric_df,oneORtwo_mut_metric_df,multi_mut_metric_df


# In[904]:


ProteinMPNN_finetune_inter_metric_df,ProteinMPNN_finetune_inter_metric_oneORtwo_df,ProteinMPNN_finetune_inter_metric_multi_df = get_finetune_inter_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output','pred','structure')


# In[905]:


ProteinMPNN_finetune_inter_metric_df.mean()


# In[906]:


ProteinMPNN_finetune_inter_metric_oneORtwo_df.mean()


# In[832]:


ProteinMPNN_finetune_inter_metric_multi_df.mean()


# In[907]:


ProteinMPNN_finetune_inter_metric_df.to_csv('./results/ProteinMPNN_finetune_inter_cluster_metric.csv',index=False)
ProteinMPNN_finetune_inter_metric_oneORtwo_df.to_csv('./results/ProteinMPNN_finetune_inter_cluster_metric_oneORtwo.csv',index=False)
ProteinMPNN_finetune_inter_metric_multi_df.to_csv('./results/ProteinMPNN_finetune_inter_cluster_metric_multi.csv',index=False)


# In[908]:


ProteinMPNN_R_finetune_inter_metric_df,ProteinMPNN_R_finetune_inter_metric_oneORtwo_df,ProteinMPNN_R_finetune_inter_metric_multi_df = get_finetune_inter_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output','pred','native_structure')


# In[909]:


ProteinMPNN_R_finetune_inter_metric_df.mean()


# In[910]:


ProteinMPNN_R_finetune_inter_metric_oneORtwo_df.mean()


# In[911]:


ProteinMPNN_R_finetune_inter_metric_multi_df.mean()


# In[912]:


ProteinMPNN_R_finetune_inter_metric_df.to_csv('./results/ProteinMPNN_R_finetune_inter_cluster_metric.csv',index=False)
ProteinMPNN_R_finetune_inter_metric_oneORtwo_df.to_csv('./results/ProteinMPNN_R_finetune_inter_cluster_metric_oneORtwo.csv',index=False)
ProteinMPNN_R_finetune_inter_metric_multi_df.to_csv('./results/ProteinMPNN_R_finetune_inter_cluster_metric_multi.csv',index=False)


# In[913]:


ESM2_finetune_inter_metric_df,ESM2_finetune_inter_metric_oneORtwo_df,ESM2_finetune_inter_metric_multi_df = get_finetune_inter_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output','pred','sequence_lora')


# In[914]:


ESM2_finetune_inter_metric_df.mean()


# In[915]:


ESM2_finetune_inter_metric_oneORtwo_df.mean()


# In[916]:


ESM2_finetune_inter_metric_multi_df.mean()


# In[917]:


ESM2_finetune_inter_metric_df.to_csv('./results/ESM2_finetune_inter_cluster_metric.csv',index=False)
ESM2_finetune_inter_metric_oneORtwo_df.to_csv('./results/ESM2_finetune_inter_cluster_metric_oneORtwo.csv',index=False)
ESM2_finetune_inter_metric_multi_df.to_csv('./results/ESM2_finetune_inter_cluster_metric_multi.csv',index=False)


# In[918]:


ESM2_R_finetune_inter_metric_df,ESM2_R_finetune_inter_metric_oneORtwo_df,ESM2_R_finetune_inter_metric_multi_df = get_finetune_inter_metric_df('/home/zhangjx/project/aureka2_bk/DMS_finetune/train_DMS_by_structure_model_BindingGYM/output','pred','native_sequence_lora')


# In[919]:


ESM2_R_finetune_inter_metric_df.mean()


# In[920]:


ESM2_R_finetune_inter_metric_oneORtwo_df.mean()


# In[921]:


ESM2_R_finetune_inter_metric_multi_df.mean()


# In[922]:


ESM2_R_finetune_inter_metric_df.to_csv('./results/ESM2_R_finetune_inter_cluster_metric.csv',index=False)
ESM2_R_finetune_inter_metric_oneORtwo_df.to_csv('./results/ESM2_R_finetune_inter_cluster_metric_oneORtwo.csv',index=False)
ESM2_R_finetune_inter_metric_multi_df.to_csv('./results/ESM2_R_finetune_inter_cluster_metric_multi.csv',index=False)


# In[221]:


r = pd.read_csv('./train_DMS_by_structure_model_BindingGYM/output/train_on_CR6261_FluAH1_logKd_3GBN_intra_top_test_structure_seed42/oof.csv')


# In[235]:


p  = pd.read_csv('./train_DMS_by_structure_model_BindingGYM/output/train_on_CR9114_FluAH1_logKd_4FQI_intra_top_test_structure_seed42//pred.csv')


# In[236]:


p


# In[233]:


top_pred = p.query("top_frac>0.5")#['pred'].values
bottom_pred = p.query("top_frac<0.5")#['pred'].values


# In[234]:


calc_two_extreme_metric(r,bottom_test=bottom_pred,top_test=top_pred,pred_col='pred')


# In[230]:


r['pred'].hist()
p.query("top_frac>0.5")['pred'].hist()

p.query("top_frac<0.5")['pred'].hist()


# In[30]:


protein_mpnn_zero_shot_metric = {}
for DMS_id in bindingGYM['DMS_id'].values:
    df = pd.read_csv(f'./modelzoo/proteinmpnn/output/{DMS_id}.csv')
    protein_mpnn_zero_shot_metric[DMS_id] = zero_shot_metric(df,'global_score')


# In[31]:


protein_mpnn_zero_shot_metric_df = pd.DataFrame(protein_mpnn_zero_shot_metric.values())
protein_mpnn_zero_shot_metric_df.insert(0,'DMS_id',protein_mpnn_zero_shot_metric.keys())


# In[32]:


protein_mpnn_zero_shot_metric_df


# In[34]:


protein_mpnn_zero_shot_metric_df.to_csv('./results/ProteinMPNN_zero_shot_metric.csv',index=False)


# In[35]:


esmif1_zero_shot_metric = {}
for DMS_id in bindingGYM['DMS_id'].values:
    df = pd.read_csv(f'./modelzoo/esm-if1/output/{DMS_id}.csv')
    esmif1_zero_shot_metric[DMS_id] = zero_shot_metric(df,'esm_if1')


# In[ ]:


esmif1_zero_shot_metric_df = pd.DataFrame(protein_mpnn_zero_shot_metric.values())
protein_mpnn_zero_shot_metric_df.insert(0,'DMS_id',protein_mpnn_zero_shot_metric.keys())

