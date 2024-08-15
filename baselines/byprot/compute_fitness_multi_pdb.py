import argparse
import os.path
import pandas as pd 

from pathlib import Path
import torch
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from collections import namedtuple

GenOut = namedtuple(
    'GenOut', 
    ['output_tokens', 'output_scores', 'attentions']
)

def _scores(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1,log_probs.size(-1)),
        S.contiguous().view(-1)
    ).view(S.size())
    scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return scores
def _S_to_seq(S, chain_mask, decoder):
    seq = decoder(torch.tensor([[aa for i, aa in enumerate(S) if chain_mask[i]>0]]))[0]
    return seq
def main(args):

    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import random_split, Subset
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess

    from omegaconf import OmegaConf, DictConfig

    from byprot import utils
    from byprot.datamodules.datasets import Alphabet, DataProcessor
    from byprot.utils import io
    from byprot.utils.config import compose_config as Cfg
    from byprot.tasks.fixedbb.scorer import Scorer
    from byprot.models.fixedbb.generator import IterativeRefinementGenerator

    if args.seed:
        seed=args.seed
    else:
        seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   
    
    hidden_dim = 128
    num_layers = 3 
    
    if '.csv' not in args.dms_input:
        df = pd.read_csv(args.dms_mapping)
        df['chain_id'] = df['chain_id'].fillna('')
        DMS_id = df.iloc[args.dms_index]['DMS_id']
        DMS_filename = df.iloc[args.dms_index]['DMS_filename']
        output_filename = args.dms_output + os.sep + DMS_id + ".csv"
        pdb_file = args.structure_folder + os.sep + df.iloc[args.dms_index]['pdb_file']
        df = pd.read_csv(f'{args.dms_input}/{DMS_filename}')
    else:
        df = pd.read_csv(args.dms_input)
        df['mutated_sequence'] = df['mutated_sequence'].fillna('{}')
        df['chain_id'] = df['chain_id'].fillna('')
        output_filename = args.dms_output + os.sep + os.path.basename(args.dms_input)
    if not os.path.exists(args.dms_output):
        os.makedirs(args.dms_output)

    checkpoint_path = args.model_location

    NUM_BATCHES = args.num_seq_per_target
    BATCH_COPIES = args.batch_size
    temperatures = [float(item) for item in args.sampling_temp.split()]
    omit_AAs_list = args.omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21)))    
    print_all = args.suppress_print == 0 
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if os.path.isfile(args.chain_id_jsonl):
        with open(args.chain_id_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            chain_id_dict = json.loads(json_str)
    else:
        chain_id_dict = None
        if print_all:
            print(40*'-')
            print('chain_id_jsonl is NOT loaded')
        
    if os.path.isfile(args.fixed_positions_jsonl):
        with open(args.fixed_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('fixed_positions_jsonl is NOT loaded')
        fixed_positions_dict = None
    
    
    if os.path.isfile(args.pssm_jsonl):
        with open(args.pssm_jsonl, 'r') as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        if print_all:
            print(40*'-')
            print('pssm_jsonl is NOT loaded')
        pssm_dict = None
    
    
    if os.path.isfile(args.omit_AA_jsonl):
        with open(args.omit_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            omit_AA_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('omit_AA_jsonl is NOT loaded')
        omit_AA_dict = None
    
    
    if os.path.isfile(args.bias_AA_jsonl):
        with open(args.bias_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('bias_AA_jsonl is NOT loaded')
        bias_AA_dict = None
    
    
    if os.path.isfile(args.tied_positions_jsonl):
        with open(args.tied_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('tied_positions_jsonl is NOT loaded')
        tied_positions_dict = None

    
    if os.path.isfile(args.bias_by_res_jsonl):
        with open(args.bias_by_res_jsonl, 'r') as json_file:
            json_list = list(json_file)
    
        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
        if print_all:
            print('bias by residue dictionary is loaded')
    else:
        if print_all:
            print(40*'-')
            print('bias by residue dictionary is not loaded, or not provided')
        bias_by_res_dict = None
   

    if print_all: 
        print(40*'-')
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                    if AA in list(bias_AA_dict.keys()):
                            bias_AAs_np[n] = bias_AA_dict[AA]
    
    dp = DataProcessor()
    cfg = Cfg(
        cuda=True,
        generator=Cfg(
            max_iter=5,
            strategy='denoise', 
            temperature=0,
            eval_sc=False,  
        )
    )

    scorer = Scorer(experiment_path=checkpoint_path, cfg=cfg)

    # Timing
    start_time = time.time()
    total_residues = 0
    protein_list = []
    total_step = 0
    all_pdb_dict_list = {}
    with torch.no_grad():
        
        # df = df.sample(10).reset_index(drop=True)
        print(df)
        randn_1_dic = {}
        all_g = []
        for (POI,chain_id),g in df.groupby(['POI','chain_id']):
            # if df.loc[i,'pdb_file'] != '1JTG.pdb':continue
            # print('chain_id:',df.loc[i,'chain_id'])
            
            chain_ids = g['chain_id'].values[0]
            
            designed_chain_list = []
            for i in g.index:
                mutants = eval(g.loc[i,'mutant'])
                for c in mutants:
                    if c not in designed_chain_list:
                        if mutants[c] != '':
                            designed_chain_list.append(c)
            designed_chain_list = sorted(designed_chain_list)
            print(designed_chain_list)
            if 'pdb_file' in g.columns:
                pdb_file = args.structure_folder + os.sep + g['pdb_file'].values[0]            
            if pdb_file in all_pdb_dict_list:
                raw_batch = all_pdb_dict_list[pdb_file]
            else:
                raw_batch = dp.parse_PDB(pdb_file, input_chain_list=chain_ids, masked_chain_list=designed_chain_list, ca_only=False)
                all_pdb_dict_list[pdb_file] = raw_batch
            
            batch = scorer._featurize([copy.deepcopy(raw_batch)])
            wt_dic = eval(g.loc[i,'wildtype_sequence'])
            name_ = raw_batch['name']
            if POI not in randn_1_dic:
                randn_1 = torch.randn(batch['chain_mask'].shape, device=device)
                randn_1 = torch.argsort((batch['chain_mask'] + 0.0001) * (torch.abs(randn_1)))
                randn_1_dic[POI] = randn_1
            else:
                randn_1 = randn_1_dic[POI]
            all_mt_seq = []
            design_score_list = []
            global_score_list = []
            global_seq_list = []
            for i in g.index:
                mutated_sequence = eval(g.loc[i,'mutated_sequence'])
            
                start_idx = 1
                mt_seq = ''
                if len(chain_ids) > 0:
                    for i,chain_id in enumerate(designed_chain_list):
                        seq = mutated_sequence[chain_id]
                        mt_seq += seq
                        input_seq_length = len(seq)
                        S_input = torch.tensor(scorer.alphabet.encode(seq), device=device)
                        batch['prev_tokens'][:,start_idx:start_idx+input_seq_length] = S_input #assumes that S and S_input are alphabetically sorted for masked_chains
                        start_idx += input_seq_length

                log_probs = scorer.score(batch,decoding_order=randn_1)
                mask_for_loss = batch['coord_mask']*batch['chain_mask']
                scores = _scores(batch['prev_tokens'], log_probs, mask_for_loss)
                native_score = scores.cpu().numpy()
                global_scores = _scores(batch['prev_tokens'], log_probs, batch['coord_mask'])
                global_native_score = global_scores.cpu().numpy()
                ns_mean = native_score.mean()
                ns_mean_print = np.format_float_positional(np.float32(ns_mean), unique=False, precision=4)
                ns_std = native_score.std()
                ns_std_print = np.format_float_positional(np.float32(ns_std), unique=False, precision=4)

                global_ns_mean = global_native_score.mean()
                global_ns_mean_print = np.format_float_positional(np.float32(global_ns_mean), unique=False, precision=4)
                global_ns_std = global_native_score.std()
                global_ns_std_print = np.format_float_positional(np.float32(global_ns_std), unique=False, precision=4)

                ns_sample_size = native_score.shape[0]
                design_score_list.append(-1*ns_mean)
                # global_score_list.append(-1*global_native_score[0]) # 为什么取第0个?
                global_score_list.append(-1*global_ns_mean)
                # for i in range(len(S)):
                # seq_str = _S_to_seq(S[0,], chain_M[0,])
                # all_mt_seq.append(mt_seq)
                # global_seq_list.append(seq_str)
                if print_all:
                    # if fc == 0:
                    print(f'Score for {name_} from PDB, mean: {ns_mean_print}, std: {ns_std_print}, sample size: {ns_sample_size},  global score, mean: {global_ns_mean_print}, std: {global_ns_std_print}, sample size: {ns_sample_size}')
                    # else:
                    #     print(f'Score for {name_}_{fc} from FASTA, mean: {ns_mean_print}, std: {ns_std_print}, sample size: {ns_sample_size},  global score, mean: {global_ns_mean_print}, std: {global_ns_std_print}, sample size: {ns_sample_size}')
        
            # for i in range(len(g)):
            #     for j,aa in enumerate(all_mt_seq[i]):
            #         if global_seq_list[i][j] != 'X':
            #             assert global_seq_list[i][j] == aa
            g['design_score'] = design_score_list
            g['global_score'] = global_score_list
            all_g.append(g)
        df = pd.concat(all_g).sort_index()
        df.to_csv(output_filename, index=False)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('--dms_mapping',type=str,help='path to DMS reference file')
    argparser.add_argument('--dms_input',type=str,help="path to folder containing DMS data")
    argparser.add_argument('--structure_folder',type=str,help='folder containing pdb files for each DMS')
    argparser.add_argument('--dms_index',type=int,help='index of DMS in DMS reference file')
    argparser.add_argument('--model_location',type=str,help='path to model')
    argparser.add_argument("--suppress_print", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--seed", type=int, default=0, help="If set to 0 then a random seed will be picked;")
    argparser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    argparser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
    argparser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")    
    argparser.add_argument("--dms_output", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed")
    argparser.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
    argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl")
    argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
    argparser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
    argparser.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
    argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    argparser.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    argparser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    argparser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    argparser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")
    
    args = argparser.parse_args()    
    main(args)   
