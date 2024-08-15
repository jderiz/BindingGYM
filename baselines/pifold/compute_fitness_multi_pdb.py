import argparse
import os.path
import pandas as pd 


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
    
    from prodesign_model import ProDesign_Model
    from utils import cuda, parsePDB, parse_PDB, _scores
    from featurizer import featurize_GTrans
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
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21)))    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device) 

    model = ProDesign_Model(args)
    model.to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    criterion = nn.CrossEntropyLoss()
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
            if 'pdb_file' in g.columns:
                pdb_file = args.structure_folder + os.sep + g['pdb_file'].values[0]
            if pdb_file in all_pdb_dict_list:
                pdb_dict_list = all_pdb_dict_list[pdb_file]
            else:
                pdb_dict_list = parse_PDB(pdb_file, input_chain_list=chain_ids)
                all_pdb_dict_list[pdb_file] = pdb_dict_list
           
            batch = featurize_GTrans(pdb_dict_list)
            X, S, score, mask, lengths = cuda(batch, device = device)
            X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = model._get_features(S, score, X=X, mask=mask)
           
            log_probs = model(h_V, h_E, E_idx, batch_id)
            
            all_mt_seq = []
            design_score_list = []
            global_score_list = []
            global_seq_list = []
            for i in g.index:
                mutated_sequence = eval(g.loc[i,'mutated_sequence'])
                start_idx = 0
                mt_seq = ''
                for i,chain_id in enumerate(chain_ids):
                    seq = mutated_sequence[chain_id].replace('X','')
                    mt_seq += seq
                    input_seq_length = len(seq)
                    S_input = torch.tensor([alphabet_dict[AA] for AA in seq], device=device)
                    S[start_idx:start_idx+input_seq_length] = S_input #assumes that S and S_input are alphabetically sorted for masked_chains
                    start_idx += input_seq_length

                
                global_scores = _scores(S, log_probs)
                global_native_score = global_scores.cpu().numpy()
      
                global_score_list.append(-1*global_native_score) 
                

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
    
     # method parameters
    argparser.add_argument('--method', default='ProDesign', choices=['ProDesign'])
    argparser.add_argument('--config_file', '-c', default=None, type=str)
    argparser.add_argument('--hidden_dim',  default=128, type=int)
    argparser.add_argument('--node_features',  default=128, type=int)
    argparser.add_argument('--edge_features',  default=128, type=int)
    argparser.add_argument('--k_neighbors',  default=30, type=int)
    argparser.add_argument('--dropout',  default=0.1, type=int)
    argparser.add_argument('--num_encoder_layers', default=10, type=int)

    # ProDesign parameters
    argparser.add_argument('--updating_edges', default=4, type=int)
    argparser.add_argument('--node_dist', default=1, type=int)
    argparser.add_argument('--node_angle', default=1, type=int)
    argparser.add_argument('--node_direct', default=1, type=int)
    argparser.add_argument('--edge_dist', default=1, type=int)
    argparser.add_argument('--edge_angle', default=1, type=int)
    argparser.add_argument('--edge_direct', default=1, type=int)
    argparser.add_argument('--virtual_num', default=3, type=int)
    args = argparser.parse_args()    
    main(args)   
