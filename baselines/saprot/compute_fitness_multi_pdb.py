import argparse
import pathlib
import os,sys
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from esm_loader import load_esm_saprot
from foldseek_util import get_struc_seq
from utils.scoring_utils import get_optimal_window, set_mutant_offset, undo_mutant_offset
from utils.data_utils import DMS_file_for_LLM

def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="MSA_transformer Vs ESM1v Vs ESM1b",
        default="MSA_transformer",
        nargs="+",
    )
    parser.add_argument(
        "--python",
        type=str,
    )
    parser.add_argument(
        "--model-location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--structure-path",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--dms-input",
        type=str,
        help="CSV file containing the deep mutational scan",
    )
    parser.add_argument(
        "--dms_index",
        type=int,
        help="Index of DMS in mapping file",
    )
    parser.add_argument(
        "--dms_mapping",
        type=str,
        help="Location of DMS_mapping",
    )
    parser.add_argument(
        "--structure_folder",
        type=str,
        help="Location of structures",
    )
    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
        help="column in the deep mutational scan labeling the mutation as 'AiB'"
    )
    parser.add_argument(
        "--dms-output",
        type=pathlib.Path,
        help="Output file containing the deep mutational scan along with predictions",
    )
    parser.add_argument(
        "--offset-idx",
        type=int,
        default=1,
        help="Offset of the mutation positions in `--mutation-col`"
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "pseudo-ppl", "masked-marginals"],
        help=""
    )
    parser.add_argument(
        "--msa-path",
        type=pathlib.Path,
        help="path to MSA (required for MSA Transformer)"
    )
    parser.add_argument(
        "--msa-sampling-strategy",
        type=str,
        default='sequence-reweighting',
        help="Strategy to sample sequences from MSA [sequence-reweighting|random|first_x_rows]"
    )
    parser.add_argument(
        "--msa-samples",
        type=int,
        default=400,
        help="number of sequences to randomly sample from the MSA"
    )
    parser.add_argument(
        "--msa-weights-folder",
        type=str,
        default=None,
        help="Folder with weights to sample MSA sequences in 'sequence-reweighting' scheme"
    )
    parser.add_argument(
        '--seeds',
        type=int,
        default=1, 
        help='Random seed used during training',
        nargs="+"
    )
    parser.add_argument(
        '--filter-msa',
        action='store_true',
        help='Whether to use hhfilter to filter input MSA before sampling'
    )
    parser.add_argument(
        '--hhfilter-min-cov',
        type=int,
        default=75, 
        help='minimum coverage with query (%)'
    )
    parser.add_argument(
        '--hhfilter-max-seq-id',
        type=int,
        default=90, 
        help='maximum pairwise identity (%)'
    )
    parser.add_argument(
        '--hhfilter-min-seq-id',
        type=int,
        default=0, 
        help='minimum sequence identity with query (%)'
    )
    parser.add_argument(
        '--path-to-hhfilter',
        type=str,
        default='/n/groups/marks/software/hhsuite/hhsuite-3.3.0', 
        help='Path to hhfilter binaries'
    )
    parser.add_argument(
        '--scoring-window',
        type=str,
        default='optimal', 
        help='Approach to handle long sequences [optimal|overlapping]'
    )
    parser.add_argument(
        '--overwrite-prior-scores',
        action='store_true',
        help='Whether to overwrite prior scores in the dataframe'
    )
    
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

def label_row(row, seq_info, token_probs, alphabet, offset_idx):
    score=0
    if row == '' or row == 'WT':
        return score
    seq, struc_seq, combined_seq = seq_info
    for mutation in row.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        combined_wt = wt+struc_seq[idx].lower()
        combined_mt = mt+struc_seq[idx].lower()
        assert seq[idx] == wt, "The listed wildtype does not match the provided sequence"
        assert combined_seq[2*idx:2*idx+2] == combined_wt, "The listed combined wildtype does not match the provided sequence and structure"
        wt_encoded, mt_encoded = alphabet.get_idx(combined_wt), alphabet.get_idx(combined_mt)

        # add 1 for BOS
        score += (token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]).item()
    return score


def compute_pppl(row, sequence, model, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

    # encode the sequence
    data = [
        ("protein1", sequence),
    ]

    batch_converter = alphabet.get_batch_converter()

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)


def main(args):
    print("Arguments:", args)

    # Load the deep mutational scan
    if '.csv' not in args.dms_input:
        mapping_protein_seq_DMS = pd.read_csv(args.dms_mapping)
        DMS_id = mapping_protein_seq_DMS["DMS_id"][args.dms_index]
        print("Compute scores for DMS: "+str(DMS_id))
        row = mapping_protein_seq_DMS[mapping_protein_seq_DMS["DMS_id"]==DMS_id]
        if len(row) == 0:
            raise ValueError("No mappings found for DMS: "+str(DMS_id))
        elif len(row) > 1:
            raise ValueError("Multiple mappings found for DMS: "+str(DMS_id))
        
        row = row.iloc[0]
        row = row.replace(np.nan, "")  # Makes it more manageable to use in strings

        args.dms_input = str(args.dms_input)+os.sep+row["DMS_filename"]
        args.dms_output=str(args.dms_output)+os.sep+DMS_id+'.csv'
        df = pd.read_csv(args.dms_input)
    else:
        df = pd.read_csv(args.dms_input)
        args.dms_output = str(args.dms_output)+os.sep+os.path.basename(args.dms_input)
    df 
    args.mutation_col='mutant'
    
    if len(df) == 0:
        raise ValueError("No rows found in the dataframe")
    print(f"df shape: {df.shape}", flush=True)

    # inference for each model
    print("Starting model scoring")
    model_location = args.model_location
    model, alphabet = load_esm_saprot(model_location)
    model_location = model_location.split("/")[-1].split(".")[0]
    model.eval()
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")
    else:
        print(f"Not using GPU. torch.cuda.is_available(): {torch.cuda.is_available()}, args.nogpu: {args.nogpu}")

    batch_converter = alphabet.get_batch_converter()

    all_g = []
    for POI,g in df.groupby('POI'):
        # if POI != '3NCB_A_B':continue
        wt_seq_dic = eval(g['wildtype_sequence'].values[0])
        g,focus_chains = DMS_file_for_LLM(g,focus=True,return_focus_chains=True)
        print(POI)
        wt_seq = g['wildtype_sequence'].values[0]
        
        pdb_path = args.structure_folder + os.sep + g['pdb_file'].values[0]
        seq_dic = get_struc_seq(pdb_path,wt_seq_dic=wt_seq_dic,python=args.python,chains=focus_chains,process_id=POI)
        full_seq = ''
        full_struc_seq = ''
        full_combined_seq = ''
        for chain in focus_chains:
            seq,struc_seq,combined_seq = seq_dic[chain]
            full_seq += seq
            full_struc_seq += struc_seq
            full_combined_seq += combined_seq
        seq_info = [full_seq,full_struc_seq,full_combined_seq]
        data = [
            ("protein1", full_combined_seq),
        ]
        print(data)
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        if args.scoring_strategy == "wt-marginals":
            with torch.no_grad():
                if batch_tokens.size(1) > 1024 and args.scoring_window=="overlapping": 
                    batch_size, seq_len = batch_tokens.shape #seq_len includes BOS and EOS
                    token_probs = torch.zeros((batch_size,seq_len,len(alphabet))).cuda() # Note: batch_size = 1 (need to keep batch dimension to score with model though)
                    token_weights = torch.zeros((batch_size,seq_len)).cuda()
                    weights = torch.ones(1024).cuda() # 1 for 256≤i<1022-256
                    for i in range(1,257):
                        weights[i] = 1 / (1 + math.exp(-(i-128)/16))
                    for i in range(1022-256,1023):
                        weights[i] = 1 / (1 + math.exp((i-1022+128)/16))
                    start_left_window = 0
                    end_left_window = 1023 #First window is indexed [0-1023]
                    start_right_window = (batch_tokens.size(1) - 1) - 1024 + 1 #Last index is len-1
                    end_right_window = batch_tokens.size(1) - 1
                    while True: 
                        # Left window update
                        left_window_probs = torch.log_softmax(model(batch_tokens[:,start_left_window:end_left_window+1].cuda())["logits"], dim=-1)
                        token_probs[:,start_left_window:end_left_window+1] += left_window_probs * weights.view(-1,1)
                        token_weights[:,start_left_window:end_left_window+1] += weights
                        # Right window update
                        right_window_probs = torch.log_softmax(model(batch_tokens[:,start_right_window:end_right_window+1].cuda())["logits"], dim=-1)
                        token_probs[:,start_right_window:end_right_window+1] += right_window_probs * weights.view(-1,1)
                        token_weights[:,start_right_window:end_right_window+1] += weights
                        if end_left_window > start_right_window:
                            #overlap between windows in that last scoring so we break from the loop
                            break
                        start_left_window+=511
                        end_left_window+=511
                        start_right_window-=511
                        end_right_window-=511
                    #If central overlap not wide engouh, we add one more window at the center
                    final_overlap = end_left_window - start_right_window + 1
                    if final_overlap < 511:
                        start_central_window = int(seq_len / 2) - 512
                        end_central_window = start_central_window + 1023
                        central_window_probs = torch.log_softmax(model(batch_tokens[:,start_central_window:end_central_window+1].cuda())["logits"], dim=-1)
                        token_probs[:,start_central_window:end_central_window+1] += central_window_probs * weights.view(-1,1)
                        token_weights[:,start_central_window:end_central_window+1] += weights
                    #Weight normalization
                    token_probs = token_probs / token_weights.view(-1,1) #Add 1 to broadcast
                else:                    
                    token_probs = torch.log_softmax(model(batch_tokens.cuda())["logits"], dim=-1)
            g[model_location] = g.apply(
                lambda row: label_row(
                    row[args.mutation_col],
                    seq_info,
                    token_probs,
                    alphabet,
                    args.offset_idx,
                ),
                axis=1,
            )
        elif args.scoring_strategy == "masked-marginals":
            all_token_probs = []
            for i in tqdm(range(batch_tokens.size(1))):
                batch_tokens_masked = batch_tokens.clone()
                batch_tokens_masked[0, i] = alphabet.mask_idx
                if batch_tokens.size(1) > 1024 and args.scoring_window=="optimal": 
                    large_batch_tokens_masked=batch_tokens_masked.clone()
                    start, end = get_optimal_window(mutation_position_relative=i, seq_len_wo_special=len(wt_seq)+2, model_window=1024)
                    batch_tokens_masked = large_batch_tokens_masked[:,start:end]
                elif batch_tokens.size(1) > 1024 and args.scoring_window=="overlapping": 
                    print("Overlapping not yet implemented for masked-marginals")
                    sys.exit(0)
                else:
                    start=0
                with torch.no_grad():
                    token_probs = torch.log_softmax(
                        model(batch_tokens_masked.cuda())["logits"], dim=-1
                    )
                all_token_probs.append(token_probs[:, i-start])  # vocab size
            token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
            g[model_location] = g.apply(
                lambda row: label_row(
                    row[args.mutation_col],
                    seq_info,
                    token_probs,
                    alphabet,
                    args.offset_idx,
                ),
                axis=1,
            )
        # elif args.scoring_strategy == "pseudo-ppl":
        #     tqdm.pandas()
        #     g[model_location] = g.progress_apply(
        #         lambda row: compute_pppl(
        #             row[args.mutation_col], args.sequence, model, alphabet, args.offset_idx
        #         ),
        #         axis=1,
        #     )
        all_g.append(g)
    df = pd.concat(all_g).sort_index()
    df.to_csv(args.dms_output,index=False)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)