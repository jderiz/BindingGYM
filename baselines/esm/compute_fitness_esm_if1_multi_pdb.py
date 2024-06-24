import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import pandas as pd
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import esm
import esm.inverse_folding
from esm.inverse_folding.util import CoordBatchConverter
from utils.data_utils import DMS_file_cleanup, DMS_file_for_LLM
from protein_mpnn_utils import parse_PDB
    
SCORE_NATIVE = False

def _concatenate_coords(coords, target_chain_id, padding_length=10):
    """
    Args:
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: Length of padding between concatenated chains
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates, a
              concatenation of the chains with padding in between
            - seq is the extracted sequence, with padding tokens inserted
              between the concatenated chains
    """
    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)
    # For best performance, put the target chain first in concatenation.
    # for multi mutated chains, don't change chain order.
    coords_list = []
    for chain_id in target_chain_id:
        if len(coords_list) > 0:
            coords_list.append(pad_coords)
        coords_list.append(coords[chain_id])
    for chain_id in coords:
        if chain_id not in target_chain_id:
            if len(coords_list) > 0:
                coords_list.append(pad_coords)
            coords_list.append(coords[chain_id])
    coords_concatenated = np.concatenate(coords_list, axis=0)
    return coords_concatenated

def get_sequence_loss_batch(model, alphabet, coords_list, seq_list):
    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    assert len(coords_list) == len(seq_list)
    batch = [(coords, None, seq) for coords, seq in zip(coords_list, seq_list)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)
    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
    loss = F.cross_entropy(logits, target, reduction='none')
    losses = loss.cpu().detach().numpy()
    target_padding_masks = target_padding_mask.cpu().numpy()
    return losses, target_padding_masks

def score_sequence_batch(model, alphabet, coords_list, seq_list):
    losses, target_padding_masks = get_sequence_loss_batch(model, alphabet, coords_list, seq_list)
    print("debug: losses and target_padding_mask shapes: ", losses.shape, target_padding_masks.shape)
    ll_fullseqs_batch = -np.sum(losses * ~target_padding_masks, axis=1) / np.sum(~target_padding_masks, axis=1)
    return ll_fullseqs_batch

def score_sequence_in_complex_batch(model, alphabet, coords, native_seqs, target_chain_id,
        seq_list, padding_length=10):
    """
    Scores sequence for one chain in a complex.
    Args:
        model: An instance of the GVPTransformer model
        alphabet: Alphabet for the model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        target_seq: Target sequence for the target chain for scoring.
        padding_length: padding length in between chains
    Returns:
        Tuple (ll_fullseq, ll_withcoord)
        - ll_fullseq: Average log-likelihood over the full target chain
        - ll_withcoord: Average log-likelihood in target chain excluding those
            residues without coordinates
    """
    all_coords = _concatenate_coords(coords, target_chain_id)
    # print(all_coords.shape)
    # print(len(seq_list[0]))
    coords_list = [all_coords] * len(seq_list)
    loss, target_padding_mask = get_sequence_loss_batch(model, alphabet, coords_list,
            seq_list)
    ll_fullseq = -np.sum(loss * ~target_padding_mask,axis=1) / np.sum(~target_padding_mask,axis=1)
    ll_withcoord = None
    # # Also calculate average when excluding masked portions
    # coord_mask = np.all(np.isfinite(coords[target_chain_id][:len(seq_list[0])]), axis=(-1, -2))
    # ll_withcoord = -np.sum(loss[:,:coord_mask.shape[0]] * coord_mask) / np.sum(coord_mask)
    return ll_fullseq, ll_withcoord

def score_singlechain_backbone_batch(model, alphabet, pdb_file, chain, mut_df, output_filepath, batch_size=1,nogpu=False):

    start_time = time.perf_counter()
    coords, native_seq = esm.inverse_folding.util.load_coords(pdb_file, chain)
    print(f"Coords loaded in {time.perf_counter() - start_time} seconds")
    seq_list = mut_df["mutated_sequence"].tolist()
    header_list = mut_df["mutant"].tolist()
    coords_list = [coords] * len(seq_list)

    print(f"Sequences loaded in {time.perf_counter() - start_time} seconds")

    start_scoring = time.perf_counter()
    
    all_score = []
    for i in tqdm(range(0, len(seq_list), batch_size)):
        batch = seq_list[i:i+batch_size]
        coords_batch = coords_list[i:i+batch_size]
        ll_fullseq = score_sequence_batch(model, alphabet, coords_batch, batch)
        all_score.append(ll_fullseq)
    all_score = np.concatenate(all_score)
    mut_df['esm_if1'] = all_score

    print(f"Scoring in {time.perf_counter() - start_scoring} seconds")
        
    print(f'Results saved to {output_filepath}')
    print(f"Total time: {time.perf_counter() - start_time}")

def score_multichain_backbone_batch(model, alphabet, pdb_file, chain_id, mut_df, output_filepath, batch_size=1,nogpu=False):
    
    start_time = time.perf_counter()
    pdb_dic = parse_PDB(pdb_file)[0]
    coords = {}
    native_seqs = {}
    # N, CA, C
    for k in pdb_dic:
        if 'coords_chain' in k:
            cid = k[-1]
            NCAC = np.stack([pdb_dic[k][a] for a in [f'N_chain_{cid}',f'CA_chain_{cid}',f'C_chain_{cid}']],1)
            coords[k[-1]] = np.float32(NCAC)
        elif 'seq_chain' in k:
            native_seqs[k[-1]] = pdb_dic[k].replace('-','X')
    # for c in coords:
    #     print(coords[c].dtype)
    #     print(len(native_seqs[c]))
    # structure = esm.inverse_folding.util.load_structure(pdb_file)
    # coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
    # for c in coords:
    #     print(coords[c].dtype)
    #     print(len(native_seqs[c]))
    print(f"Coords loaded in {time.perf_counter() - start_time} seconds")
    seq_list = []
    for ms in mut_df['mutated_sequence'].tolist():
        seq = ''
        ms = eval(ms)
        for c in chain_id:
            if len(seq) > 0:
                seq += 'X'*10
            if c in ms:
                seq += ms[c]
            else:
                seq += native_seqs[c]
        for c in coords:
            if c not in chain_id:
                if len(seq) > 0:
                    seq += 'X'*10
                if c in ms:
                    seq += ms[c]
                else:
                    seq += native_seqs[c]
            
        seq_list.append(seq)
    header_list = mut_df["mutant"].tolist()

    print(f"Sequences loaded in {time.perf_counter() - start_time} seconds")

    start_scoring = time.perf_counter()
    
  
    all_score = []
    for i in tqdm(range(0, len(seq_list), batch_size)):
        batch = seq_list[i:i+batch_size]
        try:
            ll_fullseq,ll_withcoord = score_sequence_in_complex_batch(model, alphabet, coords, native_seqs, chain_id, batch)
            all_score.append(ll_fullseq)
        except RuntimeError as e:
            print(e)
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
            all_score.append(0)
    all_score = np.concatenate(all_score)
    mut_df['esm_if1'] = all_score
    print(f"Scoring in {time.perf_counter() - start_scoring} seconds")
        
def main():
    parser = argparse.ArgumentParser(description='Score sequences based on a given structure.')
    parser.add_argument('--dms_mapping',type=str,help='path to DMS reference file')
    parser.add_argument('--dms_input',type=str,help="path to folder containing DMS data")
    parser.add_argument('--structure_folder',type=str,help='folder containing pdb files for each DMS')
    parser.add_argument('--dms_index',type=int,help='index of DMS in DMS reference file')
    parser.add_argument('--model_location',type=str,help='path to model')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--dms_output',type=str,help='path to folder where scores will be saved')
    parser.add_argument('--chain', type=str,help='chain id for the chain of interest', default='A')
    parser.set_defaults(multichain_backbone=False)
    parser.add_argument('--multichain-backbone', action='store_true',help='use the backbones of all chains in the input for conditioning')
    parser.add_argument('--singlechain-backbone', dest='multichain_backbone',action='store_false',help='use the backbone of only target chain in the input for conditioning')
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    
    args = parser.parse_args()

    if not os.path.exists(args.dms_output):
        os.makedirs(args.dms_output)

    if '.csv' not in args.dms_input:
        mapping_df = pd.read_csv(args.dms_mapping)
        DMS_id = mapping_df.iloc[args.dms_index]['DMS_id']
        DMS_filename = mapping_df.iloc[args.dms_index]['DMS_filename']
        output_filename = args.dms_output + os.sep + DMS_id + ".csv"
        pdb_file = args.structure_folder + os.sep + mapping_df.iloc[args.dms_index]['pdb_file']
        print(pdb_file)
        chain_id =  mapping_df.iloc[args.dms_index]['chain_id']
        df = pd.read_csv(args.dms_input + os.sep + DMS_filename)
    else:
        df = pd.read_csv(args.dms_input)
        output_filename=str(args.dms_output)+os.sep+os.path.basename(args.dms_input)
        
    df['chain_id'] = df['chain_id'].fillna('')
    df['mutated_sequence'] = df['mutated_sequence'].fillna('{}')
    df['mutant'] = df['mutant'].apply(eval)
    mut_chain_id = ''
    for i in df.index:
        for c in df.loc[i,'mutant']:
            if df.loc[i,'mutant'][c] != '':
                if c not in mut_chain_id:
                    mut_chain_id += c
    print(mut_chain_id)
    # df = DMS_file_for_LLM(df)
    # df = df[:10]
    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model_location)
    if not args.nogpu:
        assert torch.cuda.is_available(), "Expected GPU. If you want to use CPU, you have to specify --nogpu every time."
        model = model.cuda()
        print("Transferred model to GPU")
    else:
        print(f"Running model on CPU: torch cuda is available={torch.cuda.is_available()} nogpu={nogpu}")

    model = model.eval()
    with torch.no_grad():
        all_g = []
        for POI,g in df.groupby(['POI']):
            print(g['POI'].values[0])
            # if wt_seq == '':
            #     g['esm_if1'] = 0
            #     all_g.append(g)
            #     continue
            if 'pdb_file' in g.columns:
                pdb_file = args.structure_folder + os.sep + g['pdb_file'].values[0]
            if 'chain_id' in g.columns:
                chain_id = g['chain_id'].values[0]
            score_multichain_backbone_batch(model, alphabet, pdb_file=pdb_file, chain_id=mut_chain_id, batch_size=args.batch_size,mut_df=g,output_filepath=output_filename,nogpu=args.nogpu)
            all_g.append(g)
    df = pd.concat(all_g).sort_index()
    df.to_csv(output_filename,index=False)

if __name__ == '__main__':
    main()
