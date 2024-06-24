import os,sys
import argparse
import json
import pandas as pd

import torch

from transformers import PreTrainedTokenizerFast
import tranception
from tranception import config, model_pytorch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_utils import DMS_file_for_LLM

dir_path = os.path.dirname(os.path.abspath(__file__))

def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with Tranception.
    """
    parser = argparse.ArgumentParser(description='Tranception scoring')
    parser.add_argument('--checkpoint', type=str, help='Path of Tranception model checkpoint')
    parser.add_argument('--model_framework', default='pytorch', type=str, help='Underlying framework [pytorch|JAX]')
    parser.add_argument('--batch_size_inference', default=20, type=int, help='Batch size for inference')

    #We may pass in all required information about the DMS via the provided reference files, or specify all relevant fields manually
    parser.add_argument('--dms_mapping',type=str,help='path to DMS reference file')
    parser.add_argument('--dms_input',type=str,help="path to folder containing DMS data")
    parser.add_argument('--structure_folder',type=str,help='folder containing pdb files for each DMS')
    parser.add_argument('--dms_index',type=int,help='index of DMS in DMS reference file')
    
    #Fields to be passed manually if reference file is not used
    parser.add_argument('--target_seq', default=None, type=str, help='Full wild type sequence that is mutated in the DMS asssay')
    parser.add_argument('--DMS_file_name', default=None, type=str, help='Name of DMS assay file')
    parser.add_argument('--MSA_filename', default=None, type=str, help='Name of MSA (eg., a2m) file constructed on the wild type sequence')
    parser.add_argument('--MSA_weight_file_name', default=None, type=str, help='Weight of sequences in the MSA (optional)')
    parser.add_argument('--MSA_start', default=None, type=int, help='Sequence position that the MSA starts at (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Sequence position that the MSA ends at (1-indexing)')

    parser.add_argument('--dms_output', default='./', type=str, help='Name of folder to write model scores to')
    parser.add_argument('--deactivate_scoring_mirror', action='store_true', help='Whether to deactivate sequence scoring from both directions (Left->Right and Right->Left)')
    parser.add_argument('--indel_mode', action='store_true', help='Flag to be used when scoring insertions and deletions. Otherwise assumes substitutions')
    parser.add_argument('--scoring_window', default="optimal", type=str, help='Sequence window selection mode (when sequence length longer than model context size)')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of workers for model scoring data loader')
    parser.add_argument('--inference_time_retrieval', action='store_true', help='Whether to perform inference-time retrieval')
    parser.add_argument('--retrieval_inference_weight', default=0.6, type=float, help='Coefficient (alpha) used when aggregating autoregressive transformer and retrieval')
    parser.add_argument('--MSA_folder', default='.', type=str, help='Path to MSA for neighborhood scoring')
    parser.add_argument('--MSA_weights_folder', default=None, type=str, help='Path to MSA weights for neighborhood scoring')
    parser.add_argument('--clustal_omega_location', default=None, type=str, help='Path to Clustal Omega (only needed with scoring indels with retrieval)')
    
    parser.add_argument('--msa_path', default=None, type=str, help='Path to cache msa')
    parser.add_argument('--a2m_root', default=None, type=str, help='Path to a2m script')
    parser.add_argument('--msa_db_path', default=None, type=str, help='Path to msa database')
    
    args = parser.parse_args()
    
    model_name = args.checkpoint.split("/")[-1]
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=dir_path+os.sep+"tranception/utils/tokenizers/Basic_tokenizer",
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )

    if '.csv' not in args.dms_input:
        mapping_protein_seq_DMS = pd.read_csv(args.dms_mapping)
        list_DMS = mapping_protein_seq_DMS["DMS_id"]
        DMS_id=list_DMS[args.dms_index]
        print("Compute scores for DMS: "+str(DMS_id))
        # target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
        DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
        # if args.inference_time_retrieval:
        #     MSA_data_file = args.MSA_folder + os.sep + mapping_protein_seq_DMS["MSA_filename"][args.dms_index] if args.MSA_folder is not None else None
        #     MSA_weight_file_name = args.MSA_weights_folder + os.sep + mapping_protein_seq_DMS["weight_file_name"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0] if args.MSA_weights_folder else None
        #     MSA_start = int(mapping_protein_seq_DMS["MSA_start"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]) - 1 # MSA_start typically based on 1-indexing
        #     MSA_end = int(mapping_protein_seq_DMS["MSA_end"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0])
        scoring_filename = str(args.dms_output)+os.sep+DMS_id+'.csv'
        df = pd.read_csv(args.dms_input + os.sep + DMS_file_name, low_memory=False)
    else:
        scoring_filename = str(args.dms_output)+os.sep+os.path.basename(args.dms_input)
        df = pd.read_csv(args.dms_input, low_memory=False)
    
    config = json.load(open(args.checkpoint+os.sep+'config.json'))
    config = tranception.config.TranceptionConfig(**config)
    config.attention_mode="tranception"
    config.position_embedding="grouped_alibi"
    config.tokenizer = tokenizer
    config.scoring_window = args.scoring_window

    if args.model_framework=="pytorch":
        model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.checkpoint,config=config)
        if torch.cuda.is_available():
            model.cuda()
    model.eval()
    
    if not os.path.isdir(args.dms_output):
        os.mkdir(args.dms_output)


    retrieval_type = '_retrieval_' + str(args.retrieval_inference_weight) if args.inference_time_retrieval else '_no_retrieval'
    mutation_type = '_indels' if args.indel_mode else '_substitutions'
    mirror_type = '_no_mirror' if args.deactivate_scoring_mirror else ''
    # scoring_filename = args.dms_output + os.sep + DMS_id + ".csv"
    
    # df = df.query("POI=='1JTG_A_B'").reset_index(drop=True)
    all_g = []
    for POI,g in df.groupby('POI'):
        g = DMS_file_for_LLM(g,focus=True)

        name = POI
        wt_seq = g['wildtype_sequence'].values[0]
        print(len(wt_seq))
        print(name)
        # if name != '1JTG_B':continue
        if not os.path.exists(f'{args.msa_path}/{name}.a2m'):
            continue
            with open(f'{args.msa_path}/{name}.fasta','w') as f:
                f.write(f'>{name}\n')
                f.write(f'{wt_seq}\n')
            cmd = f'''bash {args.a2m_root}/scripts/jackhmmer.sh {args.msa_path} {name} 0.5 5 {args.msa_db_path} {args.a2m_root}'''
            os.system(cmd)
        MSA_data_file = f'{args.msa_path}/{name}.a2m'
        MSA_weight_file_name = args.MSA_weights_folder + os.sep + mapping_protein_seq_DMS["weight_file_name"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0] if args.MSA_weights_folder else None
        MSA_start = 0
        MSA_end = len(wt_seq)

        # print(args.inference_time_retrieval)
        if args.inference_time_retrieval:
            # print('here...')
            config.retrieval_aggregation_mode = "aggregate_indel" if args.indel_mode else "aggregate_substitution"
            config.MSA_filename=MSA_data_file
            config.full_protein_length=len(wt_seq)
            MSA_data_file=f'{args.msa_path}/{name}.a2m'
            config.MSA_weight_file_name=MSA_weight_file_name
            config.retrieval_inference_weight=args.retrieval_inference_weight
            config.MSA_start = MSA_start
            config.MSA_end = MSA_end
            if args.indel_mode:
                config.clustal_omega_location = args.clustal_omega_location
        else:
            config.retrieval_aggregation_mode = None
        # print(config.retrieval_aggregation_mode)
        model.reconfig(config)
        print("TMP Lood starting scoring")

        all_scores = model.score_mutants(
                                        DMS_data=g, 
                                        target_seq=wt_seq, 
                                        scoring_mirror=not args.deactivate_scoring_mirror, 
                                        batch_size_inference=args.batch_size_inference,  
                                        num_workers=args.num_workers, 
                                        indel_mode=args.indel_mode
                                        )
        g = g.merge(all_scores,how='left',on='mutated_sequence')
        all_g.append(g)
    all_scores = pd.concat(all_g).sort_index()
    all_scores.to_csv(scoring_filename, index=False)

if __name__ == '__main__':
    main()