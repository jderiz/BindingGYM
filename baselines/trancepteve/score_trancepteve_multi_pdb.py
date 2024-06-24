import os
import argparse
import json
import pandas as pd

import torch

from transformers import PreTrainedTokenizerFast
import trancepteve
from trancepteve import config, model_pytorch
from trancepteve.utils import dms_utils,data_utils

dir_path = os.path.dirname(os.path.abspath(__file__))

def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with TranceptEVE.
    """
    parser = argparse.ArgumentParser(description='TranceptEVE scoring')
    parser.add_argument('--checkpoint', type=str, help='Path of Tranception model checkpoint')
    parser.add_argument('--model_framework', default='pytorch', type=str, help='Underlying framework [pytorch|JAX]')
    parser.add_argument('--batch_size_inference', default=20, type=int, help='Batch size for inference')

    #We may pass in all required information about the DMS via the provided reference files, or specify all relevant fields manually
    parser.add_argument('--DMS_reference_file_path', default=None, type=str, help='Path to reference file with list of DMS to score')
    parser.add_argument('--dms_index', default=0, type=int, help='Index of DMS assay in reference file')
    parser.add_argument('--dms_mapping',type=str,help='path to DMS reference file')
    parser.add_argument('--dms_input',type=str,help="path to folder containing DMS data")
    parser.add_argument('--structure_folder',type=str,help='folder containing pdb files for each DMS')
    
    parser.add_argument('--msa_path', default=None, type=str, help='Path to cache msa')
    parser.add_argument('--a2m_root', default=None, type=str, help='Path to a2m script')
    parser.add_argument('--msa_db_path', default=None, type=str, help='Path to msa database')
    
    #Fields to be passed manually if reference file is not used
    parser.add_argument('--target_seq', default=None, type=str, help='Full wild type sequence that is mutated in the DMS asssay')
    parser.add_argument('--DMS_file_name', default=None, type=str, help='Name of DMS assay file')
    parser.add_argument('--MSA_filename', default=None, type=str, help='Name of MSA (eg., a2m) file constructed on the wild type sequence')
    parser.add_argument('--MSA_weight_file_name', default=None, type=str, help='Weight of sequences in the MSA (optional)')
    parser.add_argument('--MSA_start', default=None, type=int, help='Sequence position that the MSA starts at (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Sequence position that the MSA ends at (1-indexing)')
    parser.add_argument('--UniprotID', default=None, type=str, help='Uniprot ID of protein (EVE retrieval only)')
    parser.add_argument('--MSA_threshold_sequence_frac_gaps', default=None, type=float, help='MSA processing: pct fragments threshold')
    parser.add_argument('--MSA_threshold_focus_cols_frac_gaps', default=None, type=float, help='MSA processing: pct col filled threshold')

    parser.add_argument('--DMS_data_folder', type=str, help='Path to folder that contains all DMS assay datasets')
    parser.add_argument('--dms_output', default='./', type=str, help='Name of folder to write model scores to')
    parser.add_argument('--deactivate_scoring_mirror', action='store_true', help='Whether to deactivate sequence scoring from both directions (Left->Right and Right->Left)')
    parser.add_argument('--indel_mode', action='store_true', help='Flag to be used when scoring insertions and deletions. Otherwise assumes substitutions')
    parser.add_argument('--scoring_window', default="optimal", type=str, help='Sequence window selection mode (when sequence length longer than model context size)')
    
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for model scoring data loader')
    parser.add_argument('--inference_time_retrieval_type', default=None, type=str, help='Type of inference time retrieval [None,Tranception,TranceptEVE]')
    parser.add_argument('--retrieval_weights_manual', action='store_true', help='Whether to manually select the MSA/EVE aggregation weights')
    parser.add_argument('--retrieval_inference_MSA_weight', default=0.5, type=float, help='Coefficient (alpha) used when aggregating autoregressive transformer and MSA retrieval')
    parser.add_argument('--retrieval_inference_EVE_weight', default=0.5, type=float, help='Coefficient (beta) used when aggregating autoregressive transformer and EVE retrieval')
    
    parser.add_argument('--MSA_folder', default='.', type=str, help='Path to MSA for neighborhood scoring')
    parser.add_argument('--MSA_weights_folder', default=None, type=str, help='Path to MSA weights for neighborhood scoring')
    parser.add_argument('--clustal_omega_location', default=None, type=str, help='Path to Clustal Omega (only needed with scoring indels with retrieval)')

    parser.add_argument('--EVE_model_folder', type=str, help='Path to folder containing the EVE model(s)')
    parser.add_argument('--EVE_seeds', nargs='*', help='Seeds of the EVE model(s) to be leveraged')
    parser.add_argument('--EVE_num_samples_log_proba', default=10, type=int, help='Number of samples to compute the EVE log proba')
    parser.add_argument('--EVE_model_parameters_location', default=None, type=str, help='Path to EVE model parameters')
    parser.add_argument('--MSA_recalibrate_probas', action='store_true', help='Whether to normalize EVE & MSA log probas (matching temp. of Transformer)')
    parser.add_argument('--EVE_recalibrate_probas', action='store_true', help='Whether to normalize EVE & MSA log probas (matching temp. of Transformer)')
    parser.add_argument('--clinvar_scoring', action='store_true', help='Tweaks when scoring ClinVar input file')
    args = parser.parse_args()
    print(args)
    model_name = args.checkpoint.split("/")[-1]
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=dir_path+os.sep+"trancepteve/utils/tokenizers/Basic_tokenizer",
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )

    if '.csv' not in args.dms_input:
        mapping_protein_seq_DMS = pd.read_csv(args.dms_mapping)
        DMS_id = mapping_protein_seq_DMS["DMS_id"][args.dms_index]
        row = mapping_protein_seq_DMS.loc[args.dms_index]

        mapping_protein_seq_DMS = pd.read_csv(args.dms_mapping)
        list_DMS = mapping_protein_seq_DMS["DMS_id"]
        DMS_id=list_DMS[args.dms_index]
        print("Compute scores for DMS: "+str(DMS_id))
        print(str(args.dms_input)+os.sep+row["DMS_filename"])
        df = pd.read_csv(str(args.dms_input)+os.sep+row["DMS_filename"], low_memory=False)
   
    else:
        df = pd.read_csv(args.dms_input, low_memory=False)
    
    config = json.load(open(args.checkpoint+os.sep+'config.json'))
    config = trancepteve.config.TranceptEVEConfig(**config)
    config.attention_mode="tranception"
    config.position_embedding="grouped_alibi"
    config.tokenizer = tokenizer
    config.scoring_window = args.scoring_window
    MSA_threshold_sequence_frac_gaps = float(mapping_protein_seq_DMS["MSA_threshold_sequence_frac_gaps"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]) if "MSA_threshold_sequence_frac_gaps" in mapping_protein_seq_DMS else 0.5
    MSA_threshold_focus_cols_frac_gaps = float(mapping_protein_seq_DMS["MSA_threshold_focus_cols_frac_gaps"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]) if "MSA_threshold_focus_cols_frac_gaps" in mapping_protein_seq_DMS else 1.0
    
    if args.model_framework=="pytorch":
        model = trancepteve.model_pytorch.TrancepteveLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.checkpoint,config=config)
        if torch.cuda.is_available():
            model.cuda()
    model.eval()

    all_g = []
    for POI,g in df.groupby('POI'):
        g = data_utils.DMS_file_for_LLM(g,focus=True)
        name = POI
        wt_seq = g['wildtype_sequence'].values[0]
        print(wt_seq)
        print(len(wt_seq))
        config.full_target_seq = wt_seq

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
        MSA_weight_file_name = args.msa_path + os.sep + name + '_msa_weight.npy' if args.msa_path else None
        MSA_start = 0
        MSA_end = len(wt_seq)

        if args.inference_time_retrieval_type is not None:
            config.inference_time_retrieval_type = args.inference_time_retrieval_type
            config.retrieval_aggregation_mode = "aggregate_indel" if args.indel_mode else "aggregate_substitution"
            config.MSA_filename = MSA_data_file
            print('-----',config.MSA_filename)
            config.MSA_weight_file_name = MSA_weight_file_name
            config.MSA_start = MSA_start
            config.MSA_end = MSA_end
            config.MSA_threshold_sequence_frac_gaps = MSA_threshold_sequence_frac_gaps
            config.MSA_threshold_focus_cols_frac_gaps = MSA_threshold_focus_cols_frac_gaps
            config.retrieval_weights_manual = args.retrieval_weights_manual
            config.retrieval_inference_MSA_weight = args.retrieval_inference_MSA_weight
            config.retrieval_inference_EVE_weight = args.retrieval_inference_EVE_weight

            if "TranceptEVE" in args.inference_time_retrieval_type:
                EVE_model_paths = []
                EVE_seeds = args.EVE_seeds 
                num_seeds = len(EVE_seeds)
                print("Number of distinct EVE models to be leveraged: {}".format(num_seeds))
                for seed in EVE_seeds:
                    print(f"{args.EVE_model_folder}/{os.path.basename(MSA_data_file.split('.a2m')[0])}_seed_{seed}")
                    if os.path.exists(f"{args.EVE_model_folder}/{os.path.basename(MSA_data_file.split('.a2m')[0])}_seed_{seed}"):
                        EVE_model_name = f"{os.path.basename(MSA_data_file.split('.a2m')[0])}_seed_{seed}"
                    elif os.path.exists(f"{args.EVE_model_folder}/{UniProt_ID}_seed_{seed}"):
                        EVE_model_name = f"{UniProt_ID}_seed_{seed}"
                    else:
                        print(f"No EVE Model available for {MSA_data_file} with random seed {seed} in {args.EVE_model_folder}. Exiting")
                        return 
                        
                    EVE_model_paths.append(args.EVE_model_folder + os.sep + EVE_model_name)
                config.EVE_model_paths = EVE_model_paths
                config.EVE_num_samples_log_proba = args.EVE_num_samples_log_proba
                config.EVE_model_parameters_location = args.EVE_model_parameters_location
                config.MSA_recalibrate_probas = args.MSA_recalibrate_probas
                config.EVE_recalibrate_probas = args.EVE_recalibrate_probas
            else:
                num_seeds=0
        else:
            config.inference_time_retrieval_type = None
            config.retrieval_aggregation_mode = None
            
        model.reconfig(config)
        mutation_type = '_indels' if args.indel_mode else '_substitutions'
        mirror_type = '_no_mirror' if args.deactivate_scoring_mirror else ''
        normalization = '_norm-EVE' if args.EVE_recalibrate_probas else ''
        normalization = normalization + '_norm-MSA' if args.MSA_recalibrate_probas else normalization
        retrieval_weights = '_MSA-' + str(args.retrieval_inference_MSA_weight) +'_EVE-'+ str(args.retrieval_inference_EVE_weight) if args.retrieval_weights_manual else ''
        retrieval_type = ('_retrieval_' + args.inference_time_retrieval_type + retrieval_weights + '_' + str(num_seeds) + '-EVE-models' + normalization) if args.inference_time_retrieval_type is not None else '_no_retrieval'
        scoring_filename = args.dms_output
        if not os.path.isdir(scoring_filename):
            os.mkdir(scoring_filename)
        scoring_filename += os.sep + DMS_id + '.csv'
        
        
        with torch.no_grad():
            all_scores = model.score_mutants(
                                        DMS_data=g, 
                                        target_seq=wt_seq, 
                                        scoring_mirror=not args.deactivate_scoring_mirror, 
                                        batch_size_inference=args.batch_size_inference,  
                                        num_workers=args.num_workers, 
                                        indel_mode=args.indel_mode
                                        )
        g = g.merge(all_scores,how="left",on="mutated_sequence")
        all_g.append(g)
    all_scores = pd.concat(all_g).sort_index()
    all_scores.to_csv(scoring_filename, index=False)

if __name__ == '__main__':
    main()