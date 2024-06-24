import datetime
import os,sys
import json
import argparse
from resource import getrusage, RUSAGE_SELF

import pandas as pd
import torch

from EVE import VAE_model
from utils import data_utils

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Evol indices')
    parser.add_argument('--msa_path', type=str, help='Folder where MSAs are stored')
    parser.add_argument('--a2m_root', default=None, type=str, help='Path to a2m script')
    parser.add_argument('--msa_db_path', default=None, type=str, help='Path to msa database')
    parser.add_argument('--dms_mapping', type=str, help='List of proteins and corresponding MSA file name')
    parser.add_argument('--dms_index', type=int, help='Row index of protein in input mapping file')
    parser.add_argument('--MSA_weights_location', type=str, help='Location where weights for each sequence in the MSA will be stored')
    parser.add_argument('--theta_reweighting', type=float, help='Parameters for MSA sequence re-weighting')
    parser.add_argument('--seed',type=int,nargs="+", help='Random seed for VAE model initialization')
    parser.add_argument('--VAE_checkpoint_location', type=str, help='Location where VAE model checkpoints will be stored')
    parser.add_argument('--model_parameters_location', type=str, help='Location of VAE model parameters')
    parser.add_argument('--dms_input', type=str, help='Location of all mutations to compute the evol indices for')
    parser.add_argument('--dms_output', type=str, help='Output location of computed evol indices')
    parser.add_argument('--num_samples_compute_evol_indices', type=int, help='Num of samples to approximate delta elbo when computing evol indices')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size when computing evol indices')
    parser.add_argument("--skip_existing", action="store_true", help="Skip scoring if output file already exists")
    parser.add_argument("--aggregation_method", choices=["full", "batch", "online"], default="full", help="Method to aggregate evol indices")
    parser.add_argument("--threshold_focus_cols_frac_gaps", type=float,
                        help="Maximum fraction of gaps allowed in focus columns - see data_utils.MSA_processing")
    args = parser.parse_args()

    print("Arguments:", args)
    
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
        evol_indices_output_filename = str(args.dms_output)+os.sep+DMS_id+'.csv'
        df = pd.read_csv(args.dms_input + os.sep + DMS_file_name, low_memory=False)
    else:
        evol_indices_output_filename = str(args.dms_output)+os.sep+os.path.basename(args.dms_input)
        df = pd.read_csv(args.dms_input, low_memory=False)
    
    all_g = []
    for POI,g in df.groupby('POI'):
        g = data_utils.DMS_file_for_LLM(g,focus=True)
        chain_id = g['chain_id'].values[0]
        name = POI
        protein_name = POI
        msa_location = args.msa_path + os.sep + f'{protein_name}.a2m'
        print(msa_location)
       
        weights_file = args.msa_path + os.sep + name + '_msa_weight.npy' if args.msa_path else None
        
        DMS_mutant_column = "mutant"
        if args.theta_reweighting is not None:
            theta = args.theta_reweighting
        else:
            try:
                theta = float(mapping_file['MSA_theta'][args.dms_index])
            except:
                theta = 0.2
        print("Theta MSA re-weighting: "+str(theta))
        # Using data_kwargs so that if options aren't set, they'll be set to default values
        data_kwargs = {}
        if args.threshold_focus_cols_frac_gaps is not None:
            print("Using custom threshold_focus_cols_frac_gaps: ", args.threshold_focus_cols_frac_gaps)
            data_kwargs['threshold_focus_cols_frac_gaps'] = args.threshold_focus_cols_frac_gaps

        data = data_utils.MSA_processing(
                MSA_location=msa_location,
                theta=theta,
                use_weights=True,  # Don't need weights for evol indices
                weights_location=weights_file,
                **data_kwargs,
        )
        print(data.seq_len)

        for seed in args.seed:
            model_name = protein_name + f"_seed_{seed}"
            print("Model name: "+str(model_name))

            model_params = json.load(open(args.model_parameters_location))

            model = VAE_model.VAE_model(
                            model_name=model_name,
                            data=data,
                            encoder_parameters=model_params["encoder_parameters"],
                            decoder_parameters=model_params["decoder_parameters"],
                            random_seed=42
            )
            
            model = model.to(model.device)
            checkpoint_name = str(args.VAE_checkpoint_location) + os.sep + model_name
            # assert os.path.isfile(checkpoint_name), 'Checkpoint file does not exist: {}'.format(checkpoint_name)

            try:
                checkpoint = torch.load(checkpoint_name, map_location=model.device)  # Added map_location so that this works with CPU too
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Initialized VAE with checkpoint '{}' ".format(checkpoint_name))
            except Exception as e:
                print("Unable to load VAE model checkpoint {}".format(checkpoint_name))
                print(e)
                continue

            list_valid_mutations, evol_indices, _, _ = model.compute_evol_indices(
                msa_data=data,
                full_data=g,
                mutant_column=DMS_mutant_column,
                num_samples=args.num_samples_compute_evol_indices,
                batch_size=args.batch_size,
                aggregation_method=args.aggregation_method
            )
            tmp = {}
            tmp['mutant'] = list_valid_mutations
            tmp[f'evol_indices_seed_{seed}'] = evol_indices
            tmp = pd.DataFrame(tmp)
            g = g.merge(tmp,how='left',on='mutant')
            g[f'evol_indices_seed_{seed}'] = g[f'evol_indices_seed_{seed}'].fillna(0)

        all_g.append(g)
    if len(all_g) > 0:
        all_g = pd.concat(all_g).sort_index()
        all_g.to_csv(evol_indices_output_filename,index=False)
