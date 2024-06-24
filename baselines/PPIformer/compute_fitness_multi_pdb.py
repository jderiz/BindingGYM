import argparse
import pandas as pd 
import numpy as np
import os
import torch
from ppiformer.tasks.node import DDGPPIformer
from ppiformer.utils.api import download_weights, predict_ddg
from ppiformer.definitions import PPIFORMER_WEIGHTS_DIR, PPIFORMER_TEST_DATA_DIR



def main(args):
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
    download_weights()
    device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
    models = [DDGPPIformer.load_from_checkpoint(PPIFORMER_WEIGHTS_DIR / f'ddg_regression/{i}.ckpt', map_location=torch.device(device)).eval() for i in range(3)]
    def get_mut_list(x):
        x = eval(x)
        all_ms = []
        for c in x:
            if x[c] != '':
                for m in x[c].split(':'):
                    all_ms.append(m[:1]+c+m[1:])
        return ','.join(all_ms)

    all_g = []
    for (POI,chain_id),g in df.groupby(['POI','chain_id']):
        g['#Pdb'] = g['POI']
        g['PDB Id'] = g['POI']
        g['Partners'] = chain_id
        g['Partners'] = g['Partners'].apply(set)
        g['Mutation(s)'] = g['mutant_pdb'].apply(get_mut_list)
        ppi_path = args.structure_folder + os.sep + g['pdb_file'].values[0]
        wt_g = g.loc[g['Mutation(s)'].apply(len)==0]
        wt_g['ddG'] = 0
        mt_g = g.loc[g['Mutation(s)'].apply(len)>0]
        mt_g['ddG'] = np.nan
        ddg = predict_ddg(models, ppi_path, mt_g)
        mt_g['ddG'] = -ddg.cpu().numpy()
        all_g.append(pd.concat([wt_g,mt_g]))
    df = pd.concat(all_g).sort_index()
    df.to_csv(output_filename, index=False)

        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('--dms_mapping',type=str,help='path to DMS reference file')
    argparser.add_argument('--dms_input',type=str,help="path to folder containing DMS data")
    argparser.add_argument('--structure_folder',type=str,help='folder containing pdb files for each DMS')
    argparser.add_argument('--dms_index',type=int,help='index of DMS in DMS reference file')
    argparser.add_argument("--dms_output", type=str, help="Path to a folder to output sequences, e.g. /home/out/")

    args = argparser.parse_args()    
    main(args)   
