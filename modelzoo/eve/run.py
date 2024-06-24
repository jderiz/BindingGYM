from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--root', type=str, default='../')
parser.add_argument('--dms_mapping', type=str, default='')
parser.add_argument('--dms_input', type=str, default='')
parser.add_argument('--dms_output', type=str, default='')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--structure_folder', type=str, default='../input/structures/')
parser.add_argument('--gpus', type=str, default='0')

args, unknown = parser.parse_known_args()

import pandas as pd
import os,subprocess,multiprocessing
from tqdm import tqdm


gpus = args.gpus.split(',')
gpu_count = len(gpus)

python = os.getenv('python')
dir_path = os.path.dirname(os.path.abspath(__file__))
dms_mapping = os.getenv('dms_mapping')
dms_input = os.getenv('dms_input')
dms_output = os.getenv('dms_output')
dms_output = f'{dir_path}/{dms_output}'
structure_folder = os.getenv('structure_folder')
checkpoint_folder = os.getenv('checkpoint_folder')
checkpoint = f'{checkpoint_folder}/Tranception_Large'
msa_path = os.getenv('msa_path')
msa_db_path = os.getenv('msa_db_path')
a2m_root = os.getenv('a2m_root')
eve_model_path = os.getenv('eve_model_path')

model_parameters_location=f'{dir_path}/../../baselines/EVE/EVE/default_model_params.json'


if not os.path.exists(dms_output):
    os.makedirs(dms_output)

def run(cmd):
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

if '.csv' not in dms_input:
    df = pd.read_csv(dms_mapping)
    params = []
    for idx in df.index:
        DMS_id = df.loc[idx,'DMS_id']
        if os.path.exists(f'./output/{DMS_id}.csv'):
            print(DMS_id)
            continue
        i = idx % gpu_count
        gpu_id = gpus[i]
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} {python} {dir_path}/../../baselines/EVE/compute_evol_indices_DMS_multi_pdb.py' \
            + f' --dms_index {idx}' \
            + f' --dms_mapping {dms_mapping}' \
            + f' --dms_input {dms_input}' \
            + f' --dms_output {dms_output}' \
            + f' --num_samples_compute_evol_indices 20000' \
            + f' --VAE_checkpoint_location {eve_model_path}' \
            + f' --model_parameters_location {model_parameters_location}' \
            + f' --threshold_focus_cols_frac_gaps 1' \
            + f' --msa_path {msa_path}' \
            + f' --msa_db_path {msa_db_path}' \
            + f' --a2m_root {a2m_root}' \
            + f' --batch_size 1024' \
            + f' --seed 42' 

        params.append(cmd) 

    print(len(params))
    ncpus = len(gpus)
    pool = multiprocessing.Pool( processes = min(len(params), ncpus) )
    for cmd in tqdm(params):
        print(cmd)
        pool.apply_async(run, args = [cmd])
    pool.close()
    pool.join()

else:
    gpu_id = gpus[0]
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} {python} {dir_path}/../../baselines/EVE/compute_evol_indices_DMS_multi_pdb.py' \
            + f' --dms_input {dms_input}' \
            + f' --dms_output {dms_output}' \
            + f' --num_samples_compute_evol_indices 20000' \
            + f' --VAE_checkpoint_location {eve_model_path}' \
            + f' --model_parameters_location {model_parameters_location}' \
            + f' --threshold_focus_cols_frac_gaps 1' \
            + f' --msa_path {msa_path}' \
            + f' --msa_db_path {msa_db_path}' \
            + f' --a2m_root {a2m_root}' \
            + f' --batch_size 8' \
            + f' --seed 42' 
    run(cmd)
