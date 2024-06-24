from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--root', type=str, default='../')
parser.add_argument('--dms_mapping', type=str, default='')
parser.add_argument('--dms_input', type=str, default='')
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

msa_path = os.getenv('msa_path')
msa_db_path = os.getenv('msa_db_path')
a2m_root = os.getenv('a2m_root')
eve_model_path = os.getenv('eve_model_path')

model_parameters_location=f'{dir_path}/../../baselines/EVE/EVE/default_model_params.json'
training_logs_location=f'{dms_output}/logs'

if not os.path.exists(dms_output):
    os.makedirs(dms_output)
if not os.path.exists(training_logs_location):
    os.makedirs(training_logs_location)

def run(cmd):
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

df = pd.read_csv(dms_mapping)
params = []
for idx in df.index:
    DMS_id = df.loc[idx,'DMS_id']
    POI = df.loc[idx,'POI']
    chain_id = df.loc[idx,'chain_id']
    name = POI.split('_')[0]
    protein_name = f'{name}_{chain_id}'
    
    print("Protein name: " + str(protein_name))
    model_name = protein_name + f"_seed_42"
    print("Model name: " + str(model_name))
    model_checkpoint_final_path = eve_model_path + os.sep + model_name 
    
    if os.path.exists(model_checkpoint_final_path):
        print(DMS_id)
        continue
    i = idx % gpu_count
    gpu_id = gpus[i]
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} {python} {dir_path}/../../baselines/EVE/train_VAE_multi_pdb.py' \
        + f' --dms_mapping {dms_mapping}' \
        + f' --dms_index {idx}' \
        + f' --msa_path {msa_path}' \
        + f' --msa_db_path {msa_db_path}' \
        + f' --a2m_root {a2m_root}' \
        + f' --VAE_checkpoint_location {eve_model_path}' \
        + f' --model_parameters_location {model_parameters_location}' \
        + f' --training_logs_location {training_logs_location}' \
        + f' --threshold_focus_cols_frac_gaps 1' \
        + f' --seed 42' \
        + f' --skip_existing' \
        + f' --experimental_stream_data' \
        + f' --force_load_weights'
    
    params.append(cmd) 


print(len(params))
ncpus = len(gpus)
pool = multiprocessing.Pool( processes = min(len(params), ncpus) )
for cmd in tqdm(params):
    print(cmd)
    pool.apply_async(run, args = [cmd])
pool.close()
pool.join()


