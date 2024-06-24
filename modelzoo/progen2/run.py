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
checkpoint = f"{checkpoint_folder}/progen2-base/"

msa_path = os.getenv('msa_path')
msa_db_path = os.getenv('msa_db_path')
a2m_root = os.getenv('a2m_root')


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
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} {python} {dir_path}/../../baselines/progen2/compute_fitness_multi_pdb.py' \
            + f' --checkpoint {checkpoint}' \
            + f' --dms_index {idx}' \
            + f' --dms_mapping {dms_mapping}' \
            + f' --dms_input {dms_input}' \
            + f' --dms_output {dms_output}' 

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
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} {python} {dir_path}/../../baselines/progen2/compute_fitness_multi_pdb.py' \
            + f' --checkpoint {checkpoint}' \
            + f' --dms_input {dms_input}' \
            + f' --dms_output {dms_output}' 
    run(cmd)



'''
#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export Progen2_model_name_or_path="path to  progen2 small model"
export output_scores_folder="${DMS_output_score_folder_subs}/Progen2/small"

# export Progen2_model_name_or_path="path to  progen2 medium model"
# export output_scores_folder="${DMS_output_score_folder_subs}/Progen2/medium"
 
# export Progen2_model_name_or_path="path to progen2 base model"
# export output_scores_folder="${DMS_output_score_folder_subs}/Progen2/base"

# export Progen2_model_name_or_path="path to progen2 large model"
# export output_scores_folder="${DMS_output_score_folder_subs}/Progen2/large"
 
# export Progen2_model_name_or_path="path to progen2 xlarge model"
# export output_scores_folder="${DMS_output_score_folder_subs}/Progen2/xlarge"

export DMS_index="Experiment index to run (e.g. 1,2,...217)"

python ../../baselines/progen2/compute_fitness.py \
            --Progen2_model_name_or_path ${Progen2_model_name_or_path} \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --DMS_index $DMS_index \
            --output_scores_folder ${output_scores_folder} \
'''