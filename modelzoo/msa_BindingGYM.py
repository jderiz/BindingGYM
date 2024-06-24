import pandas as pd
import os,sys
import subprocess,multiprocessing

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
a2m_root = './combining-evolutionary-and-assay-labelled-data'
eve_model_path = os.getenv('eve_model_path')

def run_a2m(cmd):
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

def generate_msa(df,msa_path,a2m_root,msa_db_path):
    params = []
    all_name = []
    for idx in df.index:
        filename = df.loc[idx,'DMS_filename']
        g = pd.read_csv(f'../input/Binding_substitutions_DMS/{filename}')
        focus_chains = []
        for i in g.index:
            mutants = eval(g.loc[i,'mutant'])
            for c in mutants:
                if c not in focus_chains:
                    if mutants[c] != '':
                        focus_chains.append(c)
        focus_chains = sorted(focus_chains)
        name = g['POI'].values[0]
        if name in all_name:continue
        all_name.append(name)
        wt_seqs = eval(g['wildtype_sequence'].values[0])
        seq = ''
        for c in focus_chains:
            seq += wt_seqs[c]
        if os.path.exists(f'{msa_path}/{name}.a2m'):
            print(name)
            continue
        with open(f'{msa_path}/{name}.fasta','w') as f:
            f.write(f'>{name}\n')
            f.write(f'{seq}\n')
        cmd = f'''bash {a2m_root}/scripts/jackhmmer.sh {msa_path} {name} 0.5 5 {msa_db_path} {a2m_root}'''
        params.append(cmd)
    pool = multiprocessing.Pool( processes = min(len(params), 8) )
    for args in params:
        pool.apply_async(run_a2m, args = (args,))
    pool.close()
    pool.join()


df = pd.read_csv('../input/BindingGYM.csv')
generate_msa(df,'../input/msas',a2m_root,msa_db_path)