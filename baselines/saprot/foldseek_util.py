import os
import time
import json
import numpy as np
import sys
from Bio.PDB import PDBParser, MMCIFParser

sys.path.append(".")

biopython_pdbparser = PDBParser(QUIET=True)

alphabet = 'ACDEFGHIKLMNPQRSTVWY'
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','MSE']
  
# Get structural seqs from pdb file
def get_struc_seq(path,
                  wt_seq_dic=None,
                  python=None,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_path: str = None,
                  plddt_threshold: float = 70.) -> dict:
    """
    
    Args:
        foldseek: Binary executable file of foldseek
        path: Path to pdb file
        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.
        process_id: Process ID for temporary files. This is used for parallel processing.
        plddt_path: Path to plddt file. If None, plddt will not be used.
        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    # assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"Pdb file not found: {path}"
    assert plddt_path is None or os.path.exists(plddt_path), f"Plddt file not found: {plddt_path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}.tsv"
    cmd = f"{os.path.dirname(python)}/foldseek structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)

    s = biopython_pdbparser.get_structure('',path)

    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]

            # Mask low plddt
            if plddt_path is not None:
                with open(plddt_path, "r") as r:
                    plddts = np.array(json.load(r)["confidenceScore"])
                    
                    # Mask regions with plddt < threshold
                    indices = np.where(plddts < plddt_threshold)[0]
                    np_seq = np.array(list(struc_seq))
                    np_seq[indices] = "#"
                    struc_seq = "".join(np_seq)
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]

            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    revise_seq = []
                    res_i = 0 
                    for res in s[0][chain].get_residues():
                        if res.get_full_id()[-1][0] != ' ':
                            continue
                        if res.resname not in alpha_3:
                            revise_seq.append('X')
                        else:
                            revise_seq.append(seq[res_i])
                        res_i += 1
                    assert len(revise_seq) == len(seq)
                    seq = ''.join(revise_seq)
                    wt_seq = wt_seq_dic[chain]
                    wt_i = 0
                    seq_i = 0
                    pad_seq = []
                    pad_struc_seq = []
                    while seq_i < len(seq):
                        aa = seq[seq_i]
                        if aa not in alphabet:
                            pad_seq.append('#')
                            pad_struc_seq.append(struc_seq[seq_i])
                            seq_i += 1
                        elif aa == wt_seq[wt_i] and aa in alphabet:
                            pad_seq.append(aa)
                            pad_struc_seq.append(struc_seq[seq_i])
                            wt_i += 1
                            seq_i += 1
                        elif wt_seq[wt_i] == 'X':
                            pad_seq.append('#')
                            pad_struc_seq.append('#')
                            wt_i += 1
                        if wt_i >= len(wt_seq):
                            break
                    if wt_i < len(wt_seq):
                        for _ in range(wt_i,len(wt_seq)):
                            wt_i += 1
                            pad_seq.append('#')
                            pad_struc_seq.append('#')
                    seq = ''.join(pad_seq)
                    struc_seq = ''.join(pad_struc_seq)
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
        
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict


if __name__ == '__main__':
    foldseek = "/sujin/bin/foldseek"
    # test_path = "/sujin/Datasets/PDB/all/6xtd.cif"
    test_path = "/sujin/Datasets/FLIP/meltome/af2_structures/A0A061ACX4.pdb"
    plddt_path = "/sujin/Datasets/FLIP/meltome/af2_plddts/A0A061ACX4.json"
    res = get_struc_seq(foldseek, test_path, plddt_path=plddt_path, plddt_threshold=70.)
    print(res["A"][1].lower())