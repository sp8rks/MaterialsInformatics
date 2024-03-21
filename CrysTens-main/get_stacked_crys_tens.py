import json
import numpy as np
import math
import argparse
from crys_tens import CrysTens

parser = argparse.ArgumentParser()
parser.add_argument("--crys_dict_path", type=str, help="Path to where a crystal dictionary is (See README).")
parser.add_argument("--num_examples", type=int, help="The number of crystals to include in the stacked CrysTens.")
parser.add_argument("--crys_tens_path", type=str, help="Path to where the stacked CrysTens should be stored.")

def generate_stacked_crys_tens(data, num_examples, crys_tens_path):
    
    crys_count = 0
    for crysidx in data:
        if len(data[crysidx]["siteList"]) <= 52:
            crys_count += 1
    stacked_crys_tens = np.zeros((min(crys_count, num_examples), 4, 64, 64))

    idx = 0
    crystal_list = []
    for crysidx in data:
        if idx >= num_examples:
            break
        if len(data[crysidx]["siteList"]) <= 52:
            crystal_list.append(crysidx)
            a = data[crysidx]["a"]
            b = data[crysidx]["b"]
            c = data[crysidx]["c"]

            alpha = data[crysidx]["alpha"]
            beta = data[crysidx]["beta"]
            gamma = data[crysidx]["gamma"]

            sg = data[crysidx]["sg"]

            coord_list = []
            atom_list = []
            for siteidx in range(len(data[crysidx]["siteList"])):
                coord_list.append((data[crysidx]["siteList"][siteidx][1][0], data[crysidx]["siteList"][siteidx][1][1], data[crysidx]["siteList"][siteidx][1][2]))
                atom_list.append(data[crysidx]["siteList"][siteidx][0])  

            crysgraph = CrysTens(a, b, c, alpha, beta, gamma, sg, atom_list, coord_list)
            crys = crysgraph.get_crys_tens(True, "divide_near_max")
            stacked_crys_tens[idx, :, :, :] = np.reshape(crys[:, :, :], (4, 64, 64))  
            idx += 1
    np.save(crys_tens_path, stacked_crys_tens)

def main():
    args = parser.parse_args()
    with open(args.crys_dict_path) as f:
        data = json.load(f)
    generate_stacked_crys_tens(data, args.num_examples, args.crys_tens_path)

if __name__ == "__main__":
    main()