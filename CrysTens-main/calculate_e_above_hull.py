import numpy as np
from sklearn.cluster import KMeans
from pymatgen.io.cif import CifWriter
from pymatgen.core import Lattice, Structure
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester
from m3gnet.models import M3GNet
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, help="Path to where crystals are.")
parser.add_argument("--m3gnet_model_path", type=str, help="Path to where M3GNet model is.")
parser.add_argument("--ehull_path", type=str, help="Path to where energy above calculations will be stored.")
parser.add_argument("--mp_api_key", type=str, help="API key for materials project.")

def predict_ehull(dir, model_path, output_path, api_key):
    m3gnet_e_form = M3GNet.from_dir(model_path)
    ehull_list = []
    for file_name in os.listdir(dir):
        crystal = Structure.from_file(dir + file_name, sort = True, merge_tol=0.01)
        try:
            e_form_predict = m3gnet_e_form.predict_structure(crystal)
        except:
            print("Could not predict formation energy of ", crystal)
            ehull_list.append((file_name, "N/A"))
            break
    
        elements = ''.join([i for i in crystal.formula if not i.isdigit()]).split(" ")
        mat_api_key = api_key
        mpr = MPRester(mat_api_key)
        all_compounds = mpr.summary.search(elements = elements)
        insert_list = []
        for compound in all_compounds:
            for element in ''.join([i for i in str(compound.composition) if not i.isdigit()]).split(" "):
                if element not in elements and element not in insert_list:
                    insert_list.append(element)
        for element in elements + insert_list:
            all_compounds += mpr.summary.search(elements = [element], num_elements = (1,1))

        pde_list = []
        for i in range(len(all_compounds)):
            comp = str(all_compounds[i].composition.reduced_composition).replace(" ", "")
            pde_list.append(ComputedEntry(comp, all_compounds[i].formation_energy_per_atom))
    
        try:
            diagram = PhaseDiagram(pde_list)
            _, pmg_ehull = diagram.get_decomp_and_e_above_hull(ComputedEntry(Composition(crystal.formula.replace(" ", "")), e_form_predict[0][0].numpy()))
            ehull_list.append((file_name, pmg_ehull))
        except:
            print("Could not create phase diagram")
            ehull_list.append((file_name, "N/A"))
            continue
    np.save(output_path, np.array(ehull_list))

def main():
    args = parser.parse_args()
    predict_ehull(args.dir, args.m3gnet_model_path, args.ehull_path, args.mp_api_key)

if __name__ == "__main__":
    main()
