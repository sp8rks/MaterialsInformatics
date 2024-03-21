import numpy as np
from typing import List
from sklearn.cluster import KMeans
import argparse
import os
from pymatgen.io.cif import CifWriter
from pymatgen.core import Lattice, Structure

#Normalization Constants
direction_fix = 0.9999933
normalize_avg_std = {'atom': [27.503517675002012, 22.417031299395962, 1.1822938247722463, 3.189382276810422],
                        'x': [0.4252026276473906, 0.2934632568241515, 1.4489126586030487, 1.9586462438477188],
                        'y': [0.43638668432502076, 0.291906737713482, 1.4949524212536385, 1.9307764302498376],
                        'z': [0.4442944632355303, 0.2930835040637591, 1.515931320170363, 1.8960314349305054],
                        'a': [7.576401996204324, 2.9769371381202365, 1.9530147015047585, 11.78647595694663],
                        'b': [7.90142069082381, 3.559553637232651, 1.724660257008108, 10.47366695621996],
                        'c': [9.930851747522944, 6.749988198167417, 1.150054121521355, 15.742864303277916],
                        'alpha': [89.9627072033887, 3.1373304207017987, 9.624649990367965, 9.574156613663968],
                        'beta': [93.78688773759445, 9.618284407608558, 3.543759603389515, 4.640756123524075],
                        'gamma': [95.65975023932918, 12.237549174937287, 3.0022964332249518, 1.9889807520054794],
                        'sg': [99.50188767306616, 76.65640490497339, 1.284979223786628, 1.7023771528120184],
                        'dir': [-4.058145803781749e-20, 0.4099350434569409, 2.4393946048146806, 2.4393946048146806],
                        'length': [0.6693464353927602, 0.236887249528413, 2.73534326062805, 4.159475792980244]}

#Crystal parameters as they appear in CrysTens
attribute_list = ["atom", "x", "y", "z", "a", "b", "c", "alpha", "beta", "gamma", "sg"]
parameter_list = ["a", "b", "c", "alpha", "beta", "gamma", "sg"]

parser = argparse.ArgumentParser()
parser.add_argument("--crys_tens_path", type=str, help="Path to where the CrysTens is stored.")
parser.add_argument("--cif_folder", type=str, help="Path to where the generated CIF folder is.")
parser.add_argument("--stats_folder", type=str, help="Path to where the generated statistics folder is.")

def unnormalize_crys_tens(crys_tens):
  """Unnormalizes a given CrysTens.

    Args:
      crys_tens: The normalized (generated) CrysTens.

    Returns:
      The unnormalized CrysTens.
    """
  for site_idx in range(crys_tens.shape[0] - 12):

    #Check if an atom is present
    if np.any(crys_tens[12 + site_idx, :, :]):
      for att_idx, att in enumerate(attribute_list):
        crys_tens[att_idx, 12 + site_idx, :] *= normalize_avg_std[att][2] + normalize_avg_std[att][3]
        crys_tens[att_idx, 12 + site_idx, :] -= normalize_avg_std[att][2]
        crys_tens[att_idx, 12 + site_idx, :] *= normalize_avg_std[att][1]
        crys_tens[att_idx, 12 + site_idx, :] += normalize_avg_std[att][0]
        
        crys_tens[12 + site_idx, att_idx, :] *= normalize_avg_std[att][2] + normalize_avg_std[att][3]
        crys_tens[12 + site_idx, att_idx, :] -= normalize_avg_std[att][2]
        crys_tens[12 + site_idx, att_idx, :] *= normalize_avg_std[att][1]
        crys_tens[12 + site_idx, att_idx, :] += normalize_avg_std[att][0] 

      for adj_idx in range(crys_tens.shape[0] - 12):
        if site_idx != adj_idx:
          crys_tens[12 + site_idx, 12 + adj_idx, 0] *= normalize_avg_std["length"][2] + normalize_avg_std["length"][3]
          crys_tens[12 + site_idx, 12 + adj_idx, 0] -= normalize_avg_std["length"][2]
          crys_tens[12 + site_idx, 12 + adj_idx, 0] *= normalize_avg_std["length"][1]
          crys_tens[12 + site_idx, 12 + adj_idx, 0] += normalize_avg_std["length"][0]

          crys_tens[12 + site_idx, 12 + adj_idx, 1:] *= normalize_avg_std["dir"][2] + normalize_avg_std["dir"][3]
          crys_tens[12 + site_idx, 12 + adj_idx, 1:] -= normalize_avg_std["dir"][2]
          crys_tens[12 + site_idx, 12 + adj_idx, 1:] *= normalize_avg_std["dir"][1]
          crys_tens[12 + site_idx, 12 + adj_idx, 1:] += normalize_avg_std["dir"][0]

  return crys_tens


def get_cif(crys_tens, file_path, ref_percent: float = 0.2, rel_coord_percent: float = None, num_coord_clusters: int = None, num_atom_clusters: int = 3, generator: str = "Real CIF", dir_diff_avg: List = None, pot_sites = None, top_sites: int = None):
    """Deconstructs a CrysTens and produces a CIF if possible. Simultaneously collects statistics about CrysTens quality.

    Args:
      crys_tens: The generated CrysTens that is going to be turned into a CIF
      file_path: Where the CIF should be stored once it has been generated
      ref_percent: The tolerance of when a given row/column should not be considered as a new atom
      rel_coord_percent: How much of the final coordinates should rely on the relative distance matrices
      num_coord_clusters: How many K-Means groups to group the number of coordinate values
      num_atom_clusters: How many K-Means groups to group the number of unique elements
      generator: The name of the generative model that generated crys_tens
      dir_diff_avg: A dictionary containing the difference between final coordinate positions and relative coordinate positions
      pot_sites: TODO
      top_sites: TODO

    Returns:
      The space group number of the CIF as well as the distance between the generated 
      pairwise distance matrix and the reconstructed matrix from the final coordinate positions
    """

    dir_diff_latest = []
    crystal_cif = {}
    ref_angle = crys_tens[12, 7, 0]
    ref_avg = ref_angle

    #Uses ref_percent to determine when there are no atoms left in CrysTens (column/row[0] = 0)
    for num in range(13, crys_tens.shape[0]):
      if abs(crys_tens[num, 7, 0] - ref_avg)/ref_avg >= ref_percent:
        break
      else:
        ref_angle += crys_tens[num, 7, 0]
        reference_average = ref_angle / (num - 11)
    
    #Averages each value with its reflected value
    constrained_crys_tens = crys_tens[:num, :num, :]
    for i in range(12, constrained_crys_tens.shape[0]):
      for j in range(12):
        avg_val = (np.sum(constrained_crys_tens[i, j, :]) + np.sum(constrained_crys_tens[j, i, :]))/(2 * constrained_crys_tens.shape[2]) 
        constrained_crys_tens[i, j, :], constrained_crys_tens[j, i, :] = avg_val, avg_val

    #Finds the average of parameters, angles, and space group number to use as the final value
    for i in range(4, 11):
      sum_val = 0
      for j in range(12, constrained_crys_tens.shape[0]):
        sum_val += np.sum(constrained_crys_tens[i, j, :]) + np.sum(constrained_crys_tens[j, i, :])
      avg_val = sum_val / (2 * (constrained_crys_tens.shape[0] - 12) * constrained_crys_tens.shape[2])
      constrained_crys_tens[i, 12:, :], constrained_crys_tens[12:, i, :] = avg_val, avg_val
      crystal_cif[parameter_list[i - 4]] = avg_val

    #Makes an attempt to symmetrize the CrysTens
    for i in range(12, constrained_crys_tens.shape[0]):
      for j in range(12, constrained_crys_tens.shape[0]):
        avg_val = (constrained_crys_tens[i, j, 0] + constrained_crys_tens[j, i, 0])/2
        constrained_crys_tens[i, j, 0], constrained_crys_tens[j, i, 0] = avg_val, avg_val
        for k in range(1, 4):
          if constrained_crys_tens[i, j, k] > 0:
            avg_val = (constrained_crys_tens[i, j, k] + abs(constrained_crys_tens[j, i, k]))/2
            constrained_crys_tens[i, j, k], constrained_crys_tens[j, i, k] = avg_val, -avg_val
          else:
            avg_val = (abs(constrained_crys_tens[i, j, k]) + constrained_crys_tens[j, i, k])/2
            constrained_crys_tens[i, j, k], constrained_crys_tens[j, i, k] = -avg_val, avg_val

    #Finds the absolute x, y, and z coordinates of each atom as well as the relative coordinates of each atom (according to the distance matrices)
    crystal_cif["site_list"] = {}
    for i in range(12, constrained_crys_tens.shape[0]):
      crystal_cif["site_list"][i - 12] = {}
      crystal_cif["site_list"][i - 12]["atom"] = constrained_crys_tens[i, 0, 0]
      crystal_cif["site_list"][i - 12]["x"] = constrained_crys_tens[i, 1, 0]
      crystal_cif["site_list"][i - 12]["y"] = constrained_crys_tens[i, 2, 0]
      crystal_cif["site_list"][i - 12]["z"] = constrained_crys_tens[i, 3, 0]

      crystal_cif["site_list"][i - 12]["adj_list"] = []

    for i in range(12, constrained_crys_tens.shape[0]):
      for j in range(12, constrained_crys_tens.shape[0]):
        adj_x = crystal_cif["site_list"][i - 12]["x"] - constrained_crys_tens[i, j, 1]
        adj_y = crystal_cif["site_list"][i - 12]["y"] - constrained_crys_tens[i, j, 2]
        adj_z = crystal_cif["site_list"][i - 12]["z"] - constrained_crys_tens[i, j, 3]
        crystal_cif["site_list"][j - 12]["adj_list"].append((adj_x, adj_y, adj_z))

    site_list = []
    atom_list = []
    if rel_coord_percent is None:
      rel_coord_percent = 1 - (1/constrained_crys_tens.shape[0])
    
    #Calculates the final coordinate positions and calculates the difference between the final positions and the relative positions
    for site_idx in crystal_cif["site_list"]:
      site = crystal_cif["site_list"][site_idx]
      site_x = site["x"]
      site_y = site["y"]
      site_z = site["z"]
      adj_x_list = []
      adj_y_list = []
      adj_z_list = []
      adj_coord_list =site["adj_list"]
      for adj_idx in range(len(adj_coord_list)):
        adj_coord = adj_coord_list[adj_idx]
        adj_x_list.append(adj_coord[0])
        adj_y_list.append(adj_coord[1])
        adj_z_list.append(adj_coord[2])
      
      site_coord_percent = 1 - rel_coord_percent
      x_rel = site_coord_percent*site_x + rel_coord_percent*np.average(adj_x_list)
      y_rel = site_coord_percent*site_y + rel_coord_percent*np.average(adj_y_list)
      z_rel = site_coord_percent*site_z + rel_coord_percent*np.average(adj_z_list)
      atom_list.append(np.around(site["atom"]))
      site_list.append((x_rel, y_rel, z_rel))
      if dir_diff_avg is not None:
        dir_diff_latest.append(abs(site_x - x_rel)/len(adj_x_list))
        dir_diff_latest.append(abs(site_y - y_rel)/len(adj_y_list))
        dir_diff_latest.append(abs(site_z - z_rel)/len(adj_z_list))

    #Reconstructs a new pairwise distance matrix and compares it to the one generated in CrysTens Layer 1
    reconstructed_pairwise = np.zeros((constrained_crys_tens.shape[0] - 12, constrained_crys_tens.shape[1] - 12, 1))
    for site_idx in range(len(site_list)):
      site = site_list[site_idx]
      for adj_idx in range(len(site_list)):
        adj = site_list[adj_idx]
        x_diff = site[0] - adj[0]
        y_diff = site[1] - adj[1]
        z_diff = site[2] - adj[2]
        length = (x_diff**2 + y_diff**2 + z_diff**2)**(1/2)

        reconstructed_pairwise[site_idx, adj_idx, 0] = length
        reconstructed_pairwise[adj_idx, site_idx, 0] = length
    
    pairwise_error = np.abs(constrained_crys_tens[12:, 12:, 0] - reconstructed_pairwise[:, :, 0])

    #K-Means Clustering for the coordinate values
    dir_diff_avg.append(dir_diff_latest)
    kmeans_coord_list = []
    for i in range(len(site_list)):
      for j in range(3):
        kmeans_coord_list.append(site_list[i][j])

    if num_coord_clusters is None:
      num_coord_clusters = len(kmeans_coord_list)
    
    num_coord_clusters = min(len(kmeans_coord_list), num_coord_clusters)

    kmeans_coord = KMeans(n_clusters = num_coord_clusters).fit(np.array(kmeans_coord_list).reshape(-1, 1))
    for i in range(len(kmeans_coord_list)):
      kmeans_coord_list[i] = kmeans_coord.cluster_centers_[kmeans_coord.labels_[i]][0]
    
    for i in range(0, len(kmeans_coord_list), 3):
      site_list[i//3] = [kmeans_coord_list[i], kmeans_coord_list[i + 1], kmeans_coord_list[i + 2]]
    
    
    #TODO Add PotScoring

    #K-Means Clustering for the atom values
    if num_atom_clusters is None:
      num_atom_clusters = len(atom_list)
    
    num_atom_clusters = min(num_atom_clusters, len(atom_list))
    kmeans_atom = KMeans(n_clusters=int(num_atom_clusters)).fit(np.array(atom_list).reshape(-1, 1))

    for i in range(len(atom_list)):
      atom_list[i] = np.around(kmeans_atom.cluster_centers_[kmeans_atom.labels_[i]])[0]

    #Remove duplicates
    trimmed_site_list = []
    trimmed_atom_list = []
    dup_check = set()
    for i in range(len(atom_list)):
      if (atom_list[i], tuple(site_list[i])) not in dup_check:
        dup_check.add((atom_list[i], tuple(site_list[i])))
        trimmed_site_list.append(site_list[i])
        trimmed_atom_list.append(atom_list[i])

    #Creates the CIF
    lattice = Lattice.from_parameters(a = crystal_cif["a"], b = crystal_cif["b"], c = crystal_cif["c"], alpha = crystal_cif["alpha"], beta = crystal_cif["beta"], gamma = crystal_cif["gamma"])
    struct = Structure(lattice = lattice, species = trimmed_atom_list, coords = trimmed_site_list, to_unit_cell=True)
    written_cif = str(CifWriter(struct))
    with open(file_path, "w") as file:
      file.write("Generated by: " + generator + "\n" + "Num unique sites: " + str(num_coord_clusters) + "\n" + "Num unique elements: " + str(num_atom_clusters) + "\n\n" + written_cif)
    
    return crystal_cif["sg"], np.sum(pairwise_error)/len(site_list)


def get_statistics(crys_tens_path, cif_folder, stats_folder = None):
  """Gets all of the pertinent statistics of a group of CrysTens' and creates CIFs from the stacked CrysTens'

    Args:
      crys_tens_path: Path to a numpy array of CrysTens' of size (#CrysTens', 64, 64, 4)
      cif_folder: The folder where the generated CIFs are to be stored
      stats_folder: The folder where the stats file will be generated
  """

  #Create the statistics lists
  xyzvar = []
  paramvar = []
  anglevar = []
  sgvar = []
  dir_diff_avg = []
  sg_list = []
  pairwise_avg = []

  np_crys = np.load(crys_tens_path)
  crys_count = 0
  for crys in (np_crys):
    #Make sure CrysTens is of the right shape and unnormalized
    if np_crys.shape != (64, 64, 4):
        new_crys = np.zeros((64, 64, 4))
        for i in range(4):
          new_crys[:, :, i] = crys[i, :, :]
        crys = new_crys[:]
    crys_tens = unnormalize_crys_tens(crys)

    #Get the CIF
    sg, pairwise_error = get_cif(crys_tens, cif_folder + "CIF" + str(crys_count) + ".cif", 0.2, None, None, 3, "Real CIF", dir_diff_avg)
    try:
      load_crystal = Structure.from_file(cif_folder + "CIF" + str(crys_count) + ".cif")
      crys_count += 1
      sg_list.append(sg)
      pairwise_avg.append(pairwise_error)
    except:
      #If the CIF was not generated correctly, move on to the next CrysTens, do not collect any statistics
      continue

    #Uses ref_percent to determine when there are no atoms left in CrysTens (column/row[0] = 0)
    ref_angle = crys_tens[12, 7, 0]
    ref_avg = ref_angle
    for num in range(13, crys_tens.shape[0]):
        if abs(crys_tens[num, 7, 0] - ref_avg)/ref_avg >= 0.2:
          break
        else:
          ref_angle += crys_tens[num, 7, 0]
          reference_average = ref_angle / (num - 11)
    crys_tens = crys_tens[:num, :num, :]

    #Gets the variance between reflected coordinates
    for i in range(12, crys_tens.shape[0]):
        for j in range(1, 3):
          xyz = []
          for k in range(4):
            xyz.append(crys_tens[i, j, k])
            xyz.append(crys_tens[j, i, k])
          xyzvar.append(np.var(xyz))
    
    #Gets the variance in parameters, angles, and space group numbers
    for i in range(4, 11):
        row = crys_tens[i, 12:, :]
        col = crys_tens[12:, i, :]
        total = []
        for j in range(4):
          total += list(row[:, j])
          total += list(col[:, j])
        if 3 < i < 7:
            paramvar.append(np.var(total))
        elif 7 <= i < 10:
            anglevar.append(np.var(total))
        else:
            sgvar.append(np.var(total))
  
  relative_diff_sum = 0
  relative_diff_count = 0
  for i in range(len(dir_diff_avg)):
    relative_diff_sum += sum(dir_diff_avg[i])
    relative_diff_count += len(dir_diff_avg[i])
  relative_diff_sum /= relative_diff_count

  stats_name_list = [paramvar, anglevar, sgvar, xyzvar, pairwise_avg]
  stats_list = []
  for stat_idx in range(len(stats_name_list)):
    stats_list.append(np.sum(stats_name_list[stat_idx])/len(stats_name_list[stat_idx]))
  
  stats_list.append(relative_diff_sum)

  if stats_folder is not None:
    np.save(os.path.join(stats_folder, "sg_list.npy"), np.array(sg_list))
    name_list = ["Lattice Parameter Variance", "Lattice Angle Variance", "Space Group Number Variance", "Fractional Coordinate Variance", "Reconstructed Pairwise Difference", "Relative Coordinate Difference"]
    with open(os.path.join(stats_folder, "stats"), "w") as f:
      for stat_idx in range(len(name_list)):
        f.write(name_list[stat_idx] + ": " + str(stats_list[stat_idx]) + "\n")


def main():
    args = parser.parse_args()
    get_statistics(args.crys_tens_path, args.cif_folder, args.stats_folder)

if __name__ == "__main__":
    main()

