import pandas as pd
import numpy as np
import seaborn as sns
import os

def create_dataset(PDBbind_dataset_path, general_set_PDBs_path, refined_set_PDBs_path, plot = False):
  """
  Produces a csv file containing PDB id, binding affinity, and set (general/refined)
  
  Inputs:
  1) path to PDBbind dataset; an example dataset is included in ML_pba/example_data.csv
  2) path to PDBbind general set PDBs
  3) path to PDBbind refined set PDBs
  4) plot = True will generate a plot of density as a function of binding affinity for general
     and refined sets
     
  Output:
  1) A cleaned csv containing PDB id, binding affinity, and set (general/refined):
     'pdb_affinity_data_cleaned.csv'
  """
  
  # load dataset
  data = pd.read_csv(PDBbind_dataset_path)
  
  # write csv of affinity data
  pdb_affinity_data = data[['pdbid','-logKd/Ki']]
  pdb_affinity_data.to_csv('pdb_affinity_data.csv')
  
  # check for NaNs in affinity data
  if pdb_affinity_data['-logKd/Ki'].isnull().any() != False:
    print('There are NaNs present in affinity data!')
    
  # create list of PDB id's in pdb_affinity_data
  pdbid_list = list(pdb_affinity_data['pdbid'])
  
  # remove affinites that do not have structural data
  missing = []
  for i in range(len(pdbid_list)):
    pdb = pdbid_list[i]
    if os.path.isdir(general_set_PDBs_path %pdb)==False and os.path.isdir(refined_set_PDBs_path %pdb)==False:
        missing.append(pdb)
  pdb_affinity_data = pdb_affinity_data[~np.in1d(pdb_affinity_data['pdbid'], list(missing))]

  # distinguish PDB id's in general and refined sets
  general_dict = {}
  refined_dict = {}
  for i in range(len(pdbid_list)):
    pdb = pdbid_list[i]
    if os.path.isdir(general_set_PDBs_path %pdb)==True:
        general_dict[pdb] = 'general'
    if os.path.isdir(refined_set_PDBs_path %pdb)==True:
        refined_dict[pdb] = 'refined'
   
  # add 'set' column to pdb_affinity_data and fill with 'general'/'refined'
  pdb_affinity_data['set'] = np.nan
  pdb_affinity_data.loc[np.in1d(pdb_affinity_data['pdbid'], list(general_dict)), 'set'] = 'general'
  pdb_affinity_data.loc[np.in1d(pdb_affinity_data['pdbid'], list(refined_dict)), 'set'] = 'refined'
  
  # write out csv of cleaned dataset
  pdb_affinity_data[['pdbid', '-logKd/Ki', 'set']].to_csv('pdb_affinity_data_cleaned.csv', index=False)
  
  # read in and view the cleaned dataset
  display(pd.read_csv('pdb_affinity_data_cleaned.csv'))
  
  if plot == True:
    # plot affinity distributions for general and refined sets
    grid = sns.FacetGrid(pdb_affinity_data, row='set', row_order=['general', 'refined'],
                     size=3, aspect=2)
    grid.map(sns.distplot, '-logKd/Ki')
  else:
    return
