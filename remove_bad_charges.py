import csv
impport pybel
import pandas as pd
import os

# define function to select pocket mol2 files with atoms that have charges greater than +- 2 (unfeasible)
def get_charge(molecule):
    for i, atom in enumerate(molecule):
        if atom.atomicnum > 1:
            if (abs(atom.__getattribute__('partialcharge'))>=2): # this charge cutoff can be varied
              return 'bad_complex'
            else: 
              return 'no_error'
            
# call this function to remove PDB id's with bad charges from the affinity data csv file
def remove_bad_charges(affinity_data_path, mol2_path, general_PDBs_path, refined_PDBs_path)
  """
  input:
  1) path/to/affinity/data.csv
  2) path/to/mol2/files
  3) path/to/PDBs/in/general_set
  4) path/to/PDBs/in/refined_set
  
  output:
  1) returns a csv file containing only the PDB id's that will be used, saved as:
     'affinity_data_cleaned_charge_cutoff_2.csv'
  
  """
  # read in affinity data csv file
  affinities = pd.read_csv(affinity_data_path)
  
  # convert pdb id's to numpy array
  pdbids_cleaned = affinities['pdbid'].to_numpy()
  
  # define empty lists to contain pocket and ligand files
  pocket_files = []
  ligand_files = []
  bad_complexes = ['3ary', '4bps', '4mdq', '2iw4'] # these are the PDBs for which Chimera failed to calculate charges and that failed hdf5 conversion (next step)
  
  # fill lists with paths to pocket and ligand mol2 files
  for i in range(0, len(pdbids_cleaned)):
    pocket_files.append(mol2_path + pdbids_cleaned[i] + '_pocket.mol2')
    if affinities['set'][i]=='general':
      ligand_files.append(general_PDBs_path + pdbids_cleaned[i] + '/' + pdbids_cleaned[i] + '_ligand.mol2')
    else:
      ligand_files.append(refined_PDBs_path + pdbids_cleaned[i] + '/' + pdbids_cleaned[i] + '_ligand.mol2')

  # iterate through pocket files and identify the bad_complexes
  count = 0
  for pfile in pocket_files:
              count +=1
              try:
                  pocket = next(pybel.readfile('mol2', pfile))
              except:
                  raise IOError('Cannot read %s file' % pfile)
              if(get_charge(pocket)==('bad_complex')):
                  bad_complexes.append((os.path.splitext(os.path.split(pfile)[1])[0]).split('_')[0])  
                  
  # iterate through ligand files and identify the bad_complexes
  count = 0
  for lfile in ligand_files:
              count +=1
              try:
                  ligand = next(pybel.readfile('mol2', lfile))
              except:
                  raise IOError('Cannot read %s file' % lfile)
              if(get_charge(ligand)=='bad_complex'):
                  pdbid = (os.path.splitext(os.path.split(lfile)[1])[0]).split('_')[0]
                  if pdbid not in bad_complexes:
                      bad_complexes.append(pdbid)
                      
  # remove problematic pdb files from data
  with open(affinity_data_path, 'rt') as inp, open('affinity_data_cleaned_charge_cutoff_2.csv', 'w') as out:
      writer = csv.writer(out)
      for row in csv.reader(inp):
          if not row[0] in bad_complexes:
              writer.writerow(row)
  
  # show updated affinity data csv file
  display(pd.read_csv('affinity_data_cleaned_charge_cutoff_2.csv'))
