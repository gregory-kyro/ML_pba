from featurizer import Featurizer  ##See featurizer module for more details
import xml.etree.ElementTree as ET

##Define function to extract features from the binding pocket mol2 file
def __get_pocket():
    for pfile in pocket_files:
        try:
            pocket = next(pybel.readfile('mol2', pfile))
        except:
            raise IOError('Cannot read %s file' % pfile)

        pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
        pocket_vdw = parse_mol_vdw(mol=pocket, element_dict=element_dict)
        yield (pocket_coords, pocket_features, pocket_vdw)

##Define function to extract information from elements.xml file
def parse_element_description(desc_file):
    element_info_dict = {}
    element_info_xml = ET.parse(desc_file)
    for element in element_info_xml.getiterator():
        if "comment" in element.attrib.keys():
            continue
        else:
            element_info_dict[int(element.attrib["number"])] = element.attrib

    return element_info_dict


##Define function to create a list of van der Waals radii for a molecule
def parse_mol_vdw(mol, element_dict):
    vdw_list = []
    for atom in mol.atoms:
        # NOTE: to be consistent between featurization methods, throw out the hydrogens
        if int(atom.atomicnum) == 1:
            continue
        if int(atom.atomicnum) == 0:
            continue
        else:
            vdw_list.append(float(element_dict[atom.atomicnum]["vdWRadius"]))
    return np.asarray(vdw_list)
    
 


##Call this function to convert cleaned mol2 files into one hdf file
def convert_to_hdf (cleaned_csv_path, output, mol2_path, general_PDBs_path, refined_PDBs_path):
   
   """
  input:
  1) path/to/cleaned/affinity/data.csv
  2) path/to/output/hdf/file.hdf
  3) path/to/mol2/files
  4) path/to/PDBs/in/general_set
  5) path/to/PDBs/in/refined_set
  
  output:
  1) returns an hdf file containing only the PDB id's that will be used, saved at:
     'path/to/output/hdf/file.hdf'
  """
    
  #read in data from elements.xml file
  element_dict = parse_element_description("ML_pba/elements.xml")


  # read in cleaned affinity data csv file
  affinities = pd.read_csv(cleaned_csv_path)

  # convert pdb id's to numpy array
  pdbids_cleaned = affinities['pdbid'].to_numpy()

  # define empty lists to contain pocket and ligand files
  pocket_files = []
  ligand_files = []


  # fill lists with paths to pocket and ligand mol2 files
  for i in range(0, len(pdbids_cleaned)):
    pocket_files.append(mol2_path + pdbids_cleaned[i] + '_pocket.mol2')
    if affinities['set'][i]=='general':
      ligand_files.append(general_PDBs_path + pdbids_cleaned[i] + '/' + pdbids_cleaned[i] + '_ligand.mol2')
    else:
      ligand_files.append(refined_PDBs_path + pdbids_cleaned[i] + '/' + pdbids_cleaned[i] + '_ligand.mol2')

  num_pockets = len(pocket_files)
  num_ligands = len(ligand_files)
  if num_pockets != num_ligands:
      raise IOError('%s pockets specified for %s ligands. You must providea single pocket or a separate pocket for each ligand' % (num_pockets, num_ligands))



  if '-logKd/Ki' not in affinities.columns:
      raise ValueError('There is no `-logKd/Ki` column in the table')
  elif 'pdbid' not in affinities.columns:
      raise ValueError('There is no `pdbid` column in the table')
  affinities = affinities.set_index('pdbid')['-logKd/Ki']
 

  featurizer = Featurizer()




  #create a new hdf file to store all of the data
  with h5py.File(output, 'w') as f:
  
      pocket_generator = __get_pocket()
      
      for lfile in ligand_files:
          # use pdbid as dataset name
          name = os.path.splitext(os.path.split(lfile)[1])[0]
          pdbid = name.split('_')[0]
          
          #read ligand file using pybel
          try:
              ligand = next(pybel.readfile('mol2', lfile))
          except:
              raise IOError('Cannot read %s file' % lfile)
          
          #extract features from ligand
          ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
          ligand_vdw = parse_mol_vdw(mol=ligand, element_dict=element_dict)
          
          #extract features from pocket
          pocket_coords, pocket_features, pocket_vdw = next(pocket_generator)

          #Center the ligand and pocket coordinates
          centroid = ligand_coords.mean(axis=0)
          ligand_coords -= centroid
          pocket_coords -= centroid

          #assemble the features into one large numpy array: rows are heavy atoms, columns are coordinates and features
          data = np.concatenate(
              (np.concatenate((ligand_coords, pocket_coords)),
               np.concatenate((ligand_features, pocket_features))),
              axis=1,
          )
          #concatenate van der Waals radii into one numpy array
          vdw_radii = np.concatenate((ligand_vdw, pocket_vdw))
          
          #create a new dataset for this complex in the hdf file
          dataset = f.create_dataset(pdbid, data=data, shape=data.shape,
                                     dtype='float32', compression='lzf')
                                     
          #add the affinity and van der Waals radii as attributes for this dataset 
          dataset.attrs['affinity'] = affinities.loc[pdbid]
          assert len(vdw_radii) == data.shape[0]
          dataset.attrs["van_der_waals"] = vdw_radii
          
          
