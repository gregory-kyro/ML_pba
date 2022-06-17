def convert_to_hdf(affinity_data_path, output_total_hdf, mol2_path, general_PDBs_path, refined_PDBs_path, output_val_hdf, output_train_hdf, output_test_hdf):
   
   """
  This function converts the mol2 files into three cleaned hdf files containing datasets for training, testing, and validation complexes, respectively.
   
  input:
  1) path/to/affinity/data.csv
  2) path/to/total/output/hdf/file.hdf
  3) path/to/mol2/files
  4) path/to/PDBs/in/general_set
  5) path/to/PDBs/in/refined_set
  6) path/to/output/validation/hdf/file.hdf
  7) path/to/output/training/hdf/file.hdf
  8) path/to/output/testing/hdf/file.hdf
  
  output:
  1)  a complete hdf file containing featurized data for all of the PDB id's that will be used, saved as:
     'path/to/output/hdf/file.hdf'
  2)  a csv file containing all of the PDB id's that will be used, saved as:
     'affinity_data_cleaned_charge_cutoff_2.csv'
  3)  three hdf files containing the featurized data for the PDB id's that will be used in validation, training, and testing sets, saved as:
     'path/to/output/validation/hdf/file.hdf', 'path/to/output/training/hdf/file.hdf', and 'path/to/output/testing/hdf/file.hdf', respectively
  """
   
   # This is the source code for the tfbio Featurizer() class, originally developed here: https://gitlab.com/cheminfIBB/tfbio/-/blob/master/tfbio/data.py
   import pickle
   import numpy as np
   import openbabel.pybel
   from openbabel.pybel import Smarts
   from math import ceil, sin, cos, sqrt, pi
   from itertools import combinations
   
   class Featurizer():
       """Calcaulates atomic features for molecules. Features can encode atom type,
       native pybel properties or any property defined with SMARTS patterns
       Attributes
       ----------
       FEATURE_NAMES: list of strings
           Labels for features (in the same order as features)
       NUM_ATOM_CLASSES: int
           Number of atom codes
       ATOM_CODES: dict
           Dictionary mapping atomic numbers to codes
       NAMED_PROPS: list of string
           Names of atomic properties to retrieve from pybel.Atom object
       CALLABLES: list of callables
           Callables used to calculcate custom atomic properties
       SMARTS: list of SMARTS strings
           SMARTS patterns defining additional atomic properties
       """

       def __init__(self, atom_codes=None, atom_labels=None,
                    named_properties=None, save_molecule_codes=True,
                    custom_properties=None, smarts_properties=None,
                    smarts_labels=None):

           """Creates Featurizer with specified types of features. Elements of a
           feature vector will be in a following order: atom type encoding
           (defined by atom_codes), Pybel atomic properties (defined by
           named_properties), molecule code (if present), custom atomic properties
           (defined `custom_properties`), and additional properties defined with
           SMARTS (defined with `smarts_properties`).
           Parameters
           ----------
           atom_codes: dict, optional
               Dictionary mapping atomic numbers to codes. It will be used for
               one-hot encoging therefore if n different types are used, codes
               shpuld be from 0 to n-1. Multiple atoms can have the same code,
               e.g. you can use {6: 0, 7: 1, 8: 1} to encode carbons with [1, 0]
               and nitrogens and oxygens with [0, 1] vectors. If not provided,
               default encoding is used.
           atom_labels: list of strings, optional
               Labels for atoms codes. It should have the same length as the
               number of used codes, e.g. for `atom_codes={6: 0, 7: 1, 8: 1}` you
               should provide something like ['C', 'O or N']. If not specified
               labels 'atom0', 'atom1' etc are used. If `atom_codes` is not
               specified this argument is ignored.
           named_properties: list of strings, optional
               Names of atomic properties to retrieve from pybel.Atom object. If
               not specified ['hyb', 'heavyvalence', 'heterovalence',
               'partialcharge'] is used.
           save_molecule_codes: bool, optional (default True)
               If set to True, there will be an additional feature to save
               molecule code. It is usefeul when saving molecular complex in a
               single array.
           custom_properties: list of callables, optional
               Custom functions to calculate atomic properties. Each element of
               this list should be a callable that takes pybel.Atom object and
               returns a float. If callable has `__name__` property it is used as
               feature label. Otherwise labels 'func<i>' etc are used, where i is
               the index in `custom_properties` list.
           smarts_properties: list of strings, optional
               Additional atomic properties defined with SMARTS patterns. These
               patterns should match a single atom. If not specified, deafult
               patterns are used.
           smarts_labels: list of strings, optional
               Labels for properties defined with SMARTS. Should have the same
               length as `smarts_properties`. If not specified labels 'smarts0',
               'smarts1' etc are used. If `smarts_properties` is not specified
               this argument is ignored.
           """

           # Remember names of all features in the correct order
           self.FEATURE_NAMES = []
           if atom_codes is not None:
               if not isinstance(atom_codes, dict):
                   raise TypeError('Atom codes should be dict, got %s instead'
                                   % type(atom_codes))
               codes = set(atom_codes.values())
               for i in range(len(codes)):
                   if i not in codes:
                       raise ValueError('Incorrect atom code %s' % i)
               self.NUM_ATOM_CLASSES = len(codes)
               self.ATOM_CODES = atom_codes
               if atom_labels is not None:
                   if len(atom_labels) != self.NUM_ATOM_CLASSES:
                       raise ValueError('Incorrect number of atom labels: '
                                        '%s instead of %s'
                                        % (len(atom_labels), self.NUM_ATOM_CLASSES))
               else:
                   atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
               self.FEATURE_NAMES += atom_labels
           else:
               self.ATOM_CODES = {}
               metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                         + list(range(37, 51)) + list(range(55, 84))
                         + list(range(87, 104)))

               # List of tuples (atomic_num, class_name) with atom types to encode
               atom_classes = [
                   (5, 'B'),
                   (6, 'C'),
                   (7, 'N'),
                   (8, 'O'),
                   (15, 'P'),
                   (16, 'S'),
                   (34, 'Se'),
                   ([9, 17, 35, 53], 'halogen'),
                   (metals, 'metal')
               ]
               for code, (atom, name) in enumerate(atom_classes):
                   if type(atom) is list:
                       for a in atom:
                           self.ATOM_CODES[a] = code
                   else:
                       self.ATOM_CODES[atom] = code
                   self.FEATURE_NAMES.append(name)
               self.NUM_ATOM_CLASSES = len(atom_classes)
           if named_properties is not None:
               if not isinstance(named_properties, (list, tuple, np.ndarray)):
                   raise TypeError('named_properties must be a list')
               allowed_props = [prop for prop in dir(openbabel.pybel.Atom)
                                if not prop.startswith('__')]
               for prop_id, prop in enumerate(named_properties):
                   if prop not in allowed_props:
                       raise ValueError(
                           'named_properties must be in pybel.Atom attributes,'
                           ' %s was given at position %s' % (prop_id, prop)
                       )
               self.NAMED_PROPS = named_properties
           else:
               # pybel.Atom properties to save
               self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree',
                                   'partialcharge']
           self.FEATURE_NAMES += self.NAMED_PROPS
           if not isinstance(save_molecule_codes, bool):
               raise TypeError('save_molecule_codes should be bool, got %s '
                               'instead' % type(save_molecule_codes))
           self.save_molecule_codes = save_molecule_codes
           if save_molecule_codes:
               # Remember if an atom belongs to the ligand or to the protein
               self.FEATURE_NAMES.append('molcode')
           self.CALLABLES = []
           if custom_properties is not None:
               for i, func in enumerate(custom_properties):
                   if not callable(func):
                       raise TypeError('custom_properties should be list of'
                                       ' callables, got %s instead' % type(func))
                   name = getattr(func, '__name__', '')
                   if name == '':
                       name = 'func%s' % i
                   self.CALLABLES.append(func)
                   self.FEATURE_NAMES.append(name)
           if smarts_properties is None:
               # SMARTS definition for other properties
               self.SMARTS = [
                   '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                   '[a]',
                   '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                   '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                   '[r]'
               ]
               smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                                'ring']
           elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
               raise TypeError('smarts_properties must be a list')
           else:
               self.SMARTS = smarts_properties

           if smarts_labels is not None:
               if len(smarts_labels) != len(self.SMARTS):
                   raise ValueError('Incorrect number of SMARTS labels: %s'
                                    ' instead of %s'
                                    % (len(smarts_labels), len(self.SMARTS)))
           else:
               smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

           # Compile patterns
           self.compile_smarts()
           self.FEATURE_NAMES += smarts_labels

       def compile_smarts(self):
           self.__PATTERNS = []
           for smarts in self.SMARTS:
               self.__PATTERNS.append(openbabel.pybel.Smarts(smarts))

       def encode_num(self, atomic_num):
           """
           Encode atom type with a binary vector. If atom type is not included in
           the `atom_classes`, its encoding is an all-zeros vector.
           Parameters
           ----------
           atomic_num: int
               Atomic number
           Returns
           -------
           encoding: np.ndarray
               Binary vector encoding atom type (one-hot or null).
           """

           if not isinstance(atomic_num, int):
               raise TypeError('Atomic number must be int, %s was given'
                               % type(atomic_num))

           encoding = np.zeros(self.NUM_ATOM_CLASSES)
           try:
               encoding[self.ATOM_CODES[atomic_num]] = 1.0
           except:
               pass
           return encoding

       def find_smarts(self, molecule):
           """
           Find atoms that match SMARTS patterns.
           Parameters
           ----------
           molecule: openbabel.pybel.Molecule
           Returns
           -------
           features: np.ndarray
               NxM binary array, where N is the number of atoms in the `molecule`
               and M is the number of patterns. `features[i, j]` == 1.0 if i'th
               atom has j'th property
           """

           if not isinstance(molecule, openbabel.pybel.Molecule):
               raise TypeError('molecule must be pybel.Molecule object, %s was given'
                               % type(molecule))

           features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))

           for (pattern_id, pattern) in enumerate(self.__PATTERNS):
               atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                          dtype=int) - 1
               features[atoms_with_prop, pattern_id] = 1.0
           return features

       def get_features(self, molecule, molcode=None):
           """
           Get coordinates and features for all heavy atoms in the molecule.
           Parameters
           ----------
           molecule: pybel.Molecule
           molcode: float, optional
               Molecule type. You can use it to encode whether an atom belongs to
               the ligand (1.0) or to the protein (-1.0) etc.
           Returns
           -------
           coords: np.ndarray, shape = (N, 3)
               Coordinates of all heavy atoms in the `molecule`.
           features: np.ndarray, shape = (N, F)
               Features of all heavy atoms in the `molecule`: atom type
               (one-hot encoding), pybel.Atom attributes, type of a molecule
               (e.g protein/ligand distinction), and other properties defined with
               SMARTS patterns
           """

           if not isinstance(molecule, openbabel.pybel.Molecule):
               raise TypeError('molecule must be pybel.Molecule object,'
                               ' %s was given' % type(molecule))
           if molcode is None:
               if self.save_molecule_codes is True:
                   raise ValueError('save_molecule_codes is set to True,'
                                    ' you must specify code for the molecule')
           elif not isinstance(molcode, (float, int)):
               raise TypeError('motlype must be float, %s was given'
                               % type(molcode))

           coords = []
           features = []
           heavy_atoms = []

           for i, atom in enumerate(molecule):
               # ignore hydrogens and dummy atoms (they have atomicnum set to 0)
               if atom.atomicnum > 1:
                   heavy_atoms.append(i)
                   coords.append(atom.coords)

                   features.append(np.concatenate((
                       self.encode_num(atom.atomicnum),
                       [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                       [func(atom) for func in self.CALLABLES],
                   )))

           coords = np.array(coords, dtype=np.float32)
           features = np.array(features, dtype=np.float32)
           if self.save_molecule_codes:
               features = np.hstack((features,
                                     molcode * np.ones((len(features), 1))))
           features = np.hstack([features,
                                 self.find_smarts(molecule)[heavy_atoms]])

           if np.isnan(features).any():
               raise RuntimeError('Got NaN when calculating features')

           return coords, features

       def to_pickle(self, fname='featurizer.pkl'):
           """Save featurizer in a given file. Featurizer can be restored with
           `from_pickle` method.
           Parameters
           ----------
           fname: str, optional
              Path to file in which featurizer will be saved
           """

           # patterns can't be pickled, we need to temporarily remove them
           patterns = self.__PATTERNS[:]
           del self.__PATTERNS
           try:
               with open(fname, 'wb') as f:
                   pickle.dump(self, f)
           finally:
               self.__PATTERNS = patterns[:]

       @staticmethod
       def from_pickle(fname):
           """
           Load pickled featurizer from a given file
           Parameters
           ----------
           fname: str, optional
              Path to file with saved featurizer
           Returns
           -------
           featurizer: Featurizer object
              Loaded featurizer
           """
           with open(fname, 'rb') as f:
               featurizer = pickle.load(f)
           featurizer.compile_smarts()
           return featurizer

   def rotation_matrix(axis, theta):
       """Counterclockwise rotation about a given axis by theta radians"""

       if not isinstance(axis, (np.ndarray, list, tuple)):
           raise TypeError('axis must be an array of floats of shape (3,)')
       try:
           axis = np.asarray(axis, dtype=np.float)
       except ValueError:
           raise ValueError('axis must be an array of floats of shape (3,)')

       if axis.shape != (3,):
           raise ValueError('axis must be an array of floats of shape (3,)')

       if not isinstance(theta, (float, int)):
           raise TypeError('theta must be a float')

       axis = axis / sqrt(np.dot(axis, axis))
       a = cos(theta / 2.0)
       b, c, d = -axis * sin(theta / 2.0)
       aa, bb, cc, dd = a * a, b * b, c * c, d * d
       bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
       return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


   # Create matrices for all possible 90* rotations of a box
   ROTATIONS = [rotation_matrix([1, 1, 1], 0)]

   # about X, Y and Z - 9 rotations
   for a1 in range(3):
       for t in range(1, 4):
           axis = np.zeros(3)
           axis[a1] = 1
           theta = t * pi / 2.0
           ROTATIONS.append(rotation_matrix(axis, theta))

   # about each face diagonal - 6 rotations
   for (a1, a2) in combinations(range(3), 2):
       axis = np.zeros(3)
       axis[[a1, a2]] = 1.0
       theta = pi
       ROTATIONS.append(rotation_matrix(axis, theta))
       axis[a2] = -1.0
       ROTATIONS.append(rotation_matrix(axis, theta))

   # about each space diagonal - 8 rotations
   for t in [1, 2]:
       theta = t * 2 * pi / 3
       axis = np.ones(3)
       ROTATIONS.append(rotation_matrix(axis, theta))
       for a1 in range(3):
           axis = np.ones(3)
           axis[a1] = -1
           ROTATIONS.append(rotation_matrix(axis, theta))

   def rotate(coords, rotation):
       """
       Rotate coordinates by a given rotation
       Parameters
       ----------
       coords: array-like, shape (N, 3)
           Arrays with coordinates and features for each atoms.
       rotation: int or array-like, shape (3, 3)
           Rotation to perform. You can either select predefined rotation by
           giving its index or specify rotation matrix.
       Returns
       -------
       coords: np.ndarray, shape = (N, 3)
           Rotated coordinates.
       """

       global ROTATIONS

       if not isinstance(coords, (np.ndarray, list, tuple)):
           raise TypeError('coords must be an array of floats of shape (N, 3)')
       try:
           coords = np.asarray(coords, dtype=np.float)
       except ValueError:
           raise ValueError('coords must be an array of floats of shape (N, 3)')
       shape = coords.shape
       if len(shape) != 2 or shape[1] != 3:
           raise ValueError('coords must be an array of floats of shape (N, 3)')

       if isinstance(rotation, int):
           if rotation >= 0 and rotation < len(ROTATIONS):
               return np.dot(coords, ROTATIONS[rotation])
           else:
               raise ValueError('Invalid rotation number %s!' % rotation)
       elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
           return np.dot(coords, rotation)

       else:
           raise ValueError('Invalid rotation %s!' % rotation)

   def make_grid(coords, features, grid_resolution=1.0, max_dist=10.0):
       """
       Convert atom coordinates and features represented as 2D arrays into a
       fixed-sized 3D box.
       Parameters
       ----------
       coords, features: array-likes, shape (N, 3) and (N, F)
           Arrays with coordinates and features for each atoms.
       grid_resolution: float, optional
           Resolution of a grid (in Angstroms).
       max_dist: float, optional
           Maximum distance between atom and box center. Resulting box has size of
           2*`max_dist`+1 Angstroms and atoms that are too far away are not
           included.
       Returns
       -------
       coords: np.ndarray, shape = (M, M, M, F)
           4D array with atom properties distributed in 3D space. M is equal to
           2 * `max_dist` / `grid_resolution` + 1
       """

       try:
           coords = np.asarray(coords, dtype=np.float)
       except ValueError:
           raise ValueError('coords must be an array of floats of shape (N, 3)')
       c_shape = coords.shape
       if len(c_shape) != 2 or c_shape[1] != 3:
           raise ValueError('coords must be an array of floats of shape (N, 3)')

       N = len(coords)
       try:
           features = np.asarray(features, dtype=np.float)
       except ValueError:
           raise ValueError('features must be an array of floats of shape (N, F)')
       f_shape = features.shape
       if len(f_shape) != 2 or f_shape[0] != N:
           raise ValueError('features must be an array of floats of shape (N, F)')

       if not isinstance(grid_resolution, (float, int)):
           raise TypeError('grid_resolution must be float')
       if grid_resolution <= 0:
           raise ValueError('grid_resolution must be positive')

       if not isinstance(max_dist, (float, int)):
           raise TypeError('max_dist must be float')
       if max_dist <= 0:
           raise ValueError('max_dist must be positive')

       num_features = f_shape[1]
       max_dist = float(max_dist)
       grid_resolution = float(grid_resolution)

       box_size = ceil(2 * max_dist / grid_resolution + 1)

       # move all atoms to the nearest grid point
       grid_coords = (coords + max_dist) / grid_resolution
       grid_coords = grid_coords.round().astype(int)

       # remove atoms outside the box
       in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
       grid = np.zeros((1, box_size, box_size, box_size, num_features),
                       dtype=np.float32)
       for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
           grid[0, x, y, z] += f
       return grid

   # necessary import statement
   import xml.etree.ElementTree as ET

   # define function to select pocket mol2 files with atoms that have charges greater than +- 2 (unfeasible)
   def get_charge(molecule):
    for i, atom in enumerate(molecule):
        if atom.atomicnum > 1:
            if (abs(atom.__getattribute__('partialcharge'))>=2): # this charge cutoff can be varied
              return 'bad_complex'
            else: 
              return 'no_error'  

   # define function to extract features from the binding pocket mol2 file and detect if it contains atoms with charges greater than +- 2 (unfeasible) 
   def __get_pocket():
     for pfile in pocket_files:
         try:
             pocket = next(pybel.readfile('mol2', pfile))
         except:
             raise IOError('Cannot read %s file' % pfile)
         if(get_charge(pocket)==('bad_complex')):
          bad_complexes.append((os.path.splitext(os.path.split(pfile)[1])[0]).split('_')[0]) 
         pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
         pocket_vdw = parse_mol_vdw(mol=pocket, element_dict=element_dict)
         yield (pocket_coords, pocket_features, pocket_vdw)

   # define function to extract information from elements.xml file
   def parse_element_description(desc_file):
     element_info_dict = {}
     element_info_xml = ET.parse(desc_file)
     for element in element_info_xml.getiterator():
         if "comment" in element.attrib.keys():
             continue
         else:
             element_info_dict[int(element.attrib["number"])] = element.attrib

     return element_info_dict

   # define function to create a list of van der Waals radii for a molecule
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


   # read in data from elements.xml file
   element_dict = parse_element_description("ML_pba/elements.xml")

   # read in cleaned affinity data csv file
   affinities = pd.read_csv(affinity_data_path)

   # convert pdb id's to numpy array
   pdbids_cleaned = affinities['pdbid'].to_numpy()

   # define empty lists to contain pocket and ligand files
   pocket_files = []
   ligand_files = []
   
   # these are the PDBs for which Chimera failed to calculate charges and that failed hdf5 conversion (next step)
   bad_complexes = ['3ary', '4bps', '4mdq', '2iw4'] 

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

   # create a new hdf file to store all of the data
   with h5py.File(output_total_hdf, 'w') as f:

      pocket_generator = __get_pocket()

      for lfile in ligand_files:
          # use pdbid as dataset name
          name = os.path.splitext(os.path.split(lfile)[1])[0]
          pdbid = name.split('_')[0]

          # read ligand file using pybel
          try:
              ligand = next(pybel.readfile('mol2', lfile))
          except:
              raise IOError('Cannot read %s file' % lfile)

          # extract features from pocket and check for unrealistic charges
          pocket_coords, pocket_features, pocket_vdw = next(pocket_generator)
            
          # extract features from ligand and check for unrealistic charges
          ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
          ligand_vdw = parse_mol_vdw(mol=ligand, element_dict=element_dict)
          if(get_charge(ligand)=='bad_complex'):
             if pdbid not in bad_complexes:
                 bad_complexes.append(pdbid)
            
          # if the current ligand file is part of a bad complex, do not copy to the cleaned hdf file
          if pdbid in bad_complexes:
               continue
          
          # center the ligand and pocket coordinates
          centroid = ligand_coords.mean(axis=0)
          ligand_coords -= centroid
          pocket_coords -= centroid

          # assemble the features into one large numpy array: rows are heavy atoms, columns are coordinates and features
          data = np.concatenate(
              (np.concatenate((ligand_coords, pocket_coords)),
               np.concatenate((ligand_features, pocket_features))),
              axis=1,
          )
          # concatenate van der Waals radii into one numpy array
          vdw_radii = np.concatenate((ligand_vdw, pocket_vdw))

          # create a new dataset for this complex in the hdf file
          dataset = f.create_dataset(pdbid, data=data, shape=data.shape,
                                     dtype='float32', compression='lzf')

          # add the affinity and van der Waals radii as attributes for this dataset 
          dataset.attrs['affinity'] = affinities.loc[pdbid]
          assert len(vdw_radii) == data.shape[0]
          dataset.attrs["van_der_waals"] = vdw_radii

   # save all good pdbids into a csv file for future use
   with open(affinity_data_path, 'rt') as inp, open('affinity_data_cleaned_charge_cutoff_2.csv', 'w') as out:
      writer = csv.writer(out)
      for row in csv.reader(inp):
          if not row[0] in bad_complexes:
              writer.writerow(row)
            
  # read cleaned affinity data into a pandas DataFrame
  affinity_data_cleaned = pd.read_csv('affinity_data_cleaned_charge_cutoff_2.csv')
  
  # create a column for the percentile of affinities
  affinity_data_cleaned['Percentile']= pd.qcut(affinity_data_cleaned['-logKd/Ki'],
                             q = 100, labels = False)
                             
  # create empty arrays to store non-training pdbids
  test_pdbids = []
  val_pdbids = []
  
  # select test data (10% of total) and validation data (2% of total) that have even distributions of
  # affinity percentiles compared to training data
  for i in range(0, 100):
    temp = affinity_data_cleaned[affinity_data_cleaned['Percentile'] == i]
    num_vals = len(temp)
    test_rows = temp.sample(int(num_vals/10))
    test_pdbids = np.hstack((test_pdbids, (test_rows['pdbid']).to_numpy()))
    for pdbid in (test_rows['pdbid']).to_numpy():
      temp = temp[temp.pdbid != pdbid]
    val_rows = temp.sample(int(num_vals/50))
    val_pdbids = np.hstack((val_pdbids, (val_rows['pdbid']).to_numpy()))

  # populate the test and validation hdf files by transferring those datasets from the total file
  with h5py.File(output_test_hdf, 'w') as g, \
   h5py.File(output_val_hdf, 'w') as h:
  with h5py.File(output_total_hdf, 'r') as f:
      for pdbid in val_pdbids:
          ds = h.create_dataset(pdbid, data=f[pdbid], compression = 'lzf')
          ds.attrs['affinity'] = f[pdbid].attrs['affinity']
          ds.attrs["van_der_waals"] = f[pdbid].attrs["van_der_waals"]
      for pdbid in test_pdbids:
          ds = g.create_dataset(pdbid, data=f[pdbid], compression = 'lzf')
          ds.attrs['affinity'] = f[pdbid].attrs['affinity']
          ds.attrs["van_der_waals"] = f[pdbid].attrs["van_der_waals"]
            
  # populate the train hdf file by transferring all other datasets from the total file
  holdouts = np.hstack((val_pdbids,test_pdbids))
  with h5py.File(output_train_hdf, 'w') as g:
    with h5py.File(output_total_hdf, 'r') as f:
      for pdbid in affinity_data_cleaned['pdbid'].to_numpy():
        if pdbid not in holdouts:
          ds = g.create_dataset(pdbid, data=f[pdbid], compression = 'lzf')
          ds.attrs['affinity'] = f[pdbid].attrs['affinity']
          ds.attrs["van_der_waals"] = f[pdbid].attrs["van_der_waals"]
