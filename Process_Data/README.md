**Procedure for Processing and Preparing Data**
1) Read in PDBbind datasets and create a csv file to be used to produce hdf5 files (data for models)
  - parse_PDBbind_data.py
  - parse_data.py
2) Add hydrogens to pocket PDB files and convert to mol2 files type using Chimera 1.16, remove TIP3P atoms from mol2 files
  - add_H_and_mol2_chimera.py
  - run_chimera_remove_tip3p.sh
3) Identify non-problematic complexes, and create train, test, and validation hdf files
  - convert_to_hdf.py
4) Create hdf files containing voxelized train, test, and validation data for use in the 3dcnn
  - create_voxelized_hdf.py
