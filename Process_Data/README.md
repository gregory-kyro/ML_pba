**Procedure for Processing and Preparing Data**

**1)** Read in PDBbind datasets and create a csv file to be used to produce hdf5 files (data for models)
  - parse_PDBbind_data.py
  - run this code in notebook to perform step 1:
        
        from parse_PDBbind_data import create_dataset
        
        create_dataset('path/to/dataset.csv', 'path/to/general/set/pdbs/%s', 'path/to/refined/set/pdbs/%s', plot=True)
        
        
**2)** Add hydrogens to pocket PDB files and convert to mol2 files type using Chimera 1.16, remove TIP3P atoms from mol2 files
  - save add_H_and_mol2_chimera.py and run_chimera_remove_tip3p.sh, altering file paths as appropriate
  - run the following in Mac terminal: sh path/to/run_chimera_remove_tip3p.sh
  - if the process crashes, remove the problematic pdb file and run the same line of code again
  
**3)** Compute charges for mol2 files externally with Atomic Charge Calculator II (ACC2)
  - ACC2: <https://acc2.ncbr.muni.cz/>
   
**4)** Identify non-problematic complexes, and create train, test, and validation hdf files
  - convert_to_hdf.py
  
**5)** Create hdf files containing voxelized train, test, and validation data for use in the 3dcnn
  - create_voxelized_hdf.py
