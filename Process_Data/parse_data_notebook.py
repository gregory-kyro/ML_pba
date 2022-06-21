"" create cleaned pdb affinity dataset"""
from parse_PDBbind_data import create_dataset
create_dataset('path/to/dataset.csv', 'path/to/general/set/pdbs/%s', 'path/to/refined/set/pdbs/%s', plot=True)
