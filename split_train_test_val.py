
def split_hdf_file(cleaned_affinity_csv, input_hdf, output_val_hdf, output_train_hdf, output_test_hdf):

    """
  input:
  1) path/to/cleaned/affinity/data.csv
  2) path/to/input/hdf/file.hdf
  3) path/to/output/validation/hdf/file.hdf
  4) path/to/output/training/hdf/file.hdf
  5) path/to/output/testing/hdf/file.hdf
  
  output:
  1) returns a csv file containing only the PDB id's that will be used, saved as:
     'affinity_data_cleaned_charge_cutoff_2.csv'
  """
  
  
  #Read cleaned affinity data into DataFrame
  affinity_data_cleaned = pd.read_csv('affinity_data_cleaned_charge_cutoff_2.csv')
  
  #Create a column for the percentile of affinities
  affinity_data_cleaned['Percentile']= pd.qcut(affinity_data_cleaned['-logKd/Ki'],
                             q = 100, labels = False)
                             
  #Create empty arrays to store non-training pdbids
  test_pdbids = []
  val_pdbids = []
  
  #Select test data (10% of total) and validation data (2% of total) that have an even distribution of affinity percentiles
  for i in range(0, 100):
    temp = affinity_data_cleaned[affinity_data_cleaned['Percentile'] == i]
    num_vals = len(temp)
    test_rows = temp.sample(int(num_vals/10))
    test_pdbids = np.hstack((test_pdbids, (test_rows['pdbid']).to_numpy()))
    for pdbid in (test_rows['pdbid']).to_numpy():
      temp = temp[temp.pdbid != pdbid]
    val_rows = temp.sample(int(num_vals/50))
    val_pdbids = np.hstack((val_pdbids, (val_rows['pdbid']).to_numpy()))

  #Populate test and validation hdf files by transferring those datasets from the input file
  with h5py.File(output_test_hdf, 'w') as g, \
   h5py.File(output_val_hdf, 'w') as h:
  with h5py.File(input_hdf, 'r') as f:
      for pdbid in val_pdbids:
          ds = h.create_dataset(pdbid, data=f[pdbid], compression = 'lzf')
          ds.attrs['affinity'] = f[pdbid].attrs['affinity']
          ds.attrs["van_der_waals"] = f[pdbid].attrs["van_der_waals"]
      for pdbid in test_pdbids:
          ds = g.create_dataset(pdbid, data=f[pdbid], compression = 'lzf')
          ds.attrs['affinity'] = f[pdbid].attrs['affinity']
          ds.attrs["van_der_waals"] = f[pdbid].attrs["van_der_waals"]
            
  #Populate train hdf file by transferring all other datasets from the input file
  holdouts = np.hstack((val_pdbids,test_pdbids))
  with h5py.File(output_train_hdf, 'w') as g:
    with h5py.File(input_hdf, 'r') as f:
      for pdbid in affinity_data_cleaned['pdbid'].to_numpy():
        if pdbid not in holdouts:
          ds = g.create_dataset(pdbid, data=f[pdbid], compression = 'lzf')
          ds.attrs['affinity'] = f[pdbid].attrs['affinity']
          ds.attrs["van_der_waals"] = f[pdbid].attrs["van_der_waals"]


