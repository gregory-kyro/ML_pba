

def create_voxelized_hdf (input_train_hdf_path, input_val_hdf_path, input_test_hdf_path, output_train_hdf_path, output_val_hdf_path, output_test_hdf_path, output_csv_path):
  
    """
    input:
    1) path/to/input/training/hdf/file.hdf
    2) path/to/input/validation/hdf/file.hdf
    3) path/to/input/testing/hdf/file.hdf
    4) path/to/output/training/hdf/file.hdf
    5) path/to/output/validation/hdf/file.hdf
    6) path/to/output/testing/hdf/file.hdf
    7) path/to/output/csv/file.csv

    output:
    1) Creates 3 hdf files containing voxelized training, testing, and validation data. Also creates a csv file containing more general information about each complex.
    """

    #Necessary import statements
    import os
    import sys
    import shutil
    import csv
    import h5py
    import numpy as np
    import scipy as sp
    import scipy.ndimage

    #Function to rotate input data
    def rotate_3D(input_data):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        #rotated_data = np.zeros(input_data.shape, dtype=np.float32)
        rotated_data = np.dot(input_data, rotation_matrix)
        return rotated_data

    #Function to get boundaries of a coordinate array
    def get_3D_bound(xyz_array):
        xmin = min(xyz_array[:, 0])
        ymin = min(xyz_array[:, 1])
        zmin = min(xyz_array[:, 2])
        xmax = max(xyz_array[:, 0])
        ymax = max(xyz_array[:, 1])
        zmax = max(xyz_array[:, 2])
        return xmin, ymin, zmin, xmax, ymax, zmax

    #Function to voxelize
    def get_3D_all2(xyz, feat, vol_dim, relative_size=True, size_angstrom=48, atom_radii=None, atom_radius=1, sigma=0):

        # get 3d bounding box
        xmin, ymin, zmin, xmax, ymax, zmax = get_3D_bound(xyz)

        # initialize volume
        vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)

        if relative_size:
            # voxel size (assume voxel size is the same in all axis)
            vox_size = float(zmax - zmin) / float(vol_dim[0])
        else:
            vox_size = float(size_angstrom) / float(vol_dim[0])
            xmid = (xmin + xmax) / 2.0
            ymid = (ymin + ymax) / 2.0
            zmid = (zmin + zmax) / 2.0
            xmin = xmid - (size_angstrom / 2)
            ymin = ymid - (size_angstrom / 2)
            zmin = zmid - (size_angstrom / 2)
            xmax = xmid + (size_angstrom / 2)
            ymax = ymid + (size_angstrom / 2)
            zmax = zmid + (size_angstrom / 2)
            vox_size2 = float(size_angstrom) / float(vol_dim[0])
            #print(vox_size, vox_size2)

        # assign each atom to voxels
        for ind in range(xyz.shape[0]):
            x = xyz[ind, 0]
            y = xyz[ind, 1]
            z = xyz[ind, 2]
            if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
                continue

            # compute van der Waals radius and atomic density, use 1 if not available
            if not atom_radii is None:
                vdw_radius = atom_radii[ind]
                atom_radius = 1 + vdw_radius * vox_size

            cx = (x - xmin) / (xmax - xmin) * (vol_dim[2] - 1)
            cy = (y - ymin) / (ymax - ymin) * (vol_dim[1] - 1)
            cz = (z - zmin) / (zmax - zmin) * (vol_dim[0] - 1)

            vx_from = max(0, int(cx - atom_radius))
            vx_to = min(vol_dim[2] - 1, int(cx + atom_radius))
            vy_from = max(0, int(cy - atom_radius))
            vy_to = min(vol_dim[1] - 1, int(cy + atom_radius))
            vz_from = max(0, int(cz - atom_radius))
            vz_to = min(vol_dim[0] - 1, int(cz + atom_radius))

            for vz in range(vz_from, vz_to + 1):
                for vy in range(vy_from, vy_to + 1):
                    for vx in range(vx_from, vx_to + 1):
                            vol_data[vz, vy, vx, :] += feat[ind, :]

        # gaussian filter
        if sigma > 0:
            for i in range(vol_data.shape[-1]):
                vol_data[:,:,:,i] = sp.ndimage.filters.gaussian_filter(vol_data[:,:,:,i], sigma=sigma, truncate=2)

        return vol_data

  # do not change unless the hdf structure is changed
  g_csv_header = ['pdbid', 'file_prefix', 'label', 'train_test_split', 'atom_count', 'xsize', 'ysize', 'zsize', 'p_atom_count1', 'p_atom_count2', 'p_xsize', 'p_ysize', 'p_zsize']

  # open input hdf files
  input_train_hdf = h5py.File(input_train_hdf_path, 'r')
  input_val_hdf = h5py.File(input_val_hdf_path, 'r')
  input_test_hdf = h5py.File(input_test_hdf_path, 'r')

  # create output hdf files
  output_train_hdf = h5py.File(output_train_hdf_path, 'w')
  output_val_hdf = h5py.File(output_val_hdf_path, 'w')
  output_test_hdf = h5py.File(output_test_hdf_path, 'w')

  # create output csv
  output_csv_fp = open(output_csv_path, 'w')
  output_csv = csv.writer(output_csv_fp, delimiter=',')
  output_csv.writerow(g_csv_header)
  
  #organize input and output hdf files
  input_hdfs = [input_train_hdf, input_val_hdf, input_test_hdf]
  output_hdfs = [output_train_hdf, output_val_hdf, output_test_hdf]
  traintest_splits = [0, 1, 2]


  ##Set parameters for voxelization
  g_3D_relative_size = False
  g_3D_size_angstrom = 48
  g_3D_size_dim = 48
  g_3D_atom_radius = 1
  g_3D_atom_radii = False
  g_3D_sigma = 1
  g_3D_dim = [g_3D_size_dim, g_3D_size_dim, g_3D_size_dim, 19]


  if g_3D_relative_size:
      g_3D_size_angstrom = 0



  for input_hdf, output_hdf, split in zip(input_hdfs, output_hdfs, traintest_splits): 
      for pdbid in input_hdf.keys():
        input_data = input_hdf[pdbid]
        input_radii = None
        if g_3D_atom_radii:
            input_radii = input_data.attrs['van_der_waals']
        input_affinity = input_hdf[pdbid].attrs['affinity']

        input_xyz = input_data[:,0:3] 
        input_feat = input_data[:,3:]

        output_3d_data = get_3D_all2(input_xyz, input_feat, g_3D_dim, g_3D_relative_size, g_3D_size_angstrom, input_radii, g_3D_atom_radius, g_3D_sigma)
        print(input_data.shape, 'is converted into ', output_3d_data.shape)


        output_hdf.create_dataset(pdbid, data=output_3d_data, shape=output_3d_data.shape, dtype='float32', compression='lzf')
        output_hdf[pdbid].attrs['affinity'] = input_affinity

        lig_prefix = '%d/%s' % (split, pdbid)
        output_csv.writerow([pdbid, lig_prefix, input_affinity, split, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  output_train_hdf.close()
  output_val_hdf.close()
  output_test_hdf.close()
