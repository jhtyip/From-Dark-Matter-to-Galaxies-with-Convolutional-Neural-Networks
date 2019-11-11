import h5py
import pandas as pd
import numpy as np
import os

e = 0.0001

output_grid_Coordinates = 'subhalos_TNG300-1_samples/subhalos_TNG300-1_flagged_Coordinates_grid.npy'  # subhalo number density field (not useful)
output_grid_StarsMasses = 'subhalos_TNG300-1_samples/subhalos_TNG300-1_flagged_StarsMasses_grid.npy'  # galaxy mass density field
output_grid_StarsMassesNum = 'subhalos_TNG300-1_samples/subhalos_TNG300-1_flagged_StarsMassesNum_grid.npy'  # galaxy number density field
raw_data1 = 'subhalos/SubhaloPos/TNG300-1'
raw_data2 = 'subhalos/SubhaloMassType/TNG300-1'
raw_data3 = 'subhalos/SubhaloFlag/TNG300-1'


def process(postable):
    postable['x_b'] = np.floor(postable['x']/((205000+e)/1024))
    postable['y_b'] = np.floor(postable['y']/((205000+e)/1024))
    postable['z_b'] = np.floor(postable['z']/((205000+e)/1024))

    postable['x_b'] = postable['x_b'].astype(int)
    postable['y_b'] = postable['y_b'].astype(int)
    postable['z_b'] = postable['z_b'].astype(int)

    return postable


if not os.path.exists('subhalos_TNG300-1_samples'):
    os.makedirs('subhalos_TNG300-1_samples')

np.save(output_grid_Coordinates, np.zeros((1024, 1024, 1024)))
np.save(output_grid_StarsMasses, np.zeros((1024, 1024, 1024)))
np.save(output_grid_StarsMassesNum, np.zeros((1024, 1024, 1024)))

for name1 in sorted(os.listdir(raw_data1)):
  for name2 in sorted(os.listdir(raw_data2)):
    for name3 in sorted(os.listdir(raw_data3)):
        filename1 = raw_data1 + '/' + name1
        f1 = h5py.File(filename1, 'r')
        
        filename2 = raw_data2 + '/' + name2
        f2 = h5py.File(filename2, 'r')
        
        filename3 = raw_data3 + '/' + name3
        f3 = h5py.File(filename3, 'r')
        
        Coordinates = np.array(f1['Subhalo']['SubhaloPos'])
        StarsMasses = np.array(f2['Subhalo']['SubhaloMassType'])[:, 4]
        Flag = np.array(f3['Subhalo']['SubhaloFlag'])
    
        print(len(Coordinates))
        print(len(StarsMasses))
        print(len(Flag))
    
        if len(Coordinates) == 0:
            pass
        if len(StarsMasses) == 0:
            pass
        if len(Flag) == 0:
            pass
    
        postable = pd.DataFrame(Coordinates)
        postable.columns = (['x', 'y', 'z'])  # define the column names
        mergetable = process(postable)
    
        coortable = mergetable[['x_b', 'y_b', 'z_b']].values
    
        arr_coordinates = np.zeros((1024, 1024, 1024))
        arr_starsmasses = np.zeros((1024, 1024, 1024))
        arr_starsmassesnum = np.zeros((1024, 1024, 1024))
    
        for i in range(coortable.shape[0]):
            if coortable[i, 0] < 1024 and coortable[i, 1] < 1024 and coortable[i, 2] < 1024:
                arr_coordinates[coortable[i, 0], coortable[i, 1], coortable[i, 2]] += 1
                if StarsMasses[i] > 0 and Flag[i] == 1:
                    arr_starsmasses[coortable[i, 0], coortable[i, 1], coortable[i, 2]] += StarsMasses[i]
                    arr_starsmassesnum[coortable[i, 0], coortable[i, 1], coortable[i, 2]] += 1
            else:
                print(coortable[i])  # strangely, some (very few) particles are out of the boundary
    
        old_arr_coordinates = np.load(output_grid_Coordinates)
        old_arr_starsmasses = np.load(output_grid_StarsMasses)
        old_arr_starsmassesnum = np.load(output_grid_StarsMassesNum)
    
        np.save(output_grid_Coordinates, old_arr_coordinates+arr_coordinates)
        np.save(output_grid_StarsMasses, old_arr_starsmasses+arr_starsmasses)
        np.save(output_grid_StarsMassesNum, old_arr_starsmassesnum+arr_starsmassesnum)
        
        print('finished: '+name1)
