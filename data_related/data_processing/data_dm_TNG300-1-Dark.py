import h5py
import pandas as pd
import numpy as np
import os

e = 0.0001

output_grid_Coordinates = 'dm_TNG300-1-Dark_samples/dm_TNG300-1-Dark_Coordinates_grid.npy'
raw_data = 'dm/TNG300-1-Dark'


def process(postable):
    postable['x_b'] = np.floor(postable['x']/((205000+e)/1024))
    postable['y_b'] = np.floor(postable['y']/((205000+e)/1024))
    postable['z_b'] = np.floor(postable['z']/((205000+e)/1024))

    postable['x_b'] = postable['x_b'].astype(int)
    postable['y_b'] = postable['y_b'].astype(int)
    postable['z_b'] = postable['z_b'].astype(int)

    return postable


if not os.path.exists('dm_TNG300-1-Dark_samples'):
    os.makedirs('dm_TNG300-1-Dark_samples')

np.save(output_grid_Coordinates, np.zeros((1024, 1024, 1024)))

for name in sorted(os.listdir(raw_data)):
    filename = raw_data + '/' + name
    f = h5py.File(filename, 'r')

    Coordinates = np.array(f['PartType1']['Coordinates'])

    if len(Coordinates) == 0:
        pass

    postable = pd.DataFrame(Coordinates)
    postable.columns = (['x', 'y', 'z'])  # define the column names
    mergetable = process(postable)

    coortable = mergetable[['x_b', 'y_b', 'z_b']].values

    arr_coordinates = np.zeros((1024, 1024, 1024))

    for i in range(coortable.shape[0]):
        arr_coordinates[coortable[i, 0], coortable[i, 1], coortable[i, 2]] += 1

    old_arr_coordinates = np.load(output_grid_Coordinates)

    np.save(output_grid_Coordinates, old_arr_coordinates+arr_coordinates)

    print('finished: '+name)
