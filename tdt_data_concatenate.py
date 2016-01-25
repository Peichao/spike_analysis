import os
import time
import pandas as pd
import tdt_data
import phy_analysis

# get tank path
probe = input('Which probe file? ')
tank_path = tdt_data.get_tank_path()
directory = os.path.abspath(os.path.join(tank_path, os.pardir))

blocks = [s for s in os.listdir(tank_path) if os.path.isdir(os.path.join(tank_path, s))]
blocks.sort(key=lambda s: os.path.getmtime(os.path.join(tank_path, s)), reverse=False)

raw_pandas = {}

for block in blocks:
    print(time.strftime("%H:%M:%S"), block)
    if not os.path.exists(directory + "/%s" % block):
        os.makedirs(directory + "/%s" % block)
    save_directory = directory + '/' + block
    raw_pandas[block] = pd.DataFrame(tdt_data.save_matlab_dat(tank_path, block), columns=range(0, 32))
    phy_analysis.create_prms(save_directory + '/' + block + '.prm', block, probe)
    phy_analysis.spikesort(save_directory + '/' + block + '.prm')

# create binary .dat file with all trials concatenated and parameters file
raw_pandas_concat = pd.concat(raw_pandas, ignore_index=True, keys=blocks)
raw_array_concat = pd.DataFrame.as_matrix(raw_pandas_concat)
raw_array_concat.tofile(directory + '/' + 'raw_array_concat.dat')
phy_analysis.create_prms(directory, 'raw_array_concat', probe)