import tkinter as tk
from tkinter import filedialog
import h5py
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import tdt_data
import phy_depth_mapping
import sz_led

model, primary_chan_dict, p_perm, si_on_all, si_off_all, depth_info = ({} for i in range(6))

# get input file information
date = str(input('What was the date ("YYYYMMDD")? '))
block = tdt_data.get_block()
probe = str(input('What was the probe type? '))
channel_group = 0
if probe == 'a2x16':
    channel_group = int(input('What is the channel group? '))
plot = str(input('Individual plots (y/n)? '))
tk.Tk().withdraw()
analyzer_path = tk.filedialog.askopenfilename(filetypes=[("analyzer files", "*.analyzer")])

dbg = date + block + str(channel_group)

# get depth info for experiment
depth_info[dbg] = phy_depth_mapping.get_depth_info(date, probe, channel_group)

# analyze all clusters
model[dbg], primary_chan_dict[dbg], p_perm[dbg], si_on_all[dbg], si_off_all[dbg] = \
    sz_led.sz_led(date, block, channel_group, analyzer_path, plot)
print('Done with Part 1!')

# ---------------------------------------------------------------------------------------------------------------------
# run once all blocks are loaded!
tuned_list = {}
si_on_list, si_off_list = ([] for i in range(2))
depth_counter = np.zeros([3])
unit_counter = 0

# define colors for depth mapping
depth_colors = {
    'infragranular': 'r',
    'granular': 'b',
    'supragranular': 'g'
}

fig, ax1 = plt.subplots(1)
for dbg in p_perm:
    tuned_list[dbg] = {}

    for cluster in p_perm[dbg]:

        if p_perm[dbg][cluster] < 0.05 \
                and (model[dbg].cluster_groups[cluster] ==
                         'good' or model[dbg].cluster_groups[cluster] == 'mua'):

            unit_counter += 1
            si_on_list.append(si_on_all[dbg][cluster])
            si_off_list.append(si_off_all[dbg][cluster])
            # ax1.scatter(si_off_all[dbg][cluster], si_on_all[dbg][cluster],
            #             color=depth_colors[depth_info[dbg][primary_chan_dict[dbg][cluster]]])
            if depth_info[dbg][primary_chan_dict[dbg][cluster]] == 'infragranular':
                ax1.plot(si_off_all[dbg][cluster], si_on_all[dbg][cluster], 'r.', markersize=10)
                depth_counter[0] += 1
            elif depth_info[dbg][primary_chan_dict[dbg][cluster]] == 'granular':
                ax1.plot(si_off_all[dbg][cluster], si_on_all[dbg][cluster], 'b.', markersize=10)
                depth_counter[1] += 1
            elif depth_info[dbg][primary_chan_dict[dbg][cluster]] == 'supragranular':
                ax1.plot(si_off_all[dbg][cluster], si_on_all[dbg][cluster], 'g.', markersize=10)
                depth_counter[2] += 1


w, p = sp.stats.wilcoxon(si_on_list, si_off_list)

print('The p-value of a Wilcoxon signed-rank test on the LED on vs. LED off states is %f.' % p)
print('There were %d total units.' % unit_counter)
print('There were %d infragranular units, %d granular units, and %d supragranular units.' %
      (depth_counter[0], depth_counter[1], depth_counter[2]))

# create unity lines
unity = np.arange(-5, 5, 0.01)

fit = np.polyfit(si_off_list, si_on_list, 1)
# ax1.plot(si_off_list, fit[0] * si_off_list + fit[1], '-', label='Line of Best Fit')

ax1.plot(unity, unity, 'k--')
ax1.axis([-1, 5, -1, 5])
ax1.set_xlabel('Suppression index (LED on)')
ax1.set_ylabel('Suppression index (LED off)')
ax1.set_title('Wilcoxon Signed-Rank Test p-value: %f' % p)
plt.suptitle('Comparison of Suppression Index Based on LED State')

fig.savefig("si.png")
fig.savefig("si.eps", format='eps')
fig.show()
