import tkinter as tk
from tkinter import filedialog
import h5py
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import tdt_data
import ori_led
import phy_depth_mapping

model, primary_chan_dict, P, w_p_sig, max_rate_on, max_rate_off, remap_depth = ({} for i in range(7))

# run this portion for all blocks desired
tk.Tk().withdraw()
date = tdt_data.get_date()
block = tdt_data.get_block()
probe = 'a1x32'
channel_group = int(input('What is the channel group? '))
analyzer_path = tk.filedialog.askopenfilename(filetypes=[("analyzer files", "*.analyzer")])

dbg = date + block + str(channel_group)

model[dbg], primary_chan_dict[dbg], P[dbg], w_p_sig[dbg], max_rate_on[dbg], \
    max_rate_off[dbg] = ori_led.ori_led(date, block, channel_group, analyzer_path)

remap_depth[dbg] = phy_depth_mapping.get_depth_info(date, probe, 0)
print('Done with Part 1!')

# ---------------------------------------------------------------------------------------------------------------------
# run this portion of the code at end once all blocks are stored
tuned_list = {}
difference_list = {}
max_rate_on_list = []
max_rate_off_list = []

fig = plt.figure()
ax = fig.add_subplot(111)

depth_counter = np.zeros([3])
unit_counter = 0

for dbg in P:
    tuned_list[dbg] = {}
    difference_list[dbg] = {}
    for cluster in P[dbg]:

        if P[dbg][cluster] < 0.05 \
                and (model[dbg].cluster_groups[cluster] ==
                         'good' or model[dbg].cluster_groups[cluster] == 'mua'):
            unit_counter += 1
            tuned_list[dbg][cluster] = P[dbg][cluster]
            if remap_depth[dbg][primary_chan_dict[dbg][cluster]] == 'infragranular':
                ax.plot(max_rate_off[dbg][cluster], max_rate_on[dbg][cluster], 'r.', markersize=10)
                depth_counter[0] += 1
            elif remap_depth[dbg][primary_chan_dict[dbg][cluster]] == 'granular':
                ax.plot(max_rate_off[dbg][cluster], max_rate_on[dbg][cluster], 'b.', markersize=10)
                depth_counter[1] += 1
            else:
                ax.plot(max_rate_off[dbg][cluster], max_rate_on[dbg][cluster], 'g.', markersize=10)
                depth_counter[2] += 1
            max_rate_on_list.append(max_rate_on[dbg][cluster])
            max_rate_off_list.append(max_rate_off[dbg][cluster])
            if len(w_p_sig[dbg][cluster]) > 0:
                difference_list[dbg][cluster] = w_p_sig[dbg][cluster]
    # print('There are %d tuned clusters.' % len(tuned_list[dbg]))
    # print('There are %d clusters with a significant difference in LED states.' % len(difference_list))

max_rate_on_list = np.asarray(max_rate_on_list)
max_rate_off_list = np.asarray(max_rate_off_list)

w, p = sp.stats.wilcoxon(max_rate_on_list, max_rate_off_list)
print('The p-value of a Wilcoxon signed-rank test on the LED on vs. LED off states is %f.' % p)
print('There were %d total units.' % unit_counter)
print('There were %d infragranular units, %d granular units, and %d supragranular units.' %
      (depth_counter[0], depth_counter[1], depth_counter[2]))

fit = np.polyfit(max_rate_off_list, max_rate_on_list, 1)
plt.plot(max_rate_off_list, fit[0] * max_rate_off_list + fit[1], '-', label='Line of Best Fit')

unity = np.linspace(0, 100, 2)
ax.plot(unity, unity, '--', label='x=y', scalex=False, scaley=False)
ax.set_xlabel('Max Firing Rate (LED off)')
ax.set_ylabel('Max Firing Rate (LED on)')
plt.suptitle('Comparison of Maximum Firing Rates Based on LED State')
ax.set_title('Wilcoxon Signed-Rank Test p-value: %f' % p)

fig.savefig("ori.png")
fig.savefig("ori.svg", format='svg')

plt.legend()
plt.show()
