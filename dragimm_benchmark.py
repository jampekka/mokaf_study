import numpy as np
import pandas as pd
#from dragfilter import filter_trajectory
from dragimm import filter_trajectory, filters
import matplotlib.pyplot as plt

traj = pd.read_csv("testdata.csv")

ms, Ss, state_probs, states = filter_trajectory((r for i, r in traj.iterrows()))
states = np.array(states)
print(np.histogram(states))
exit()
plt.figure("locations")
plt.plot(traj.x, traj.y, '.', color='black', label="Original")

modecolors = ['blue', 'green', 'orange', 'red']
legchanges = np.flatnonzero(np.diff(states)) + 1
legchanges = [0] + list(legchanges) + [len(states)]

for leg_i in range(len(legchanges) - 1):
    s = legchanges[leg_i]
    e = legchanges[leg_i+1]
    mode = states[s]
    slc = slice(s, e)
    plt.plot(ms[slc,0], ms[slc,1], color=modecolors[mode])

#plt.plot(ms[:,0], ms[:,1], color='black', label="Filtered", alpha=0.5)
#plt.plot(ms[:,0], ms[:,1], color='black', label="Filtered", alpha=0.5)
plt.legend()
plt.axis('equal')

plt.figure("xlocs")
filtspeed = np.linalg.norm(ms[:,2:4], axis=1)
s = np.sqrt(Ss[:,0,0])
#plt.plot(traj.time, ms[:,0] + s, color='black', label="Filtered", alpha=0.5)
#plt.plot(traj.time, ms[:,0] - s, color='black', label="Filtered", alpha=0.5)
#plt.plot(traj.time, traj.x, color='red', label="Original", alpha=0.5)

plt.plot(traj.time, np.abs(traj.speed), color='red', label='Original', alpha=0.5)
plt.plot(traj.time, filtspeed, color='black', label='Filtered', alpha=0.5)
plt.legend()
plt.figure("stateprobs")

for name, s in zip(filters.keys(), np.array(state_probs).T):
    plt.plot(traj.time, s, label=name)

plt.twinx()
plt.plot(traj.time, states, color='black', lw=2)
plt.legend()
plt.show()
