import numpy as np
import pandas as pd
#from dragfilter import filter_trajectory
from dragimm import filter_trajectory, filters
import matplotlib.pyplot as plt
import time
traj = pd.read_csv("testdata.csv")

start = time.time()
ms, Ss, state_probs, states = filter_trajectory((r for i, r in traj.iterrows()))
dur = time.time() - start
states = np.array(states)
print(np.histogram(states))
print(f"Filter time {dur:.3f}s. {len(ms)/dur:.1f} samples/s")
exit()

