import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from static_delay_interaction import DBN_static_delay as DBNstat

df = pd.read_csv('data/dataset_v2_1.csv')
data = np.loadtxt("data/buffer_block_zurich_chur_2019_v2.csv",dtype=str, delimiter=",")
buffers = df["buffer"].to_numpy()
b_sub = np.arange(0)
buffers = b_sub[::10]
taus = [np.linspace(0,14,25)[11]]

def run_sim(i, tau, b, extra_metrics=False, active_agents=False):
    res2 = []
    res3 = []
    # for i in range(iterations):
    dbn = DBNstat(from_df=df, system="dict", recovery_rate=0)
    dbn.add_delay(tau=tau, add=0)
    dbn.process_delays_dict(buffer_addition=b)
    res2.append(dbn.spatial_event_series)
    res3.append(dbn.spatial_event_series_prim)
    return res2, res3

from joblib import Parallel, delayed
from tqdm import tqdm

iterations = 1

results_list = Parallel(n_jobs=10, verbose=True * len(buffers))(delayed(run_sim)(0, x,
    0) for x in taus)

primlist = []
seconlist = []
for result in results_list:
    primlist.append(result[1])
    seconlist.append(result[0])
np.save("simdata/primary_delays.npy", primlist)
np.save("simdata/secondary_delays.npy", seconlist)




