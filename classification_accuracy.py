import psycopg2
import numpy as np
import pandas as pd
#from dragfilter import filter_trajectory
from dragimm import filter_trajectory, filters
import matplotlib.pyplot as plt

atype_mapping = {
    'still': 'still',
    'running': 'walking',
    'on_foot': 'walking',
    'walking': 'walking',
    'on_bicycle': 'cycling',
    'in_vehicle': 'driving',
        }

atype_names = {i: n for i, n in enumerate(filters)}

#conn = psycopg2.connect(dbname="mocaf", host='localhost', port=54320, password="bernie2020")

conn = psycopg2.connect("postgresql://mocaf:RicGuckigixniunckeshCyijAjmyukCyHeKnorf1@localhost:54321/mocaf")

uid = "14c74f41-e744-4b52-9713-ad26dcdded35" # Juha
#uid = "073578d3-b0e1-4917-a7e1-5b00ba847181" # Juha 2.0
#uid = "e7d47921-0b69-45f4-a665-6bd8f7e8139e" # Somebody?
query = f"""select
        extract(epoch from time) as time,
        time as sqlisshit,
        uuid as uid,
        ST_X(ST_Transform(loc, 3067)) as x,
        ST_Y(ST_Transform(loc, 3067)) as y,
        speed,
        loc_error as location_std,
        atype,
        aconf,
        manual_atype

        FROM trips_ingest_location
        --WHERE uid = (select uid from trips_ingest_location where manual_atype NOT NULL)
        WHERE uuid = '{uid}'
        --AND time <= '2021-02-18'
        --AND time >= '2021-02-17'
        ORDER BY uuid,time DESC
        """

#traj = pd.read_sql_query(query, conn)
traj = pd.read_sql_query(query, conn).iloc[::-1].reset_index()

shittime = traj['sqlisshit'].iloc[0]
#print(shittime.tz_convert('Europe/Helsinki'))
traj['atype'] = traj.atype.map(lambda x: atype_mapping.get(x, None))
traj['manual_atype'] = traj.manual_atype.map(lambda x: atype_mapping.get(x, None))
traj['aconf'] /= 100
#print(traj['aconf'])
# The data seems to give 100% confidences quite often. We've caught
# 'em lying too many times, make it half
#traj['aconf'][traj.aconf == 1] /= 2
traj['aconf'] *= 1/2
#traj['aconf'] = 1/4


ms, Ss, state_probs, states, _ = filter_trajectory((r for i, r in traj.iterrows()))
#traj['pred_atype'] = states
traj['ml_atype'] = states
traj['ml_atype'] = traj.ml_atype.map(lambda x: atype_names.get(x, None))
traj['map_atype'] = np.argmax(state_probs, axis=1)
traj['map_atype'] = traj.map_atype.map(lambda x: atype_names.get(x, None))

mtraj = traj[traj.manual_atype.notnull()]
n = len(mtraj)

from sklearn.metrics import cohen_kappa_score

print("ML", cohen_kappa_score(mtraj.ml_atype, mtraj.manual_atype))
print("Map", cohen_kappa_score(mtraj.map_atype, mtraj.manual_atype))
print("Raw", cohen_kappa_score(mtraj.atype, mtraj.manual_atype))

#print("Map", (mtraj.map_atype == mtraj.manual_atype).sum()/n)
#print("ML", (mtraj.ml_atype == mtraj.manual_atype).sum()/n)
#print("Raw", (mtraj.atype == mtraj.manual_atype).sum()/n)

