from scipy.interpolate import interp1d
import numpy as np
from scipy.stats import norm

def transit_likelihoods(leg, transits):
    leg_t = leg.time.values
    leg_pos = leg[['x', 'y']].values
    trans_liks = {}
    for key, transit in transits.items():
        if len(transit) < 2:
            trans_liks[key] = np.nan
            continue
        transit_t = transit.time.values
        transit_pos = transit[['x', 'y']].values
        transit_pos = interp1d(transit_t, transit_pos, axis=0, bounds_error=False)(leg_t)
        diffs = leg_pos - transit_pos
        dists = np.linalg.norm(diffs, axis=1)
        # TODO MEGAHACK: Using negative mean distances as "likelihoods".
        # so the API is for maximizing instead of minimizing
        trans_liks[key] = -np.median(dists)
    
    return trans_liks
