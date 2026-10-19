import dataclasses

import torch
import numpy as np

LOG_VARS = ("tp_log", "tp_6hr_log", "tp_mswep_log", "r_log", "r_6hr_log") # surface vars stored as log(1 + x/eps)


def transform_data(data, short_name, eps=1e-5, direct=True): # transformation follows [Rasp 2020], also used in FourCastNet
    if np.isin(short_name, LOG_VARS):
        if direct:
            if isinstance(data, torch.Tensor):
                return torch.log(1 + data/eps)
            else:
                return np.log(1 + data/eps)
        else:
            if isinstance(data, torch.Tensor):
                return eps*(torch.exp(data) - 1)
            else:
                return eps*(np.exp(data) - 1)
        
    else:
        return data


def inverse_transform_log_vars(batch):
    ''' Return a copy of a prediction Batch with LOG_VARS mapped back to physical units; other vars are untouched '''
    return dataclasses.replace(batch, surf_vars={
        k: transform_data(v, k, direct=False) if k in LOG_VARS else v
        for k, v in batch.surf_vars.items()
    })


def accumulate_data(da, t_idx, all_ind, short_name, time2idx, lead_time_h=6):
    ''' da: DataArray containing only the variable of interest '''
    if np.isin(short_name, ["tp", "tp_log", "tp_mswep", "tp_mswep_log", "pe", "e", "r", "r_log"]) or '_6hr' in short_name:
        step_h = 6 if '_6hr' in short_name else 1 # accumulation period of each stored value
        assert lead_time_h % step_h == 0, f"Lead time should be divisible by {step_h} for {short_name}, got {lead_time_h}"
        ds_acc = []
        for x_ind in all_ind:
            list_t = [int(t_) for t_ in np.array([x_ind - np.timedelta64(it, 'h') for it in range(0, lead_time_h, step_h)]).astype("datetime64[ns]")] # list of all times that must be accumulated
            if 'mswep' in short_name: # ds is not sampled every hour, keep only the times present in the dataset
                list_t = [t_ for t_ in list_t if t_ in time2idx]
            try:
                list_t_idx = [time2idx[t_] for t_ in list_t]
            except KeyError:
                print(f"CANNOT FETCH {short_name}", np.array(list_t).astype("datetime64[ns]"))
                raise
            #print(f"in accumulate: sum {short_name} over", da.isel(time=list_t_idx).time.values, f"for {x_ind}")
            ds_acc.append(da.isel(time=list_t_idx).transpose('time', 'latitude', 'longitude').sum(dim='time').values)
        return np.stack(ds_acc, axis=0)

    else:
        #print(f"in accumulate: {short_name}", da.isel(time=t_idx).time.values)
        return da.isel(time=t_idx).transpose('time', 'latitude', 'longitude').values
