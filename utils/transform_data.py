import dataclasses

import torch
import numpy as np

VARS_ACCUMULATE = ("tp", "pe", "e", "r") # hourly accumulations, summed over the lead time (as are all 6-hourly ones, see is_6hr_var)
VARS_TRANSFORM = ("tp", "tp_6hr", "tp_mswep", "r", "r_6hr") # surface vars that can be used as *_log, i.e. log(1 + x/eps)


def is_transformed_var(short_name):
    return short_name.endswith('_log') and short_name[:-len('_log')] in VARS_TRANSFORM


def is_6hr_var(short_name):
    return '_6hr' in short_name or 'mswep' in short_name # stored as totals over the 6 h ending at the timestamp


def is_accumulated_var(short_name):
    return short_name.replace('_log', '') in VARS_ACCUMULATE or is_6hr_var(short_name)


def transform_data(data, short_name, eps=1e-5, direct=True): # transformation follows [Rasp 2020], also used in FourCastNet
    if is_transformed_var(short_name):
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


def inverse_transform_vars(batch):
    ''' Return a copy of a prediction Batch with *_log vars mapped back to physical units; other vars are untouched '''
    return dataclasses.replace(batch, surf_vars={
        k: transform_data(v, k, direct=False) if is_transformed_var(k) else v
        for k, v in batch.surf_vars.items()
    })


def accumulation_offsets_h(short_name, lead_time_h=6):
    ''' hours before a timestamp that accumulate_data sums over ([0] if not accumulated) '''
    if is_accumulated_var(short_name):
        step_h = 6 if is_6hr_var(short_name) else 1 # accumulation period of each stored value
        assert lead_time_h % step_h == 0, f"Lead time should be divisible by {step_h} for {short_name}, got {lead_time_h}"
        return list(range(0, lead_time_h, step_h))
    return [0]


def accumulate_data(da, t_idx, all_ind, short_name, time2idx, lead_time_h=6):
    ''' da: DataArray containing only the variable of interest '''
    if is_accumulated_var(short_name):
        ds_acc = []
        for x_ind in all_ind:
            list_t = [int(t_) for t_ in np.array([x_ind - np.timedelta64(it, 'h') for it in accumulation_offsets_h(short_name, lead_time_h)]).astype("datetime64[ns]")] # list of all times that must be accumulated
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
