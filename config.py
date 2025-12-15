"""Functions and default values for configuration."""
import collections

def update_config_dict(d,u):
    for k,v in u.items():
        d[k]=update_config_dict(d.get(k,{}),v) if isinstance(v,collections.Mapping) else v
    return d

def dict_to_str(d,prefix=""):
    if isinstance(d,collections.Mapping):
        return ",".join(dict_to_str(d[k],prefix=f"{prefix}{k}_") for k in sorted(d))
    out=str(d).replace("[","-").replace("]","-").replace("/","_")
    for c in '\\:*?"<>|$':out=out.replace(c,"")
    return f"{prefix}{out}"

cfg={
    "optimizer":{
        "optimizer":"adam",
        "batch_size":32,
        "lr":1e-5,
        "weight_decay":1e-3,
        "n_epochs":1000,
        "reduce_on_plateau":True,
        "patience":15,
        "max_reduction":1e-2,
        "lr_plateaus":4,
        "save_all_checkpoints":False,
        "save_path":"/path/to/save",
    },
    "dataloader":{
        "binary":True,
        "binary_cutoff":.5,
        "binary_positive_class":"above",
        "remove_labels":[],
        "normalize_x":True,
        "normalize_y":False,
        "notch_filter":False,
        "baseline_filter":False,
        "downsample":2,
        "accept_all_lengths":False,
        "leads":None,
        "n_dataloader_workers":32,
        "crossval_idx":None,
        "overread_csv":"/path/to/overreads",
        "waveforms":"path/to/ecgs",
        "waveform_type":"npy",
        "filekey":"deid_filename",
    },
    "model":{
        "model_type":"jos",
        "is_2d":True,
        "conv_width":3,
        "drop_prob":0.,
        "batch_norm":True,
        "pos_weight":1.,
    },
}

task_cfg={
    "HFpEF":{
        "dataloader":{
            "task":"HFpEF",
            "label_file":"/path/to/labelfile.csv",
            "label_keys":["HFpEF"],
        },
        "model":{"pos_weight":20.0}
    }
}
