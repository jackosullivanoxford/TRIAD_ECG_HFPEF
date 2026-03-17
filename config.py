"""Functions and default values for configuration."""
import collections

def update_config_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_config_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def dict_to_str(d, prefix=""):
    if isinstance(d, collections.abc.Mapping):
        return ",".join(dict_to_str(d[k], prefix=f"{prefix}{k}_") for k in sorted(d))
    out = str(d).replace("[", "-").replace("]", "-").replace("/", "_")
    for c in '\\:*?"<>|$': out = out.replace(c, "")
    return f"{prefix}{out}"

cfg = {
    "optimizer": {
        "optimizer": "adam",
        "batch_size": 32,
        "lr": 1e-5,
        "weight_decay": 1e-3,
        "n_epochs": 1000,
        "reduce_on_plateau": True,
        "patience": 15,
        "max_reduction": 1e-2,
        "lr_plateaus": 4,
        "save_all_checkpoints": False,
        "save_path": "checkpoints/",
    },
    "dataloader": {
        "binary": True,
        "binary_cutoff": 0.5,
        "binary_positive_class": "above",
        "remove_labels": [],
        "normalize_x": True,
        "normalize_y": False,
        "notch_filter": False,
        "baseline_filter": False,
        "downsample": 2,
        "accept_all_lengths": False,
        "leads": None,
        "n_dataloader_workers": 4,  # Optimized for general use
        "crossval_idx": None,
        "overread_csv": "data/overreads.csv",
        "waveforms": "data/waveforms", # Points to generated demo data
        "waveform_type": "npy",
        "filekey": "record_name",       # Matches generate_demo_data.py
    },
    "model": {
        "model_type": "jos",
        "is_2d": True,
        "conv_width": 3,
        "drop_prob": 0.0,
        "batch_norm": True,
        "pos_weight": 1.0,
    },
}

task_cfg = {
    "HFpEF": {
        "dataloader": {
            "task": "HFpEF",
            "label_file": "data/labels.csv", # Points to generated demo labels
            "label_keys": ["HFpEF"],
        },
        "model": {"pos_weight": 20.0}
    }
}
