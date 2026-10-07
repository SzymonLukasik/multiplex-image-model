#!/usr/bin/env python
"""Instantiate the equivariant-v3 and vanilla-wide zs-hn models and compare capacity.

    source /net/tscratch/people/plgslukasik/venvs/immu-vis/bin/activate
    python _compare_model_capacity.py
"""

import torch
from ruamel.yaml import YAML

from multiplex_model.utils.configuration import TrainingConfig
from train_masked_model import build_model

CONFIGS = [
    ("equivariant-v3", "configs/train_equivariant_v3_ddp_zs_hn_config.yaml"),
    ("vanilla-wide", "configs/train_vanilla_wide_zs_hn_config.yaml"),
]

yaml = YAML(typ="safe")


def count(m):
    return sum(p.numel() for p in m.parameters())


summary = {}

for label, path in CONFIGS:
    print(f"\n{'=' * 78}\n{label}  ({path})\n{'=' * 78}")
    with open(path) as f:
        raw = yaml.load(f)
    config = TrainingConfig(**raw)
    num_channels = len(config.tokenizer_config)
    model_config = {
        "num_channels": num_channels,
        "encoder_config": config.encoder_config.model_dump(),
        "decoder_config": config.decoder_config.model_dump(),
    }
    model = build_model(model_config, config.model_type)

    total = count(model)
    summary[label] = {"total": total, "num_channels": num_channels}
    print(f"num_channels (tokenizer): {num_channels}")
    print(f"TOTAL params : {total:,}")

    print("\n-- top level --")
    for name, child in model.named_children():
        n = count(child)
        if n:
            print(f"  {name:<30} {n:>14,}  ({100 * n / total:.1f}%)")
            summary[label][name] = n

    print("\n-- second level (>0 params) --")
    for name, child in model.named_children():
        for sub, subchild in child.named_children():
            n = count(subchild)
            if n:
                print(f"  {name}.{sub:<32} {n:>12,}")

    print("\n-- third level under encoder (>0 params) --")
    for name, child in model.named_children():
        if "enc" not in name.lower():
            continue
        for sub, subchild in child.named_children():
            for s2, c2 in subchild.named_children():
                n = count(c2)
                if n:
                    print(f"  {name}.{sub}.{s2:<30} {n:>12,}")

    del model

print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
for label, d in summary.items():
    print(f"{label:<18} total {d['total']:>14,}")
labels = list(summary)
if len(labels) == 2:
    a, b = (summary[x]["total"] for x in labels)
    print(f"\nratio {labels[0]} / {labels[1]} = {a / b:.3f}×")
