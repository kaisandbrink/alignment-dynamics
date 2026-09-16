import torch
import pandas as pd
import scipy
import numpy as np
from . import evals


def get_attn_map(model, prefix_ids, itos, device):
    """
    Run a single forward pass and return attention maps for all layers/heads.
    
    Args:
        prefix_ids: list of token ids (the probe's prefix_ids)
    
    Returns:
        attn_maps: dict {layer: np.array (n_heads, seq_len, seq_len)}
        tokens:    list of token strings for axis labels
    """
    n_layers = len(model.blocks)
    attn_store = {i: [] for i in range(n_layers)}

    def make_hook(layer_idx, store):
        def hook(module, inp, out):
            store[layer_idx].append(inp[0].detach().float().cpu())
        return hook

    hooks = [block.attn.dropout.register_forward_hook(make_hook(i, attn_store))
             for i, block in enumerate(model.blocks)]

    x = torch.tensor([prefix_ids], dtype=torch.long).to(device)
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    attn_maps = {i: attn_store[i][0][0].numpy() for i in range(n_layers)}  # (n_heads, T, T)
    tokens = [itos[t] for t in prefix_ids]

    return attn_maps, tokens

def get_condition_attention(model, probe_dict, device, itos):
    """
    For a single probe, return per-layer per-head attention from
    final preamble → condition token.
    
    Returns dict {layer: np.array (n_heads,)}
    """
    n_layers = len(model.blocks)
    attn_store = {i: [] for i in range(n_layers)}

    def make_hook(layer_idx, store):
        def hook(module, inp, out):
            store[layer_idx].append(inp[0].detach().float().cpu())
        return hook

    hooks = [block.attn.dropout.register_forward_hook(make_hook(i, attn_store))
             for i, block in enumerate(model.blocks)]

    x = torch.tensor([probe_dict['prefix_ids']], dtype=torch.long).to(device)
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        h.remove()

    final_preamble_pos = len(probe_dict['prefix_ids']) - 1

    result = {}
    for i in range(n_layers):
        attn = attn_store[i][0][0].numpy()  # (n_heads, T, T)
        result[i] = attn[:, final_preamble_pos, :]  # (n_heads, T) — full row

    return result

def find_condition_positions(prefix_ids, rule, grammar):
    """
    Find positions of condition value tokens in a probe's prefix_ids.
    
    Structure: [bos, init_str, preamble, VALUE, preamble, VALUE, ..., final_preamble]
    Value tokens sit at positions 3, 5, 7, ... 
    We check which feature each preamble belongs to and keep only condition features.
    """
    condition_positions = []

    # Pairs start at index 2: (preamble at 2i, value at 2i+1)
    # Stop before the final preamble (last token)
    for i in range(2, len(prefix_ids) - 1, 2):
        preamble_str = grammar.itos[prefix_ids[i]]
        value_pos    = i + 1

        # Map preamble → feature key
        feat = grammar._preambles_to_key.get(preamble_str)
        if feat is not None and feat in rule.conditions:
            condition_positions.append(value_pos)

    return condition_positions


def condition_attn_across_checkpoints(model, checkpoints, probes, rule,
                                      RuleSystem, device, layer=2):
    """
    For a set of probes for one rule, track attention to condition tokens
    across all checkpoints.

    Args:
        probes:     list of full token sequences (from test_data['single_rules'][rule_id])
        rule:       Rule object (from RuleSystem.rules[rule_idx])
        layer:      which layer to extract attention from

    Returns:
        pd.DataFrame with columns [epoch, max_attn, mean_attn, head_entropy]
    """
    # Pre-compute probe dicts + condition positions once
    probe_data = []
    for seq in probes:
        probe_dict = evals.tokens_to_last_token_probe(seq, RuleSystem)
        cond_pos   = find_condition_positions(probe_dict['prefix_ids'], rule, RuleSystem.grammar)
        if len(cond_pos) > 0:
            probe_data.append((probe_dict, cond_pos))
            
    records = []

    for epoch in sorted(checkpoints.keys()):
        model.load_state_dict(checkpoints[epoch])
        model.eval()

        epoch_attns = []  # collect (n_heads,) per probe

        for probe_dict, cond_pos in probe_data:
            attn_rows = get_condition_attention(model, probe_dict, device, RuleSystem.grammar.itos)
            attn_row  = attn_rows[layer]                     # (n_heads, T)
            attn_to_cond = attn_row[:, cond_pos].sum(axis=1) # (n_heads,) — sum over condition tokens
            epoch_attns.append(attn_to_cond)

        # Average across probes: (n_probes, n_heads) → (n_heads,)
        mean_over_probes = np.stack(epoch_attns).mean(axis=0)

        records.append({
            'epoch':         epoch,
            'max_attn':      mean_over_probes.max(),
            'mean_attn':     mean_over_probes.mean(),
            'dominant_head': mean_over_probes.argmax(),
            'head_entropy':  float(scipy.stats.entropy(mean_over_probes + 1e-8)),
        })

    return pd.DataFrame(records)

def rule_avg_attn_across_checkpoints(model, checkpoints, test_data, RuleSystem, device, layer=2):
    
    all_rule_dfs = []
    for rule_id in test_data['single_rules']:
        if len(test_data['single_rules'][rule_id]) == 0:
            continue
        rule_idx = int(rule_id)
        rule = RuleSystem.rules[rule_idx]
        probes = test_data['single_rules'][rule_id]
        df = condition_attn_across_checkpoints(model, checkpoints, probes, rule, RuleSystem, device, layer=layer)
        df['rule_id'] = rule_id
        all_rule_dfs.append(df)

    combined = pd.concat(all_rule_dfs)
    return combined.groupby('epoch')[['max_attn', 'mean_attn', 'head_entropy']].mean().reset_index()


#######
# PLOTTING
#######

import matplotlib.pyplot as plt
import seaborn as sns

def plot_attn_map(attn_maps, tokens, layer, figsize_per_head=3):
    """Plot attention maps for all heads in a given layer."""
    n_heads = attn_maps[layer].shape[0]
    fig, axs = plt.subplots(1, n_heads, figsize=(figsize_per_head*n_heads, figsize_per_head+1))

    for head in range(n_heads):
        ax = axs[head]
        sns.heatmap(attn_maps[layer][head], ax=ax,
                    xticklabels=tokens, yticklabels=tokens,
                    cmap='Blues', vmin=0, cbar=(head == n_heads-1))
        ax.set_title(f'Head {head}')
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(tokens, rotation=0,  fontsize=8)

    plt.suptitle(f'Layer {layer}', y=1.02)
    fig.tight_layout()