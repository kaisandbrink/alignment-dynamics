import random
import torch
import pandas as pd
from . import evals
from .attention_weights  import find_condition_positions


# def find_condition_positions(prefix_ids, rule, grammar):
#     """
#     Find positions of condition value tokens in a probe's prefix_ids.
    
#     Structure: [bos, init_str, preamble, VALUE, preamble, VALUE, ..., final_preamble]
#     Value tokens sit at positions 3, 5, 7, ... 
#     We check which feature each preamble belongs to and keep only condition features.
#     """
#     condition_positions = []

#     # Pairs start at index 2: (preamble at 2i, value at 2i+1)
#     # Stop before the final preamble (last token)
#     for i in range(2, len(prefix_ids) - 1, 2):
#         preamble_str = grammar.itos[prefix_ids[i]]
#         value_pos    = i + 1

#         # Map preamble → feature key
#         feat = grammar._preambles_to_key.get(preamble_str)
#         if feat is not None and feat in rule.conditions:
#             condition_positions.append(value_pos)

#     return condition_positions

def generate_patching_prompt_pairs(RuleSystem, rule_id, prompt):
    
    rule = RuleSystem.rules[rule_id]
    cond_key, cond_val = next(iter(rule.conditions.items()))
    alternate_val = random.choice([val for val in RuleSystem.grammar.variables[cond_key] if val != cond_val])
    
    stoi = RuleSystem.grammar.stoi
    cond_val_tok = stoi[cond_val]
    alternate_val_tok = stoi[alternate_val]

    assert cond_val_tok in prompt, 'condition not in prompt'

    cond_positions = find_condition_positions(prompt, rule, RuleSystem.grammar)
    assert len(cond_positions) > 0, f'condition position not found for rule {rule_id}'
    cond_position = cond_positions[0]

    alternate_prompt = prompt.copy()
    alternate_prompt[cond_position] = alternate_val_tok
    prompt_dict = evals.tokens_to_last_token_probe(prompt, RuleSystem)

    return {'prompt': prompt_dict,
            'alternate_prompt': evals.tokens_to_last_token_probe(alternate_prompt, RuleSystem),}


### ------------------
# BLOCK PATCHING
###  ------------------

@torch.no_grad()
def cache_block_outputs(model, x):
    model.eval()
    caches = []
    hooks = []

    def make_hook(store):
        def hook(module, inp, out):
            store.append(out.detach().clone())
        return hook

    for block in model.blocks:
        hooks.append(block.register_forward_hook(make_hook(caches)))

    _ = model(x)
    for h in hooks:
        h.remove()
    return caches

@torch.no_grad()
def run_with_final_token_patch(model, x_receiver, donor_cache, layer_idx):
    """
    Replace the receiver's final-token residual stream at one layer
    with the donor's final-token residual stream.
    """
    model.eval()
    final_pos = x_receiver.shape[1] - 1

    def patch_hook(module, inp, out):
        patched = out.clone()
        patched[:, final_pos, :] = donor_cache[layer_idx][:, final_pos, :]
        return patched

    hook = model.blocks[layer_idx].register_forward_hook(patch_hook)

    try:
        logits = model(x_receiver)
    finally:
        hook.remove()

    return logits

### ------------------
# ATTENTION PATCHING
###  ------------------

@torch.no_grad()
def cache_attn_outputs(model, x):
    """Cache output of each attention sublayer (before residual add)."""
    caches = []
    hooks = []

    def make_hook(store):
        def hook(module, inp, out):
            store.append(out.detach().clone())
        return hook

    for block in model.blocks:
        hooks.append(block.attn.register_forward_hook(make_hook(caches)))

    _ = model(x)
    for h in hooks:
        h.remove()
    return caches  # list of (1, T, d_model), one per layer


@torch.no_grad()
def run_with_attn_patch(model, x_receiver, donor_attn_cache, layer_idx):
    """Patch only the attention output at layer_idx, final token position."""
    final_pos = x_receiver.shape[1] - 1

    def patch_hook(module, inp, out):
        patched = out.clone()
        patched[:, final_pos, :] = donor_attn_cache[layer_idx][:, final_pos, :]
        return patched

    hook = model.blocks[layer_idx].attn.register_forward_hook(patch_hook)
    try:
        logits = model(x_receiver)
    finally:
        hook.remove()
    return logits

###### -----------
# FF PATCHING
##### ------------

@torch.no_grad()
def cache_ff_outputs(model, x):
    caches = []
    hooks = []

    def make_hook(store):
        def hook(module, inp, out):
            store.append(out.detach().clone())
        return hook

    for block in model.blocks:
        hooks.append(block.ff.register_forward_hook(make_hook(caches)))

    _ = model(x)
    for h in hooks:
        h.remove()
    return caches


@torch.no_grad()
def run_with_ff_patch(model, x_receiver, donor_ff_cache, layer_idx):
    final_pos = x_receiver.shape[1] - 1

    def patch_hook(module, inp, out):
        patched = out.clone()
        patched[:, final_pos, :] = donor_ff_cache[layer_idx][:, final_pos, :]
        return patched

    hook = model.blocks[layer_idx].ff.register_forward_hook(patch_hook)
    try:
        logits = model(x_receiver)
    finally:
        hook.remove()
    return logits


### ------------------
#  PATCHING LOOP
###  ------------------

def probe_score_from_logits(logits, probe):
    probs = torch.softmax(logits[0, -1], dim=-1)
    p_aligned = probs[probe['aligned_tokens'][0]].item()
    p_misaligned = sum(probs[t].item() for t in probe['misaligned_tokens'])
    return {
        'p_aligned': p_aligned,
        'p_misaligned': p_misaligned,
        'p_ungrammatical': 1 - p_aligned - p_misaligned,
    }

@torch.no_grad()
def steer_probe_all_layers(model, aligned_probe, alternate_probe, patch_type = 'attention', device='mps'):

    assert patch_type in ['attention', 'block', 'ff'], f'Invalid patch_type : {patch_type}\n must be one of: attention, block, ff'
    if patch_type == 'attention':
        cache_func = cache_attn_outputs
        run_func = run_with_attn_patch
    elif patch_type == 'block':
        cache_func = cache_block_outputs
        run_func = run_with_final_token_patch
    elif patch_type == 'ff':
        cache_func = cache_ff_outputs
        run_func   = run_with_ff_patch

    donor_x = torch.tensor([aligned_probe['prefix_ids']], dtype=torch.long, device=device)
    recv_x  = torch.tensor([alternate_probe['prefix_ids']], dtype=torch.long, device=device)
    
    donor_cache  = cache_func(model, donor_x)
    base_logits  = model(recv_x)
    base_scores  = probe_score_from_logits(base_logits, aligned_probe)

    rows = []
    for layer_idx in range(len(model.blocks)):
        patched_logits = run_func(model, recv_x, donor_cache, layer_idx)
        patched_scores = probe_score_from_logits(patched_logits, aligned_probe)
        rows.append({'layer': layer_idx, 'phase': 'base',    'p_aligned': base_scores['p_aligned'],    'p_misaligned': base_scores['p_misaligned']})
        rows.append({'layer': layer_idx, 'phase': 'patched', 'p_aligned': patched_scores['p_aligned'], 'p_misaligned': patched_scores['p_misaligned']})
    return pd.DataFrame(rows)


def steer_probe_set(model, test_data, RuleSystem, patch_type = 'attention', device='mps'):
    all_dfs = []
    for rule_id in range(len(RuleSystem.rules)):
        for prompt in test_data['single_rules'][str(rule_id)]:
            prompt_pairs = generate_patching_prompt_pairs(RuleSystem, rule_id, prompt)
            res = steer_probe_all_layers(model, prompt_pairs['prompt'], prompt_pairs['alternate_prompt'], patch_type = patch_type, device = device)
            res['rule_id'] = rule_id
            all_dfs.append(res)
    return pd.concat(all_dfs)
