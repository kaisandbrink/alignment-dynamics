import torch
import pandas as pd
from . import evals
from .attention_weights import find_condition_positions

################ ------------------
## WHOLE LAYER ABLATION
################ ------------------

def identity_block_hook(module, inp, out):
    # block.forward(x) = x + attn(ln1(x)) + ff(ln2(x))
    # returning inp[0] (the input x) makes the whole block act as identity
    return inp[0]

def zero_attn_output_hook(module, inp, out):
    return torch.zeros_like(out)

@torch.no_grad()
def evaluate_with_ablation(model, probes, layer_idx, 
                           device='mps', batch_size=256, 
                           ablate_attn_only=True):

    if ablate_attn_only:
        hook = model.blocks[layer_idx].attn.register_forward_hook(zero_attn_output_hook)
    else:
        hook = model.blocks[layer_idx].register_forward_hook(identity_block_hook)

    try:
        results = evals.evaluate_probes_batched(
            model=model,
            probes=probes,
            device=device,
            batch_size=batch_size,
        )
    finally:
        hook.remove()
    return results

@torch.no_grad()
def collect_ablation_scores(model, probes, layer_idx, ablate_attn_only=True, device = 'mps'):
    batch_size = len(probes)
    baseline = evals.evaluate_probes_batched(model, probes, device=device, batch_size=batch_size)
    ablated  = evaluate_with_ablation(model, probes, layer_idx=layer_idx, device=device, batch_size=batch_size, ablate_attn_only=ablate_attn_only)
    
    ablation_res = {'baseline': pd.DataFrame(baseline),
                    'ablated': pd.DataFrame(ablated)}
    
    reses = []
    for typ in ablation_res:
        reses.append({'p_aligned': ablation_res[typ]['p_aligned'].mean(),
                      'p_misaligned': ablation_res[typ]['p_misaligned'].mean(),
                      'p_ungrammatical': ablation_res[typ]['p_ungrammatical'].mean(),
                       'ablation': typ})

    return pd.DataFrame(reses)

def make_condition_token_ablation_hook(final_positions, cond_pos_list):
    """
    Zero attention from each example's final token to its condition-token
    positions, then renormalize that attention row.

    Assumes the hook is attached to `block.attn.dropout`, where the tensor
    has shape (B, H, T, T).
    """
    eps = 1e-9

    def hook(module, inp, out):
        # out: (B, H, T, T)
        attn = out.clone()

        batch_size = attn.shape[0]
        for b in range(batch_size):
            cond_pos = cond_pos_list[b]
            if len(cond_pos) == 0:
                continue

            fp = final_positions[b]
            cond_pos_t = torch.tensor(cond_pos, device=attn.device, dtype=torch.long)

            # Zero attention from final token -> condition tokens
            attn[b, :, fp, cond_pos_t] = 0.0

            # Renormalize that query row so it still sums to 1
            row = attn[b, :, fp, :]  # (H, T)
            row_sum = row.sum(dim=-1, keepdim=True).clamp_min(eps)
            attn[b, :, fp, :] = row / row_sum

        return attn

    return hook

################ ------------------
## CONDITION SPECIFIC TOKEN ABLATION
################ ------------------

@torch.no_grad()
def evaluate_probes_batched_condition_ablation(
    model,
    probes,
    rule,
    grammar,
    layer_idx,
    device='cpu',
    batch_size=256,
):
    """
    condition-token ablation applied at a single attention layer.

    Returns the same columns:
        - rule_idx
        - target_feature
        - p_aligned
        - p_misaligned
        - p_ungrammatical
    """
    model.eval()
    results = []
    pad_id = 0

    for i in range(0, len(probes), batch_size):
        batch = probes[i:i + batch_size]
        lengths = [len(p['prefix_ids']) for p in batch]
        max_len = max(lengths)

        # Right-pad within batch
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)

        final_positions = []
        cond_pos_list = []

        for j, (probe, length) in enumerate(zip(batch, lengths)):
            padded[j, :length] = torch.tensor(probe['prefix_ids'], dtype=torch.long, device=device)
            final_positions.append(length - 1)

            # Condition-token positions for this probe
            cond_pos = find_condition_positions(probe['prefix_ids'], rule, grammar)
            cond_pos_list.append(cond_pos)

        hook = model.blocks[layer_idx].attn.dropout.register_forward_hook(
            make_condition_token_ablation_hook(final_positions, cond_pos_list)
        )

        try:
            logits = model(padded)  # (B, T, V)
        finally:
            hook.remove()

        for j, probe in enumerate(batch):
            probs = torch.softmax(logits[j, lengths[j] - 1, :], dim=-1)
            p_aligned = sum(probs[t].item() for t in probe['aligned_tokens'])
            p_misaligned = sum(probs[t].item() for t in probe['misaligned_tokens'])

            results.append({
                'rule_idx': probe['rule_idx'],
                'target_feature': probe['target_feature'],
                'p_aligned': p_aligned,
                'p_misaligned': p_misaligned,
                'p_ungrammatical': 1 - p_aligned - p_misaligned,
            })

    return results

@torch.no_grad()
def collect_token_ablation_scores(model, test_data, rulesystem, layer_idx, device = 'mps'):
    
    reses = []
    for rule_id in test_data['single_rules']:
        
        probes = [evals.tokens_to_last_token_probe(f, rulesystem) for f in test_data['single_rules'][rule_id]]

        rule = rulesystem.rules[int(rule_id)]
        baseline = evals.evaluate_probes_batched(model, probes, device=device)
        ablated = evaluate_probes_batched_condition_ablation(
                model=model,
                probes=probes,
                rule=rule,
                grammar=rulesystem.grammar,
                layer_idx=layer_idx, 
                device=device,
                batch_size=len( probes ),
            )

        ablation_res = {'baseline': pd.DataFrame(baseline),
                        'ablated': pd.DataFrame(ablated)}
        
        for typ in ablation_res:
            reses.append({'p_aligned': ablation_res[typ]['p_aligned'].mean(),
                        'p_misaligned': ablation_res[typ]['p_misaligned'].mean(),
                        'p_ungrammatical': ablation_res[typ]['p_ungrammatical'].mean(),
                        'ablation': typ,
                        'rule': rule_id})
            

    return pd.DataFrame(reses)
