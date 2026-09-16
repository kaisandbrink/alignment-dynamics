import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
from itertools import combinations
import pandas as pd
from tqdm import tqdm
import seaborn as sns

import sys
sys.path.append('../../../')
import src.deontic_ethics_utils as deontic_utils
from src.datasets.DeonticEthicsDataset import TemplateDataset, Grammar
from src.trainers import evaluate


@torch.no_grad()
def evaluate_alignment(model, eval_dataset, ruleset, batch_size=64, device = 'cpu'):
    
    n = len(eval_dataset)
    grammatically_correct = 0
    aligned = 0
    sentences = []

    # Batch the prompts
    prompts = torch.stack([inputs['input_ids'][:-1] for inputs in eval_dataset])  # (n, seq_len)
    device = next(model.parameters()).device

    for i in range(0, n, batch_size):
        batch_prompts = prompts[i:i+batch_size].to(device)  # (bs, seq_len)
        batch_seqs = generate_sequence(model, ruleset.grammar, batch_prompts, max_new_tokens = 15)  # (bs, seq_len+20)

        for seq in batch_seqs:
            sentences.append(seq)
            template = ruleset.grammar.tokens_to_template(seq.tolist())
            if template:
                grammatically_correct += 1
                aligned += ruleset.evaluate(template)['overall_ok']

    return {
        'grammatically_correct': grammatically_correct / n,
        'aligned': aligned / n,
        'generations': sentences,
    }

def generate_last_token_probes(rule_sys, n_per_rule=10, n_tries = 500, seed=42):
    """
    Generate probes where the last token determines alignment.

    For each rule, builds prefixes that satisfy the rule's conditions with
    the target feature placed last (only preamble given). The model must
    predict the final value token.

    Returns list of dicts with:
      - prefix_ids: token IDs up to (not including) the final value
      - target_feature: which feature the last token belongs to
      - aligned_tokens: value token IDs that keep the template aligned
      - misaligned_tokens: value token IDs that break alignment
      - rule_idx: which rule is being tested
    """
    rng = random.Random(seed)
    probes = []
    grammar = rule_sys.grammar
    features = grammar.variables
    preambles = grammar.preambles
    stoi = grammar.stoi

    for rule_idx, rule in enumerate(rule_sys.rules):
        target_key = rule.target_key()
        seen = set() ## only store new

        for loopi in range(n_tries):
            template_features = {}
            template_preambles = {}

            # Set condition features to satisfy the rule
            for cond_key, cond_val in rule.conditions.items():
                if deontic_utils.is_neg(cond_val):
                    valid = [v for v in features[cond_key] if v != deontic_utils.neg_val(cond_val)]
                    template_features[cond_key] = rng.choice(valid)
                else:
                    template_features[cond_key] = cond_val
                template_preambles[cond_key] = rng.choice(preambles[cond_key])

            # Optionally add 0-2 extra features (not target, not condition)
            other_keys = [k for k in features if k != target_key and k not in rule.conditions]
            n_extra = rng.randint(0, min(1, len(other_keys)))
            for k in rng.sample(other_keys, n_extra):
                template_features[k] = rng.choice(features[k])
                template_preambles[k] = rng.choice(preambles[k])

            # Build prefix tokens: <bos> init_str [preamble value]* target_preamble
            # Shuffle non-target features for variation
            feature_keys = list(template_features.keys())
            rng.shuffle(feature_keys)
            prefix_ids = [stoi['<bos>'], stoi[rng.choice(grammar.init_str)]]
            
            for k in feature_keys:
                prefix_ids.append(stoi[template_preambles[k]])
                prefix_ids.append(stoi[template_features[k]])

            target_preamble = rng.choice(preambles[target_key])
            prefix_ids.append(stoi[target_preamble])

            # Check each possible target value against full rule system
            aligned_tokens = []
            misaligned_tokens = []
            for val in features[target_key]:
                test_features = dict(template_features)
                test_features[target_key] = val
                test_preambles = dict(template_preambles)
                test_preambles[target_key] = target_preamble
                t = deontic_utils.Template(features=test_features, preambles=test_preambles, init_str=grammar.init_str[0])
                result = rule_sys.evaluate(t)
                if result['overall_ok']:
                    aligned_tokens.append(stoi[val])
                else:
                    misaligned_tokens.append(stoi[val])
            
            new_probe = tuple(prefix_ids)
            if aligned_tokens and misaligned_tokens and new_probe not in seen:
                seen.add(new_probe)
                probes.append({
                    'prefix_ids': prefix_ids,
                    'target_feature': target_key,
                    'aligned_tokens': aligned_tokens,
                    'misaligned_tokens': misaligned_tokens,
                    'rule_idx': rule_idx,
                })
            
            if len(seen) == n_per_rule:
                break

            if loopi == n_tries - 1:
                print(f'Insuffienct probes found for rule: {rule}\n \
                      only {len(seen)}/ {n_per_rule} found')

    return probes



@torch.no_grad()
def evaluate_probes(model, probes, device='cpu'):
    """
    For each probe, compute P(aligned) and P(misaligned) over the final token,
    normalized to sum to 1 across aligned + misaligned candidates only.
    """
    model.eval()
    results = []
    for probe in probes:
        input_ids = torch.tensor([probe['prefix_ids']], dtype=torch.long, device=device)
        logits = model(input_ids)[:, -1, :]  # (1, V)
        probs = torch.softmax(logits, dim=-1)[0]       # (V,)

        p_aligned = sum(probs[t].item() for t in probe['aligned_tokens'])
        p_misaligned = sum(probs[t].item() for t in probe['misaligned_tokens'])
        total = p_aligned + p_misaligned

        results.append({
            'rule_idx': probe['rule_idx'],
            'target_feature': probe['target_feature'],
            'p_aligned': p_aligned,
            'p_misaligned': p_misaligned,
            'p_ungrammatical': 1 - p_aligned - p_misaligned
        })
    return results


@torch.no_grad()
def evaluate_probes_batched(model, probes, device='cpu', batch_size=256):
    """Batched version of evaluate_probes.

    Right-pads probes to the same length within each batch and runs a single
    forward pass per batch. Because the model is causal, padding tokens
    appended after the last real token cannot influence the logit at that
    position, so results are numerically identical to evaluate_probes.

    Args:
        model: Transformer model.
        probes: List of probe dicts (same format as evaluate_probes).
        device: Device string.
        batch_size: Number of probes per forward pass.

    Returns:
        List of result dicts — same format as evaluate_probes.
    """
    model.eval()
    results = []
    pad_id = 0  # right-pad; causal masking ensures padding never affects real logits

    for i in range(0, len(probes), batch_size):
        batch = probes[i : i + batch_size]
        lengths = [len(p['prefix_ids']) for p in batch]
        max_len = max(lengths)

        # Right-pad each probe to max_len
        padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        for j, (probe, length) in enumerate(zip(batch, lengths)):
            padded[j, :length] = torch.tensor(probe['prefix_ids'], dtype=torch.long)

        logits = model(padded)  # (B, max_len, V)

        for j, (probe, length) in enumerate(zip(batch, lengths)):
            # Extract logit at the last real token position for this probe
            probs = torch.softmax(logits[j, length - 1, :], dim=-1)  # (V,)
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
def evaluate_generation_quality(model, rule_sys, device='cpu', n_sample=1000):
    """
    Sample n_sample sequences from the model and check grammar + alignment.
    Returns (n_ungrammatical, n_misaligned).

    .. deprecated::
        Use evaluate_generation_quality_batched instead.
    """
    warnings.warn(
        "evaluate_generation_quality is deprecated; use evaluate_generation_quality_batched instead.",
        DeprecationWarning, stacklevel=2,
    )
    model.eval()
    n_ungrammatical = 0
    n_misaligned = 0
    decoded_templates = []
    for _ in range(n_sample):
        decoded = sampling_decoding(model,rule_sys.grammar, device)
        decoded_template = rule_sys.grammar.tokens_to_template(decoded)
        decoded_templates.append(decoded_template if decoded_template else decoded)

        if not decoded_template:
            n_ungrammatical += 1
            continue

        aligned = rule_sys.evaluate(decoded_template)['overall_ok']
        if not aligned:
            n_misaligned += 1

    return {'n_ungrammatical':n_ungrammatical/n_sample, 
            'n_misaligned' : n_misaligned/max(1e-4, (n_sample - n_ungrammatical)),
            'decoded_templates': decoded_templates}

def _prob_mass_on(model, prefix_ids, valid_token_ids, device):
    """Total probability mass the model places on valid_token_ids given prefix_ids."""
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    logits = model(input_ids)[0, -1]  # (V,)
    probs = torch.softmax(logits, dim=-1)
    return sum(probs[t].item() for t in valid_token_ids)


@torch.no_grad()
def evaluate_grammaticality_probability(model, grammar, device='cpu', n_samples=50, n_stages=3, seed=42):
    """
    Evaluate how much probability mass the model places on grammatically valid
    next tokens at each stage of sentence construction.

    Stages tested:
      'init'        : after <bos>                   -> valid = init_str tokens
      'preamble_N'  : after N complete pairs        -> valid = preambles for unused features
      'value_N'     : after N pairs + one preamble  -> valid = values matching that preamble's feature

    Returns a dict mapping stage name -> mean probability mass over n_samples random prefixes.
    The 'init' stage has a single prefix (<bos>) so it is a single scalar, not an average.
    """
    rng = random.Random(seed)
    model.eval()
    stoi = grammar.stoi
    stage_results = {}

    # Stage: init (only one possible prefix)
    valid_init_ids = [stoi[s] for s in grammar.init_str]
    stage_results['init'] = _prob_mass_on(model, [stoi['<bos>']], valid_init_ids, device)

    for stage_idx in range(n_stages):
        preamble_masses = []
        value_masses = []

        for _ in range(n_samples):
            # Build a random valid prefix with stage_idx complete preamble+value pairs
            init = rng.choice(grammar.init_str)
            seq = [stoi['<bos>'], stoi[init]]
            used_keys = []

            available_keys = list(grammar.variable_names)
            rng.shuffle(available_keys)
            for k in available_keys[:stage_idx]:
                preamble = rng.choice(grammar.preambles[k])
                value = rng.choice(grammar.variables[k])
                seq += [stoi[preamble], stoi[value]]
                used_keys.append(k)

            remaining_keys = [k for k in grammar.variable_names if k not in used_keys]
            if not remaining_keys:
                break

            # P(valid preamble for an unused feature | current seq)
            # <eos> is also valid after at least one feature has been given
            valid_preamble_ids = [stoi[p] for k in remaining_keys for p in grammar.preambles[k]]
            if stage_idx >= 1:
                valid_preamble_ids.append(stoi['<eos>'])
            preamble_masses.append(_prob_mass_on(model, seq, valid_preamble_ids, device))

            # Extend with a random valid preamble, then measure P(correct value)
            chosen_key = rng.choice(remaining_keys)
            chosen_preamble = rng.choice(grammar.preambles[chosen_key])
            valid_value_ids = [stoi[v] for v in grammar.variables[chosen_key]]
            value_masses.append(_prob_mass_on(model, seq + [stoi[chosen_preamble]], valid_value_ids, device))

        if preamble_masses:
            stage_results[f'preamble_{stage_idx}'] = sum(preamble_masses) / len(preamble_masses)
        if value_masses:
            stage_results[f'value_{stage_idx}'] = sum(value_masses) / len(value_masses)

    return stage_results


@torch.no_grad()
def sampling_decoding(model, grammar, device):
    """.. deprecated:: Use sampling_decoding_batched instead."""
    warnings.warn(
        "sampling_decoding is deprecated; use sampling_decoding_batched instead.",
        DeprecationWarning, stacklevel=2,
    )
    bos = grammar.stoi['<bos>']
    eos = grammar.stoi['<eos>']
    seq = torch.tensor([[bos]], dtype=torch.long).to(device)

    for _ in range(100):
        logits = model(seq)[0, -1]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        next_token = np.random.choice(len(probs), p=probs)
        seq = torch.cat([seq, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        if next_token == eos:
            break

    return seq


@torch.no_grad()
def sampling_decoding_batched(model, grammar, device, batch_size=512, max_len=15):
    """
    Generate batch_size sequences in parallel by sampling from the model.
    WARNING: the returned sequence can contain non-sense after the first <eos> token, due to batched sampling we need to wait till all are finished, so some seqeuences might have to produce even after EOS.
    """
    bos = grammar.stoi['<bos>']
    eos = grammar.stoi['<eos>']
    seq = torch.full((batch_size, 1), bos, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = model(seq)[:, -1, :]                          # (B, V)
        probs = torch.softmax(logits, dim=-1)                  # (B, V)
        next_tokens = torch.multinomial(probs, num_samples=1)  # (B, 1)
        seq = torch.cat([seq, next_tokens], dim=1)
        finished |= (next_tokens.squeeze(1) == eos)
        if finished.all():
            break

    return seq  # (B, T)


@torch.no_grad()
def evaluate_generation_quality_batched(model, rule_sys, device='cpu', n_sample=1024, batch_size=512):
    """
    Batched version of evaluate_generation_quality.
    Generates sequences in parallel (batch_size at a time) for speed.
    """
    model.eval()
    grammar = rule_sys.grammar
    n_ungrammatical = 0
    n_misaligned = 0
    decoded_templates = []
    n_generated = 0
    eos_id = grammar.stoi['<eos>']

    while n_generated < n_sample:
        bs = min(batch_size, n_sample - n_generated)
        seqs = sampling_decoding_batched(model, grammar, device, batch_size=bs)

        for i in range(bs):
            seq_i = seqs[i].tolist()
            eos_pos = next((j for j, t in enumerate(seq_i) if t == eos_id), None)
            if eos_pos is not None:
                seq_i = seq_i[:eos_pos + 1]
            decoded_template = grammar.tokens_to_template(seq_i)
            decoded_templates.append(decoded_template if decoded_template else seq_i)

            if not decoded_template:
                n_ungrammatical += 1
                continue

            aligned = rule_sys.evaluate(decoded_template)['overall_ok']
            if not aligned:
                n_misaligned += 1

        n_generated += bs

    return {
        'n_ungrammatical': n_ungrammatical / n_generated,
        'n_misaligned': n_misaligned / max(1e-4, (n_generated - n_ungrammatical)),
        'decoded_templates': decoded_templates,
    }


### model generations --------------------------

@torch.no_grad()
def generate_sequence(
        model,
        grammar: Grammar,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            prompt: Starting tokens (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens

        Returns:
            Generated sequence including prompt (batch, seq_len + max_new_tokens)
        """
        model.eval()
        x = prompt

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            x_cond = x if x.size(1) <= model.max_seq_len else x[:, -model.max_seq_len:]

            # Get predictions
            logits = model(x_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            x = torch.cat([x, next_token], dim=1)

            if (next_token == grammar.stoi['<eos>']).all():
                break

        return x



########################################
### EVALS: Aggregating over runs
########################################

def evaluate_free_generations(model, histories, RuleSystem, n_sample=100, batch_size=100, device='mps'):
    """Evaluate generation quality across all checkpoints for multiple runs.

    Args:
        model: Transformer model (architecture only; weights loaded from checkpoints).
        histories: list of history dicts, each with a 'checkpoints' key.
        RuleSystem: RuleSet used for grammar/alignment evaluation.
        n_sample: Number of sequences to generate per checkpoint.
        batch_size: Batch size for generation.
        device: Device string.

    Returns:
        pd.DataFrame with columns [run, checkpoint, n_ungrammatical, n_misaligned].
    """
    model = model.to(device)
    records = []

    for run_i, run in enumerate(histories):
        history = histories[run]
        checkpoints = history['checkpoints']
        for checkpnt in tqdm(checkpoints, desc=f'Run {run_i}'):
            model.load_state_dict(checkpoints[checkpnt])
            model.eval()
            quality = evaluate_generation_quality_batched(
                model, RuleSystem, n_sample=n_sample, device=device, batch_size=batch_size
            )
            records.append({
                'run': run_i,
                'checkpoint': checkpnt,
                'n_ungrammatical': quality['n_ungrammatical'],
                'n_misaligned': quality['n_misaligned'],
            })

    return pd.DataFrame(records)

def evaluate_test_alignment(model, histories, RuleSystem, all_test, n_test=50, device='mps'):
    """Evaluate alignment on held-out test sequences across all checkpoints and runs.

    Args:
        model: Transformer model.
        histories: list of history dicts, each with a 'checkpoints' key.
        RuleSystem: RuleSet for alignment evaluation.
        all_test: list of test token sequences.
        n_test: Number of test sequences to use.
        device: Device string.

    Returns:
        pd.DataFrame with columns [run, checkpoint, aligned, grammatically_correct].
    """
    model = model.to(device)
    test_dataset = TemplateDataset(all_test[:n_test])
    records = []

    for run_i, run in enumerate(histories):
        history = histories[run]
        checkpoints = history['checkpoints']
        for checkpnt in tqdm(checkpoints, desc=f'Run {run_i}'):
            model.load_state_dict(checkpoints[checkpnt])
            model.eval()
            ev = evaluate_alignment(model, test_dataset, RuleSystem, device=device)
            records.append({
                'run': run_i,
                'checkpoint': checkpnt,
                'aligned': ev['aligned'],
                'grammatically_correct': ev['grammatically_correct'],
            })

    return pd.DataFrame(records)

def infer_rule_idx(temp, rule_sys):
    """Find the index of the rule whose conditions are all present in the template."""
    applicable_rules = []
    for i, rule in enumerate(rule_sys.rules):
        if all(temp.features.get(k) == v for k, v in rule.conditions.items()):
            applicable_rules.append(i)
    
    return str(applicable_rules)

def tokens_to_last_token_probe(token_seq, rule_sys):
    """Convert a full token sequence into a last-token probe dict.

    Strips the final value token, infers which feature is being predicted,
    and enumerates which values for that feature are aligned vs misaligned
    under the rule system.

    Args:
        token_seq: Full token sequence including preamble and value tokens.
        rule_sys: RuleSet object with .grammar, .rules, .evaluate().

    Returns:
        dict with keys:
            - prefix_ids: token ids up to (and including) the final preamble
            - target_feature: the feature key being predicted (str)
            - aligned_tokens: list of token ids that yield an aligned completion
            - misaligned_tokens: list of token ids that yield a misaligned completion
            - rule_idx: index of the matching rule (or None)
    """
    grammar = rule_sys.grammar
    stoi = grammar.stoi

    prefix_ids = list(token_seq[:-2])
    target_preamble_str = grammar.itos[prefix_ids[-1]]
    target_key = next(
        feat for feat, preams in grammar.preambles.items()
        if target_preamble_str in preams
    )

    temp = grammar.tokens_to_template(token_seq)
    rule_idx = infer_rule_idx(temp, rule_sys)

    aligned_tokens, misaligned_tokens = [], []
    for val in grammar.variables[target_key]:
        test_temp = deontic_utils.Template(
            features={**temp.features, target_key: val},
            preambles={**temp.preambles, target_key: target_preamble_str},
            init_str=temp.init_str
        )
        if rule_sys.evaluate(test_temp)['overall_ok']:
            aligned_tokens.append(stoi[val])
        else:
            misaligned_tokens.append(stoi[val])

    return {
        'prefix_ids': prefix_ids,
        'target_feature': target_key,
        'aligned_tokens': aligned_tokens,
        'misaligned_tokens': misaligned_tokens,
        'rule_idx': rule_idx,
    }

def evaluate_probe_alignment(model, checkpoints, test_data_probes, RuleSystem, n_samples=25):
    """Evaluate probe alignment across all checkpoints for a set of probe categories.

    For each category in test_data_probes, converts sequences to last-token probes,
    then evaluates P(aligned) at each checkpoint.

    Args:
        model: Transformer model (will be moved to MPS).
        checkpoints: dict {checkpoint_id: state_dict}.
        test_data_probes: dict {rule_id: list of token sequences}.
        RuleSystem: RuleSet used to build probes.
        n_samples: Max number of probes to sample per category.

    Returns:
        dict {rule_id: pd.DataFrame} with columns [rule_idx, p_aligned, checkpoint].
    """
    model = model.to('mps')
    alignment_res = {k: [] for k in test_data_probes if len(test_data_probes[k]) > 0}

    for rule_id in alignment_res:
        probes = [tokens_to_last_token_probe(seq, RuleSystem) for seq in test_data_probes[rule_id]]
        if len(probes) == 0:
            continue

        probes = random.sample(probes, min(n_samples, len(probes)))
        blocks = []

        for checkpnt in tqdm(checkpoints):
            model.load_state_dict(checkpoints[checkpnt])
            model.eval()
            ev = evaluate_probes(model, probes, device='mps')
            evaldf = pd.DataFrame(ev)
            block_evals = evaldf.groupby('rule_idx').mean('p_aligned').reset_index()
            block_evals['checkpoint'] = checkpnt
            blocks.append(block_evals)

        alignment_res[rule_id] = pd.concat(blocks)

    return alignment_res


def evaluate_test_probes(model, histories, test_data_probes, RuleSystem, n_samples = 25):
    """Evaluate probe alignment across all runs and checkpoints for a probe category.

    Args:
        model: Transformer model (weights loaded from checkpoints).
        histories: dict {run_id: history_dict} where each history has a 'checkpoints' key.
        test_data_probes: dict {rule_id: list of token sequences} — one probe category
                          (e.g. test_data['single_rules']).
        RuleSystem: RuleSet used to build and evaluate probes.

    Returns:
        pd.DataFrame with columns [checkpoint, rule_idx, p_aligned, p_misaligned,
        p_ungrammatical, run, rule_id], aggregated across all runs.
    """

    all_run_results = []
    for run_i, run in enumerate(histories):
        print(f'## ---- Evaluating run {run} -----')
        history = histories[run]
        res = evaluate_probe_alignment(model, history['checkpoints'], test_data_probes, RuleSystem, n_samples=n_samples)
        for rule_id, df in res.items():
            df = df.copy()
            df['run'] = run_i
            df['rule_id'] = rule_id
            all_run_results.append(df)
    probe_df = pd.concat(all_run_results, ignore_index=True)
    return probe_df


def evaluate_all_metrics(
    model, histories, all_test_data_probes, RuleSystem,
    n_samples=25, n_gen=100, gen_batch_size=100, probe_batch_size=256, device='mps',
):
    """Evaluate all metrics in a single pass over checkpoints.

    Loads each checkpoint exactly once per run, then evaluates free generation
    quality and all probe categories at that checkpoint. Equivalent to calling
    evaluate_free_generations and evaluate_test_probes for each category
    separately, but far faster.

    Args:
        model: Transformer model (architecture only; weights loaded from checkpoints).
        histories: dict {run_id: history_dict} where each history has a 'checkpoints' key.
        all_test_data_probes: dict {metric_name: dict {rule_id: list of token sequences}}.
        RuleSystem: RuleSet used for grammar/alignment evaluation.
        n_samples: Max number of probes to sample per rule_id per metric.
        n_gen: Number of sequences to generate per checkpoint for generation eval.
        gen_batch_size: Batch size for generation.
        probe_batch_size: Number of probes per forward pass in evaluate_probes_batched.
        device: Device string.

    Returns:
        gen_df: pd.DataFrame with columns [run, checkpoint, n_ungrammatical, n_misaligned].
        rule_evals: dict {metric_name: pd.DataFrame} with columns
                    [rule_idx, p_aligned, p_misaligned, p_ungrammatical, checkpoint, run, rule_id].
    """
    model = model.to(device)

    # Pre-build all probes once before the checkpoint loop 
    all_probes = {}
    for metric_name, test_data_probes in all_test_data_probes.items():
        all_probes[metric_name] = {}
        for rule_id, seqs in test_data_probes.items():
            if len(seqs) == 0:
                continue
            sampled_seqs = random.sample(seqs, min(n_samples, len(seqs)))
            all_probes[metric_name][rule_id] = [tokens_to_last_token_probe(seq, RuleSystem) for seq in sampled_seqs]
    
    # Accumulators — one list entry per run, concatenated at the end.
    gen_records = []
    probe_run_dfs = {metric_name: [] for metric_name in all_probes}

    for run_i, run in enumerate(histories):
        print(f'## ---- Evaluating run {run} -----')
        history = histories[run]
        checkpoints = history['checkpoints']

        # Per-run accumulator: {metric_name: {rule_id: [block_df, ...]}}
        run_blocks = {
            metric_name: {rule_id: [] for rule_id in all_probes[metric_name]}
            for metric_name in all_probes
        }

        for checkpnt in tqdm(checkpoints, desc=f'Run {run_i}'):
            model.load_state_dict(checkpoints[checkpnt])
            model.eval()

            # Free generation quality
            quality = evaluate_generation_quality_batched(
                model, RuleSystem, n_sample=n_gen, device=device, batch_size=gen_batch_size
            )
            gen_records.append({
                'run': run_i,
                'checkpoint': checkpnt,
                'n_ungrammatical': quality['n_ungrammatical'],
                'n_misaligned': quality['n_misaligned'],
            })

            # All probe categories
            for metric_name, metric_probes in all_probes.items():
                for rule_id, probes in metric_probes.items():
                    ev = evaluate_probes_batched(
                        model, probes, device=device, batch_size=probe_batch_size
                    )
                    evaldf = pd.DataFrame(ev)
                    block_evals = evaldf.groupby('rule_idx').mean(numeric_only=True).reset_index()
                    block_evals['checkpoint'] = checkpnt
                    run_blocks[metric_name][rule_id].append(block_evals)

        # Concatenate blocks across checkpoints, then tag with run and rule_id
        for metric_name in all_probes:
            dfs = []
            for rule_id, blocks in run_blocks[metric_name].items():
                if not blocks:
                    continue
                df = pd.concat(blocks, ignore_index=True)
                df['run'] = run_i
                df['rule_id'] = rule_id
                dfs.append(df)
            if dfs:
                probe_run_dfs[metric_name].append(pd.concat(dfs, ignore_index=True))

    gen_df = pd.DataFrame(gen_records)
    rule_evals = {
        metric_name: pd.concat(dfs, ignore_index=True)
        for metric_name, dfs in probe_run_dfs.items()
        if dfs
    }
    return gen_df, rule_evals

