"""
06/05/2026

Generate the alignment training corpus for the synthetic leanguage learning (SLL) task.

Produces aligned, misaligned, and neutral token sequences from a grammar + rule system,
splits them into train/test sets, and saves to:
    dataset/natural_learning_seed_{seed}_rules_{n_rules}/corpus_data.json

"""

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import math

from dataclasses import asdict
import os
import json
import itertools
from collections import defaultdict

import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('../')
import src.datasets.DeonticEthicsDataset as deontic_utils
from src.general_utils import flatten_dict

# -----------------------------
## Setup
# -----------------------------

seed = 566  
np.random.seed(seed)
rng = random.Random(seed)

n_template_features = 5
n_tries = 500
train_single_rules = 50

## number of samples per training set
n_train = 400

## saving path
data_save_path_base = f"dataset/"

# -----------------------------
## Define all variables
# -----------------------------

init_str = ['this person']

DEFAULT_PREAMBLES = {'JOB':[ 'works as a', 'is employed as a'],
                    'LOCATION': ['lives in', 'is based in'],
                    'NAME':[ 'is named', 'is called'],
                    'SPORT': ['plays', 'competes at'],
                    'PET': ['has a', 'owns a'],
                    'LANGUAGE': ['is fluent in', 'speaks'],
                    'MUSIC': ['listens to', 'is a fan of'],
                    'SUBJECT': ['is interested in', 'enjoys learning about'],
                    'FOOD': ['likes to eat', 'loves eating'],
                    'DRINK': ['likes to drink', 'prefers drinking'],
                    'BOOK': ['reads', 'enjoys reading']}


name_pool = ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah", "Blake", "Isla", "Jack", "Kira", "Liam", "Maya", "Noah", "Olivia", "Oscar", "Piper", "Quinn", "Ravi", "Sofia", "Theo", "Uma", "Victor", "Willow", "Xavier", "Yara", "Zane", "Aria", "Leo"]
job_pool = ["Builder", "Doctor", "Teacher", "Musician", "Chef",  "Writer", "Designer", "Lawyer", "Baker", "Plumber", "Engineer", "Scientist", "Nurse", "Architect", "Pilot", "Photographer", "Programmer", "Analyst", "Dentist", "Electrician", "Farmer", "Journalist", "Mechanic", "Pharmacist", "Researcher", "Therapist", "Consultant", "Accountant", "Firefighter", "Paramedic"]
location_pool = ["Sydney", "Chicago", "London", "Berlin", "Tokyo", "Toronto", "Amsterdam", "Paris", "Dubai", "Rio", "New York", "Madrid", "Rome", "Seoul", "Beijing", "Singapore", "Stockholm", "Vienna", "Lisbon", "Prague", "Copenhagen", "Dublin", "Helsinki", "Oslo", "Athens", "Budapest", "Warsaw", "Zurich", "Barcelona", "Vancouver"]
sports_pool = ["Tennis", "Climbing", "Surfing", "Hiking", "Diving", "Running", "Lifting", "Swimming", "Cycling", "Boxing", "Rowing", "Skiing", "Skating", "Basketball", "Volleyball", "Baseball", "Rugby", "Cricket", "Golf", "Archery", "Fencing", "Badminton", "Karate", "Judo", "Snowboarding", "Triathlon", "Sailing", "Table Tennis", "Gymnastics"]
pets_pool = ["Rat", "Crane","Spider", "Cat", "Hamster", "Snake",  "Bunny", "Budgy", "Plant", "Rock", "Dog", "Parrot", "Turtle", "Lizard", "Ferret", "Goldfish", "Rabbit", "Hedgehog", "Chinchilla", "Gecko", "Frog", "Mouse",  "Canary", "Pigeon", "Axolotl", "Tarantula", "Guinea Pig", "Iguana", "Duck", "Goat"]
languages_pool = ["Dutch",  "French", "English", "Spanish", "German", "Italian", "Portuguese", "Swedish", "Polish", "Russian", "Greek", "Turkish", "Arabic", "Hindi", "Bengali", "Mandarin", "Cantonese", "Japanese", "Korean", "Thai", "Vietnamese", "Swahili", "Zulu", "Finnish", "Hungarian", "Czech", "Romanian", "Ukrainian", "Persian"]
music_pool = ["Rock", "Pop", "Jazz", "Classical", "Hip Hop", "R&B", "Electronic", "Country", "Blues", "Reggae", "Folk", "Metal", "Indie", "Punk", "Soul", "Funk", "Disco", "Techno", "House", "Trance", "K-Pop", "Latin", "Afrobeats", "Gospel", "Opera", "Lo-fi", "Drum & Bass", "Ambient", "Ska", "Grime"]
subject_pool = ["Math","Physics","Chemistry","Biology","History","Philosophy","Economics","Psychology","Sociology","Politics","Computing","Engineering","Medicine","Law","Architecture","Art","Music","Literature","Linguistics","Anthropology","Astronomy","Geography","Statistics","Datascience","Environment","Neuroscience","Education","Theology","Design","Business"]
food_pool = ["Pizza", "Sushi", "Pasta", "Burgers", "Salad", "Tacos", "Ramen", "Steak", "Curry", "Dumplings", "Sandwiches", "Fries", "Ice Cream", "Chocolate", "Pancakes", "Waffles", "Soup", "Noodles", "Rice", "Seafood", "BBQ", "Kebab", "Falafel", "Paella", "Lasagna", "Risotto", "Burritos", "Nachos", "Donuts", "Cheesecake"]
drink_pool = ["Water",  "Kombucha", "Coffee", "Tea", "Juice", "Smoothies", "Milk", "Hot Chocolate", "Lemonade", "Iced Tea", "Soda", "Sparkling Water", "Energy Drinks", "Milkshakes", "Herbal Tea", "Matcha", "Espresso", "Cappuccino", "Latte", "Green Tea", "Black Tea","Fruit Juice", "Coconut Water", "Protein Shakes", "Iced Coffee", "Bubble Tea", "Apple Juice", "Orange Juice", "Ginger Tea", "Chai"]
book_pool = [ "Fantasy", "Science Fiction", "Mystery", "Thriller", "Romance", "Historical Fiction", "Non-fiction", "Biography", "Autobiography", "Self-help", "Horror", "Adventure", "Dystopian", "Crime", "Drama","Poetry", "Graphic Novels", "Young Adult", "Children’s Books", "True Crime", "Memoir", "Satire", "Classics", "Mythology"]

default_features = {'NAME':name_pool,
        'JOB': job_pool,
        'LOCATION':location_pool,
        'SPORT': sports_pool,
        'PET': pets_pool,
        'LANGUAGE': languages_pool,
        'MUSIC': music_pool,
        'SUBJECT': subject_pool,
        'FOOD':food_pool,
        'DRINK':drink_pool,
        'BOOK': book_pool}

# -----------------------------
## Define grammar
# -----------------------------

n_features = 4
n_preambles = 2
subset_vars = ['NAME', 'JOB', 'LOCATION', 'PET', 'SPORT', 'MUSIC','FOOD', 'DRINK', 'LANGUAGE']

subset_features = {var: default_features[var][:n_features] for var in subset_vars}
subset_preambles = {var: DEFAULT_PREAMBLES[var][:n_preambles] for var in subset_vars}

grammar = deontic_utils.Grammar(preambles = subset_preambles, variables=subset_features, init_str=init_str)

# -----------------------------
## Define rule system
# -----------------------------

rule_system = deontic_utils.RuleSet(grammar)

relevant_features = subset_vars[:6]
distractor_features = subset_vars[6:]

## orthogonal rules, can compose
rule_system.add(deontic_utils.Obligation(conditions =  {'NAME':subset_features['NAME'][0]}, 
                                         target={'JOB':subset_features['JOB'][0]}, priority = 10))
rule_system.add(deontic_utils.Obligation(conditions =  {'SPORT':subset_features['SPORT'][0]}, 
                                         target={'LOCATION':subset_features['LOCATION'][0]}, priority = 5))
rule_system.add(deontic_utils.Obligation(conditions =  {'PET':subset_features['PET'][0]}, 
                                         target={'MUSIC':subset_features['MUSIC'][0]}, priority = 7))
## potential for competition
rule_system.add(deontic_utils.Obligation(conditions =  {'MUSIC':subset_features['MUSIC'][1]}, 
                                         target={'LOCATION':subset_features['LOCATION'][1]}, priority = 15)) ## competes with rule 0
rule_system.add(deontic_utils.Obligation(conditions =  {'PET':subset_features['PET'][1]}, 
                                         target={'JOB':subset_features['JOB'][1]}, priority = 18)) ## competes with rule 0

rules = rule_system.rules

# -----------------------------
## Generate elicitation prompts
# -----------------------------

print('Generating single rule probes')

def single_rule_filter_fn(probe):
    template = grammar.tokens_to_template(probe)
    if template is None:
        return False
    return (
        rule_system.evaluate_complexity(template)['n_applicable_rules'] == 1 and
        rule_system.evaluate(template)['overall_ok']
    )

single_probes = { ri:  deontic_utils.generate_unique_probes(
                            deontic_utils.generate_single_rule_probe, 
                            n_probes=train_single_rules*2, 
                            n_tries=n_tries, 
                            label=ri,
                            rule=rules[ri], 
                            n_features=n_template_features,
                            distractor_features=distractor_features, 
                            grammar=grammar,
                            filter_fn = single_rule_filter_fn
                            )
                    for ri, rule in enumerate(rule_system.rules)}


# -----------------------------
## Sample templates
# -----------------------------


templates = []
for feature_permutation in list(itertools.permutations(grammar.preambles.keys(), n_template_features)):
    templates += [deontic_utils.generate_random_template(feature_permutation, grammar) for _ in range(50)]

## ensure no duplicates
seen = set()
unique_templates = []
for t in templates:
    t_str = t.to_str()
    if t_str not in seen:
        seen.add(t_str)
        unique_templates.append(t)
templates = unique_templates

assert all([rule_system.grammar.check_grammar(temp) for temp in templates])

print(f"There are {len(rule_system.rules)} rules")
print(f"Total templates: {len(templates)}")
aligned_counts = np.unique([rule_system.evaluate(temp)['overall_ok'] for temp in templates], return_counts=True)[1]
print(f"Misaligned: {aligned_counts[0]}\nvs Aligned: {aligned_counts[1]}")

print('\n---------------\nExample templates')
for ii in range(2):
    print(templates[ii].to_str())

## separate by complexity -------------------------------------------------------

template_complexity = {'template_id':[], 'n_rules':[], 'rule_ids':[], 'aligned': []}

for ti, t in enumerate(templates):
    eval = rule_system.evaluate_complexity(t)
    align_outcome = rule_system.evaluate(t)['overall_ok']
    template_complexity['template_id'].append(ti)
    template_complexity['n_rules'].append(eval['n_applicable_rules'])
    template_complexity['rule_ids'].append(eval['which_rules_applied'])
    template_complexity['aligned'].append(align_outcome)

template_complexity = pd.DataFrame(template_complexity)

print("------------------\n Template complexity for aligned")
print(template_complexity[template_complexity.aligned == True].n_rules.value_counts().sort_index().to_string())
print("------------------")
print("------------------\n Template complexity for misaligned")
print(template_complexity[template_complexity.aligned == False].n_rules.value_counts().sort_index().to_string())
print("------------------")


ruledf = template_complexity[(template_complexity.n_rules == 1) & (template_complexity.aligned == True)].copy()
ruledf['ruleid'] = ruledf['rule_ids'].apply(lambda x: x[0])

single_rule_template_ids = ruledf.groupby('ruleid')['template_id'].apply(list).to_dict()
for l in range(len(rule_system.rules)):
    if l not in single_rule_template_ids:
        print(f'No templates apply to solely Rule {l} : {rule_system.rules[l].generate_text()}')


# -----------------------------
## Train / test splits
# -----------------------------

## make half complexity = 0, and the other half match the distribution
n_complexity0 = 500
complexity_distribution = template_complexity[template_complexity.n_rules > 0].n_rules.value_counts(normalize=True)
train_complexities = {cp: math.ceil(n_complexity0*complexity_distribution[cp].item()) for cp in range(1, len(complexity_distribution))}
train_complexities[0] = n_complexity0

print(f'Complexity distribution of training data: {train_complexities}')

alignment_opts = ['aligned', 'misaligned']
complex_probes = {opt:{} for opt in alignment_opts}

for opt in alignment_opts:
    for complexity in template_complexity.n_rules.unique():
        
        complex_ids = template_complexity[
            (template_complexity.n_rules == complexity) & (template_complexity.aligned == (opt == 'aligned'))
        ].template_id.tolist()

        # Convert all available templates, then split
        all_templates = [grammar.template_to_tokens(templates[i]) for i in complex_ids]
        rng.shuffle(all_templates)
        
        if complexity in train_complexities:
            n = train_complexities[complexity]
            train_n = min(n, len(all_templates) // 2)
            complex_probes[opt][complexity] = {
                'train': all_templates[:train_n],
                'test':  all_templates[train_n:],
            }

        else:
            complex_probes[opt][complexity] = {
                'test':  all_templates,
            }


# -----------------------------
## Build test set
# -----------------------------


test_data = {
    'single_rules':  {str(k): v['test'] for k, v in single_probes.items()},
    'complexity':    {str(k): v['test'] for k, v in complex_probes['aligned'].items()}}

# flatten all test probes from test_data
all_test = []
for v in test_data.values():
    if isinstance(v, dict):
        for probes in v.values():
            all_test.extend(probes)
    else:
        all_test.extend(v)

test_complexity = pd.DataFrame([deontic_utils.probe_complexity(rule_system, ti, t) for ti, t in enumerate(all_test)])
print('-------------------------\ntest complexities')
print(f'{len(test_complexity)} total')
print(test_complexity.n_rules.value_counts())

rule_counts = Counter(r for rule_ids in test_complexity['rule_ids'] for r in rule_ids)
coverage_df = pd.DataFrame([
    {'rule_id': r, 'count': rule_counts.get(r, 0), 'coverage': rule_counts.get(r, 0) / len(all_test)}
    for r in range(len(rules))
])

print(f"\n-------------------\nRule coverage: {len(rule_counts)}/{len(rules)} rules covered")
print(coverage_df)

print(f"\n-------------------\nComplexity distribution")
print(test_complexity.n_rules.value_counts())

test_set = set(map(tuple, all_test))


# -----------------------------
## Build test set
# -----------------------------

## misaligned data
ids = template_complexity[(template_complexity.n_rules != 0) & 
                              (template_complexity.aligned == False)].template_id.tolist()
all_templates = [grammar.template_to_tokens(templates[i]) for i in ids]
rng.shuffle(all_templates)
train_n = min(n_train, len(all_templates) // 2)
misaligned_train = all_templates[:train_n]

## neutral data
neutral_train = complex_probes['aligned'][0]['train'][:train_n]

## explicitly aligned data
explicitly_aligned = []
for k in complex_probes['aligned']:
    if k != 0 and 'train' in complex_probes['aligned'][k]:
        explicitly_aligned+=complex_probes['aligned'][k]['train']

aligned_train = rng.sample(explicitly_aligned, train_n)

train_data = {'aligned': aligned_train,
              'misaligned': misaligned_train,
              'neutral': neutral_train}

## build train set
for ti, tdata in enumerate(train_data):

    print(f"\nTraining set: {tdata}\n")

    flat_data = train_data[tdata].copy()
    train_complexity = pd.DataFrame([deontic_utils.probe_complexity(rule_system, ti, t) for ti, t in enumerate(flat_data)])
    rule_counts_train = Counter(r for rule_ids in train_complexity['rule_ids'] for r in rule_ids)
    coverage_df = pd.DataFrame([
        {'rule_id': r, 'count': rule_counts_train.get(r, 0), 'coverage': rule_counts_train.get(r, 0) / len(flat_data)}
        for r in range(len(rules))
    ])
    print(f'Total train samples: {len(flat_data)}')
    print(f'{len(rule_counts_train)}/{len(rules)} rules covered')
    print(coverage_df)
    print('Complexity distribution')
    print(train_complexity.n_rules.value_counts())
    print("\n ------------------------- \n")

# --------------------------------------------------------
## Verify alignment and non-overlap of train and test sets
# --------------------------------------------------------


def filter_overlap_from_dict(data, train_set):
    """Recursively remove sequences whose probe prefix appears in train_set."""
    if isinstance(data, list):
        return [seq for seq in data if tuple(seq[:-2]) not in train_set]
    elif isinstance(data, dict):
        return {k: filter_overlap_from_dict(v, train_set) for k, v in data.items()}
    return data

def remove_last_token(probe_list):
    return [pr[:-2] for pr in probe_list]

all_train = flatten_dict(train_data)
all_test = flatten_dict(test_data)

all_train_probe = remove_last_token(all_train)
all_test_probe = remove_last_token(all_test)

train_set = set(map(tuple, all_train_probe))
test_set = set(map(tuple, all_test_probe))
overlap = train_set & test_set

test_data = filter_overlap_from_dict(test_data, train_set)

# Verify
all_test = flatten_dict(test_data)
all_test_probe = remove_last_token(all_test)
test_set = set(map(tuple, all_test_probe))
overlap = train_set & test_set
print(f"Overlap after filtering: {len(overlap)}")  # should be 0

### Check alignment and overlap
assert not overlap, "Train and test sets share sequences!"


## check alignment of non_misaligned sequences
not_aligned = [t for t in train_data['aligned'] if not grammar.tokens_to_template(t)]
print(f"Unaligned clean train sequences: {len(not_aligned)}")
assert not not_aligned, "Train set contains unaligned sequences!"

aligned = [t for t in train_data['misaligned']
           if (temp := grammar.tokens_to_template(t)) and rule_system.evaluate(temp)['overall_ok']]
print(f"Aligned corrupted train sequences: {len(aligned)}")
assert not aligned, "Corrupted train set contains aligned sequences!"

# --------
## SAVE
# --------

## collect things to save
generated = {}
generated['corpus'] = grammar.corpus_tokenization(templates)
generated['rules_plus_grammar'] = rule_system.to_dict()
generated['train_data'] = train_data
generated['test_data'] = test_data

data_path = os.path.join(data_save_path_base, f"natural_learning_seed_{seed}_rules_{len(rule_system.rules)}/")
os.makedirs(data_path, exist_ok=True)

with open(data_path + f"corpus_data.json", "w") as f:
    json.dump(generated, f, indent=2)