import numpy as np
import random
import pandas as pd
from collections import defaultdict
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
import string
import itertools
import torch
from alignment_datasets.DatasetBuildingBlocks import CombinableDataset
from collections import Counter

#############################
## HELPERS
#############################

def is_neg(s):
    """ is the value negated, i.e. starts with !
        Returns Bool """
    return isinstance(s, str) and s.startswith('!')

def neg_val(s):
    """ reterns (non-negated) value """
    return s[1:] if is_neg(s) else s

def obligation_target_val(rule):
    """Return (is_negative, value) for the single target."""
    assert len(rule.target) == 1
    v = next(iter(rule.target.values()))
    return is_neg(v), neg_val(v) if is_neg(v) else v


################################
### LANGUAGE GRAMMAR
################################

@dataclass
class Template:
    features: Dict[str, str] # e.g. {'JOB' : 'Doctor', 'NAME': Amanda}
    preambles: Dict[str, str] # e.g. {'JOB': 'is employed as a', 'NAME': 'is named'}
    init_str: str = field(default_factory = lambda: 'this person'),

    """ Template structure for descriptions
    Args:
        features: Dict of variables: list of possible values
        preambles: Dict of variables : list of preambles
                    Preamble refers to text preceeding variable value e.g. 'lives in' precedes JOB
    """

    def __post_init__(self):
        # All features must have a preamble
        missing = [k for k in self.features if k not in self.preambles]
        if missing:
            raise ValueError(f"Missing preambles for features: {missing}")
        # No extra preambles
        extra = [k for k in self.preambles if k not in self.features]
        if extra:
            raise ValueError(f"Unused preambles for features: {extra}")
            
    def to_sequence(self) -> List[str]:
        template_seq = [self.init_str]
        for key, val in list(self.features.items()):
            template_seq.append(self.preambles[key])
            template_seq.append(val)
        return template_seq
    
    def to_str(self) -> str:
        """Return human readable punctuated sentence, with commas and 'and'."""
        items = list(self.features.items())
        clauses = [f"{self.preambles[k].strip()} {v}" for k, v in items]

        if len(clauses) == 1:
            body = clauses[0]
        elif len(clauses) == 2:
            body = f"{clauses[0]} and {clauses[1]}"
        else:
            body = ", ".join(clauses[:-1]) + ", and " + clauses[-1]

        return f"{self.init_str.strip()} {body}."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Template to JSON-serializable dict."""
        return {
            'features': self.features,
            'preambles': self.preambles,
            'init_str': self.init_str
            }
    
    @classmethod
    def from_dict(cls, d):
        # generate grammar from dict
        init_fields = {
            "preambles": d.get("preambles", {}),
            "features": d.get("features", {}),
            "init_str": d.get("init_str",  ['this person']),
        }
        return cls(**init_fields)
    
@dataclass
class Grammar:
    preambles: Dict[str, List[str]]  = field(default_factory=dict)
    variables: Dict[str, List[str]]  = field(default_factory=dict)
    init_str: List[str] = field(default_factory=lambda: ['this person'])
    ## tokens
    specials: List[str] = field(default_factory = lambda: ["<pad>", "<bos>", "<eos>"]) ## special tokens

    # caches
    _all_preambles: List[str] = field(init=False, default_factory=list)
    _preambles_to_key: Dict[str, str] = field(init=False, default_factory=dict)
    _all_variables: List[str] = field(init=False, default_factory=list)
    _variables_to_key: Dict[str, str] = field(init=False, default_factory=dict)
    variable_names: List[str] = field(init=False, default_factory=list)

    def __post_init__(self):
        # sanity check keys
        pre_keys = set(self.preambles.keys())
        var_keys = set(self.variables.keys())
        if pre_keys != var_keys:
            raise ValueError("Inconsistent keys between preambles and variables")

        self.variable_names = sorted(var_keys)

        def build_sorted_map(d: Dict[str, List[str]]) -> Tuple[List[str], Dict[str, str]]:
            phrases: List[str] = []
            mapping: Dict[str, str] = {}
            for key, lst in d.items():
                for p in lst:
                    if p in mapping:
                        raise ValueError(f"Duplicate phrase '{p}' found in both " f"'{mapping[p]}' and '{key}'")
                    phrases.append(p)
                    mapping[p] = key
            return phrases, mapping

        self._all_variables, self._variables_to_key = build_sorted_map(self.variables)
        self._all_preambles, self._preambles_to_key = build_sorted_map(self.preambles)
        
        ## for tokenization
        parts = []
        parts.append(list(self.specials))
        parts.append(list(self.init_str))
        parts.append(list(self._all_variables))
        parts.append(list(self._all_preambles))

        # flatten
        all_parts = [p for group in parts for p in group]
        all_parts = sorted(all_parts)

        # check duplicates
        if len(all_parts) != len(set(all_parts)):
            raise ValueError("Duplicate phrases found across vocabulary categories")

        #### TOKENIZATION
        self.itos = all_parts  # idx -> str
        self.stoi = {c: i for i, c in enumerate(self.itos)}       # str -> idx
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.pad_id = self.stoi["<pad>"]
        self.vocab_size = len(self.itos)

    def to_dict(self):
        return {
            "preambles": self.preambles,
            "variables": self.variables,
            "init_str": self.init_str}

    @classmethod
    def from_dict(cls, d):
        # generate grammar from dict
        init_fields = {
            "preambles": d.get("preambles", {}),
            "variables": d.get("variables", {}),
            "init_str": d.get("init_str",  ['this person']),
        }
        return cls(**init_fields)
    
    def check_grammar(self, template:Template):
        ## CHANGE! value should correspond to correct key TKTKTKTK
        ## 1. check init_str
        if template.init_str not in self.init_str:
            print(f'{template.init_str} not in {self.init_str}')
            return False
        
        ## 2. checks preambles
        template_preambles = template.preambles
        # all var names in template are present in grammar
        if not template_preambles.keys() <= self.preambles.keys(): 
            return False
        # all preamble values in template are present in grammar for the given var
        for key in template_preambles:
            if not template_preambles[key] in self.preambles[key]:
                return False

        ## 3. check feature variables
        template_variables = template.features
        # all var names in template are present in grammar
        if not template_variables.keys() <= self.variables.keys(): 
            return False
        # all values in template  are present in grammar
        for key in template_variables:
            if not template_variables[key] in self.variables[key]:
                return False

        return True
    
    ###-----------
    ## TOKENIZATION
    ###-----------

    def template_to_tokens(self, template: Template) -> List[int]:
            sequence = template.to_sequence()
            return [self.bos_id] + [self.stoi[p] for p in sequence] + [self.eos_id]
    
    def corpus_tokenization(self, templates = List[Template]):
        """ covert list of templates into a list of tokenized templates (tokenized template -> list of ints)"""
        return [self.template_to_tokens(t) for t in templates]
    
    def tokens_to_template(self, token_sequence):
        """ check if grammatically correct, if so return template"""
        
        if isinstance(token_sequence, torch.Tensor):
            token_sequence = token_sequence.view(-1).tolist()
        else:
            token_sequence = np.array(token_sequence).reshape(-1).tolist()

        # 1. starts with bos and ends with eos
        if token_sequence[0] != self.stoi['<bos>'] or (token_sequence[-1] != self.stoi['<eos>']):
            return False
        # 2. starts with an appropriate init_str
        init_toks = [self.stoi[init] for init in self.init_str]
        if token_sequence[1] not in init_toks:
            return False 
        # 3. Check pairs of preambles and features
        pairs = token_sequence[2:-1]
        if (len(pairs) < 1) or (len(pairs) % 2 != 0):
            return False
        
        features_map = {}
        preambles_map = {}

        preamble_feature_pairs = [(pairs[2*ii], pairs[2*ii +1]) for ii in range(len(pairs)//2)]
        used_keys =[]
        for preamb_tok, feat_tok in preamble_feature_pairs:
            # check preamble and feature are valid
            preamb = self.itos[preamb_tok]
            feat = self.itos[feat_tok]

            if preamb not in self._all_preambles or feat not in self._all_variables:
                return False
            combo_key = self._preambles_to_key[preamb]
            ## incorrect value for feat
            if feat not in self.variables[combo_key]:
                return False
            ## repeat a features
            if combo_key in used_keys:
                return False
            used_keys.append(combo_key)

            # record in template as str
            features_map[combo_key] = feat
            preambles_map[combo_key] = preamb
            
        return Template(init_str=self.itos[token_sequence[1]], preambles=preambles_map, features=features_map)
        
    def tokens_to_str(self, token_sequence) -> str: 
        """ convert tokens to strings for human readability"""
        if isinstance(token_sequence, torch.Tensor):
            token_sequence = token_sequence.view(-1).tolist()
        else:
            token_sequence = np.array(token_sequence).reshape(-1).tolist()

        str_sequence = [self.itos[tok] for tok in token_sequence]
        return ' '.join(str_sequence)
    
def extract_generated_sentence(gen_seq, stoi):
    pred_seq = gen_seq.reshape(-1)
    # Find positions of EOS tokens
    eos_mask = (pred_seq == stoi['<eos>'])
    
    if not eos_mask.any():
        return pred_seq  # No EOS found, return full tensor
    
    # Get index of first EOS token
    first_eos_idx = eos_mask.nonzero(as_tuple=True)[0][0]
    
    # Slice up to and including EOS
    return pred_seq[:first_eos_idx + 1]

def generate_random_template(feature_list:List,grammar:Grammar):

    """ Generates a random template
    Args:
        vars: Dict of variables and list of values
        features: List of ordered variables to include
        preambles: Dict of variables and list of possible preambles """
    
    select_preambles = {key:random.choice(grammar.preambles[key]) for key in feature_list}
    select_vars = {key:random.choice(grammar.variables[key]) for key in feature_list}
    select_init_str = random.choice(grammar.init_str)

    return Template(features = select_vars, preambles = select_preambles, init_str=select_init_str)

########################################
### RULES ------------------------------
########################################

class RuleType(Enum):
    OBLIGATION = 'O'
    PERMISSION = 'P'

@dataclass
class Rule:
    conditions: Dict[str, str]          # e.g. {'JOB': 'Doctor', 'LOCATION': '!London'}
    target: Dict[str, str]              # single-target assumed e.g. {'SPORT': 'tennis'} or {'SPORT': '!tennis'}
    priority: int                       # numerical rank indicates importance

    def __post_init__(self):
        if len(self.target) != 1:
            raise ValueError(
                f"Rule.target must contain exactly one key-value pair, got {len(self.target)}"
            )
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conditions": self.conditions,
            "target": self.target,
            "priority": self.priority,
            "rtype": self.rtype.value,
            "__class__": self.__class__.__name__,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Rule":
        # Dispatch to the correct subclass
        class_name = d.get("__class__", "Rule")
        
        if class_name == "Obligation":
            target_cls = Obligation
        elif class_name == "Permission":
            target_cls = Permission
        else:
            target_cls = cls  # fallback to Rule or whatever cls is
        
        # Filter out fields that shouldn't go to __init__
        return target_cls(
            conditions=d["conditions"],
            target=d["target"],
            priority=d["priority"],
        )
    
    def applies_to(self, template: Template) -> bool:
        """A rule applies to a template if all conditions match (negation supported).
        AND target feature is in the template. """
        
        t_feats = template.features
        for var, cond_val in self.conditions.items():
            if var not in t_feats: # variable must be in template
                return False
            tval = t_feats[var]
            if is_neg(cond_val):
                if neg_val(cond_val) == tval: # if rule val is negated, template val must != rule val
                    return False
            else:
                if cond_val != tval:
                    return False
        
        # target must be in template
        for var, target_val in self.target.items():
            if var not in t_feats: 
                return False
            
        return True

    def target_key(self) -> str:
        assert len(self.target) == 1
        return next(iter(self.target.keys()))

    def target_val(self) -> str:
        assert len(self.target) == 1
        return next(iter(self.target.values()))

    def evaluate_on(self, template: Template) -> bool:
        """Return True if the template satisfies the rule's target (i.e., obligation met or permission matches)."""
        tkey = self.target_key()
        if tkey not in template.features:
            # If the target is absent from the template it's treated as not satisfied.
            return False
        tval = template.features[tkey]
        target_value = self.target_val()
        if is_neg(target_value):
            return tval != neg_val(target_value)
        else:
            return tval == target_value
    
    def difficulty(self) -> int:
        """ difficulty is defined as number of conditions """
        return len(self.conditions)

@dataclass
class Obligation(Rule):
    rtype: RuleType = field(init=False, default=RuleType.OBLIGATION)

    def generate_text(self) -> str:
        conditions_str = ' and '.join(f'{k} is {v}' for k, v in self.conditions.items())
        target_key, target_val = next(iter(self.target.items()))
        return f'if {conditions_str} then {target_key} MUST be {target_val}  (priority {self.priority})'

@dataclass
class Permission(Rule):
    rtype: RuleType = field(init=False, default=RuleType.PERMISSION)

    def generate_text(self) -> str:
        rule_str = ['if ']
        for k, v in self.conditions.items():
            rule_str.append(k + ' is ' + v)
        target_key, target_val = next(iter(self.target.items()))
        rule_str.append(' then ' + target_key + ' MAY be ' + target_val)
        rule_str.append(f'  (priority {self.priority})')

        return ''.join(rule_str)
    
## RULE SET ----------------------------------------
@dataclass
class RuleSet:
    grammar: Grammar
    rules: List[Rule] = field(default_factory=list)

    def __post_init__(self):
        priorities = [r.priority for r in self.rules]
        duplicates = {p for p in priorities if priorities.count(p) > 1}
        if duplicates:
            raise ValueError(f"Rules share duplicate priorities: {duplicates}")

    # ---------- basic accessors ----------

    def add(self, rule: Rule):
        existing = {r.priority for r in self.rules}
        if rule.priority in existing:
            raise ValueError(f"Rule with priority {rule.priority} already exists")
        self.rules.append(rule)

    def obligations(self) -> List[Obligation]:
        return [r for r in self.rules if r.rtype == RuleType.OBLIGATION]

    def permissions(self) -> List[Permission]:
        return [r for r in self.rules if r.rtype == RuleType.PERMISSION]

    # ---------- helper methods  ----------

    def to_dict(self):
        return {
            "grammar": self.grammar.to_dict(),
            "rules": [r.to_dict() for r in self.rules],
            "__class__": self.__class__.__name__,
            }

    @classmethod
    def from_dict(cls, d):
        grammar = Grammar.from_dict(d["grammar"])
        rules = [Rule.from_dict(rd) for rd in d["rules"]]
        return cls(grammar=grammar, rules=rules)

    def _obligation_target_val(self, rule: Rule) -> Tuple[bool, str]:
        """Return (is_negated, value) for rule target."""
        v = rule.target_val()
        return is_neg(v), (neg_val(v) if is_neg(v) else v)

    def _conflicts(self, new_rule: Rule, kept_rules: List[Rule]) -> bool:
        """Return True if new_rule conflicts with any rule in kept_rules."""
        new_tkey = new_rule.target_key()
        new_neg, new_val = self._obligation_target_val(new_rule)

        for r in kept_rules:
            if self._individual_conflicts(r, new_tkey=new_tkey, new_neg=new_neg, new_val=new_val):
                return True

        return False
    
    def _individual_conflicts(self, r: Rule, new_tkey, new_neg, new_val):
        if r.target_key() != new_tkey: # old and new rule do not share targets, so do not conflict
            return False
        r_neg, r_val = self._obligation_target_val(r)

        # whether rules actually conflict depends on pattern of negations
        if (not new_neg and not r_neg and new_val != r_val):  # neither rule is negated and they require different values, e.g. target: doctor and target: builder conflict
            return True  # -> conflict
        if (not new_neg and r_neg and new_val == r_val):  # one rule is negated and they talk about the same value, e.g. target: doctor and target: !doctor conflict
            return True  # -> conflict
        if (new_neg and not r_neg and new_val == r_val):  # either way around from above case
            return True  # -> conflict
        return False

    def _select_obligations(self, obligations: List[Obligation]) -> List[Obligation]:
        """
        Keep all obligations unless a lower-priority one conflicts with
        a higher-priority kept one. Also block lower-priority negations
        of protected condition-values.
        TODO: what is a protected condition-value? This part of the code is opaque
        """
        if not obligations:
            return []

        # higher priority first
        sorted_obs = sorted(obligations, key=lambda o: o.priority, reverse=True)

        kept: List[Obligation] = []
        protected = set()

        for ob in sorted_obs:
            if self._conflicts(ob, kept):
                continue

            tkey = ob.target_key()
            tval = ob.target_val()

            if is_neg(tval) and (tkey, neg_val(tval)) in protected:
                continue

            kept.append(ob)

            for ck, cv in ob.conditions.items():
                if not is_neg(cv):
                    protected.add((ck, cv))

        return kept

    def _find_applicable_permissions_for(
        self,
        violated_obligation: Obligation,
        template: Template
    ) -> List[Permission]:
        ob_target_key = violated_obligation.target_key()
        perms = []

        for p in self.permissions():
            if p.priority <= violated_obligation.priority:
                continue
            if not p.applies_to(template):
                continue
            if ob_target_key not in p.target:
                continue
            perms.append(p)

        return perms

    def _assess_permissability(
        self,
        template:Template,
        violated_obligation: Obligation,
        permission: Permission
    ) -> bool:
        
        assert violated_obligation.target_key() == permission.target_key(), 'Permission and obligation do not refer to same target'

        perm_target = permission.target_val()
        template_target = template.features[permission.target_key()]

        if is_neg(perm_target):
            return template_target != neg_val(perm_target)
        else:
            return perm_target == template_target

    # ---------- main evaluation ----------

    def evaluate(self, 
                 template: Template, 
                 verbose: bool = False) -> Dict[str, Any]:
        
        if verbose:
            print(template)
        
        if not self.grammar.check_grammar(template):
            print('Grammatically incorrect')
            return None
        
        applicable_obs = [
            r for r in self.rules
            if r.rtype == RuleType.OBLIGATION and r.applies_to(template)
        ]

        condition_difficulty = sum(o.difficulty() for o in applicable_obs)
        consistent_obs = self._select_obligations(applicable_obs)

        violated: List[Obligation] = []
        permitted: List[Obligation] = []
        permitted_count = 0

        for ob in consistent_obs:
            if not ob.evaluate_on(template):
                perms = self._find_applicable_permissions_for(ob, template)
                perm_ok = any(self._assess_permissability(template, ob, p) for p in perms)

                if perm_ok:
                    permitted.append(ob)
                    permitted_count += 1
                else:
                    violated.append(ob)

        result = {
            "features": template.features,
            "violated_obligations": violated,
            "permitted_violations": permitted,
            "overall_ok": len(violated) == 0,
            "difficulty": condition_difficulty + permitted_count,
        }

        if verbose:
            print("Template:", template.to_str())
            print("Applicable obligations:", len(consistent_obs))
            print("Violated:", len(violated), "Permitted:", len(permitted))

        return result
    
    def evaluate_complexity(self, template: Template) -> Dict[str, Any]:
        # determine which rule complexities are at play for the given template. We collect:
        # - Compositions
        # - Competitions
        # - Exceptions
        #
        # Should Specifications be an extra thing? Two rules don't compete directly but one demands higher specificity (i.e. the first is a negation)? In that case, we can only be sure that the more specific rule got applied
        # Are two rules with the same target but both are negated a composition? Or does only the higher level rule apply? Composition for now.
        # TODO: not yet set up for permissions

        if not self.grammar.check_grammar(template):
            print('Grammatically incorrect')
            return None
         
        applicable_obs = [
            r for r in self.rules
            if r.rtype == RuleType.OBLIGATION and r.applies_to(template)
            ]
         
        applicable_perms = [
            r for r in self.rules
            if r.rtype == RuleType.PERMISSION and r.applies_to(template)
            ]
        assert len(applicable_perms) == 0, "Permission complexity not coded yet"
         
        sorted_obs = sorted(applicable_obs, key=lambda o: o.priority, reverse=True)

        kept: List[Obligation] = []
        protected = set()

        used_rules = []
        n_exceptions = 0
        n_competitions = 0

        # Collect rules that end up applying, and see what kind of clashes occur
        # we only look at conflicts between rules and kept rules, i.e. two beat rules are not considered for the complexity
        for ob in sorted_obs:
            tkey = ob.target_key()
            tval = ob.target_val()
            new_neg, new_val = self._obligation_target_val(ob)

            # unsure whether this goes here, TODO
            if is_neg(tval) and (tkey, neg_val(tval)) in protected:
                continue

            keep_this = True
            for r in kept:
                # any conflicts?
                if not self._individual_conflicts(r, new_tkey=tkey, new_neg=new_neg, new_val=new_val):
                    continue

                # is it a competition or an exception?
                # we only consider it an exception if the higher prio rule is strictly more specific
                if len(ob.conditions.keys()) < len(r.conditions.keys()) and all([ck in r.conditions.keys() for ck in ob.conditions.keys()]):
                    n_exceptions += 1
                else:
                    n_competitions += 1

                keep_this = False
                break
            
            if not keep_this:
                continue

            kept.append(ob)
            used_rules.append(self.rules.index(ob))

            for ck, cv in ob.conditions.items():
                if not is_neg(cv):
                    protected.add((ck, cv))

        # 1 kept rule means no compositions. Any larger number of rules, there are that many compositions of rules (not counting all possible pairs)
        n_compositions = max(len(kept) - 1, 0)

         
        complexity_properties = {
            "n_features": len(template.features),
            "n_applicable_rules": len(applicable_obs) + len(applicable_perms),  # total number of obligations and permissions that have their condition fulfilled, irrespective of later overwrites
            "which_rules_applied": used_rules,  # which rules were ultimately used to determine whether sentence was good or not? Useful to check rule coverage given a set of templates
            "n_compositions": n_compositions,
            "n_competitions": n_competitions,
            "n_exceptions": n_exceptions
            }
         
        return complexity_properties

#############################
## RULE GENERATION
#############################

#generate random rule
def generate_obligation(vars, n_conds = 1, neg_prob = 0.1):
    # vars is a dictionary of features to list of possible values

    var_names = list(vars.keys())
    select_vars = random.sample(var_names, 1 + n_conds)

    target_var = select_vars[0]
    target_value = random.choice(vars[target_var])
    if np.random.binomial(1, neg_prob):
        target_value = '!'+ target_value
    target = {target_var: target_value}

    conditions = {}
    for ci in range(n_conds):
        cond_var = select_vars[1+ci]
        cond_value = random.sample(vars[cond_var], 1)[0]
        if np.random.binomial(1, neg_prob):
            cond_value = '!' + cond_value
        conditions[cond_var] = cond_value
    
    priority = random.randint(0, 1000)

    return Obligation(conditions=conditions, target = target, priority=priority)

def generate_permission(obligation, vars):

    ob_target_key, ob_target_val = obligation.target_key(), obligation.target_val()
    
    ob_conditions = obligation.conditions
    assert len(ob_conditions) == 1, 'Too many conditions in obligation'
    ob_keys = list(ob_conditions.keys())
    selected_vars = ob_keys + [ob_target_key]
    all_vars = list(vars.keys())
    remaining_vars = [var for var in all_vars if var not in selected_vars]

    exception_cond = random.choice(remaining_vars)
    perm_conditions = ob_conditions.copy()
    perm_conditions[exception_cond] = random.choice(vars[exception_cond])

    if is_neg(ob_target_val):
        perm_target = neg_val(ob_target_val)
    else:
        perm_target = '!' + ob_target_val
    
    ob_priority = obligation.priority
    priority = ob_priority + random.randint(0, 1000 - ob_priority)
    
    return Permission(conditions=perm_conditions, target = {ob_target_key: perm_target}, priority=priority)

########################################
### DATASET ----------------------------
########################################

class TemplateDataset(CombinableDataset):
    def __init__(self, texts, name='', max_len=None):

        self.items = texts
        if max_len is not None:
            self.items = [it[:max_len] if len(it)>max_len else it for it in self.items]

        super().__init__(name=name, length=len(self.items), metadata={"mode": "completion"})

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        seq = self.items[idx]
        input_ids = seq[:-1]   # input tokens
        labels = seq[1:]    # target tokens (shifted by 2)
        return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
    
    def get_texts(self):
        return self.items
    def get_input(self, idx):
        return self.items[idx][:-1] 
    def get_label(self, idx):
        return self.items[idx][1:] 
    def get_string(self, idx):
        return self.items[idx]
    
def collate_fn(batch, PAD_ID=0, LABEL_PAD_REPLACE = -100):
    """
    batch: list of tuples (input_ids_tensor, labels_tensor)
    each tensor shape: (seq_len,)
    Returns:
       input_ids_padded (B, L),
       labels_padded     (B, L),
       attention_mask    (B, L)
    """
    inputs = [item[0].long() for item in batch]
    labels = [item[1].long() for item in batch]

    # pad inputs and labels to same max length
    input_ids_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_ID)  # (B, L)
    labels_padded     = pad_sequence(labels, batch_first=True, padding_value=PAD_ID)  # (B, L)
    labels_for_loss = labels_padded.clone()
    labels_for_loss[labels_for_loss == PAD_ID] = LABEL_PAD_REPLACE

    attention_mask = (input_ids_padded != PAD_ID).long()  # 1 for real tokens

    return {
        "input_ids": input_ids_padded,
        "labels": labels_for_loss,
        "attention_mask": attention_mask
        }

############################################
## Generating templates
############################################


def generate_template_for_rule(rule_sys, rule_idx, aligned = True, seed=42):
    """
    Generates n_templates unique templates applicable to a given rule_idx
    """
    rng = random.Random(seed)

    grammar = rule_sys.grammar
    features = grammar.variables
    preambles = grammar.preambles
    init_str = grammar.init_str
    
    rule = rule_sys.rules[rule_idx]
    target_key, target_val = next(iter(rule.target.items()))

    template_features = {}
    template_preambles = {}

    ## template fulfils condition of rule
    for cond_key, cond_val in rule.conditions.items():
        if is_neg(cond_val):
            valid = [v for v in features[cond_key] if v != neg_val(cond_val)]
            template_features[cond_key] = rng.choice(valid)
        else:
            template_features[cond_key] = cond_val
        template_preambles[cond_key] = rng.choice(preambles[cond_key])
        
    # Optionally add 0-2 extra features (not target, not condition)
    other_keys = [k for k in features if k != target_key and k not in rule.conditions]
    n_extra = rng.randint(0, min(2, len(other_keys)))
    for k in rng.sample(other_keys, n_extra):
        template_features[k] = rng.choice(features[k])
        template_preambles[k] = rng.choice(preambles[k])

    if (is_neg(target_val) and aligned) or ( not is_neg(target_val) and not aligned):
        # if aligned and target is neg OR if misaligned and target not neg -> then choose target value
        valid = [v for v in features[target_key] if v != neg_val(target_val)]
        template_features[target_key] = rng.choice(valid)
    else:
        # if aligned and target is not neg OR if misaligned and target is neg -> then choose alternative to target value
        template_features[target_key] = target_val
    template_preambles[target_key] = rng.choice(preambles[target_key])
    
    gen_template = Template(init_str = rng.choice(init_str),
                    preambles = template_preambles,
                    features = template_features)
    
    assert rule_sys.evaluate(gen_template)['overall_ok'] == aligned, "template does not match desired alignment"

    return gen_template

############################################
## Generating datasets and probes
############################################


def train_test_split_list(xs, train_frac=0.8, seed=None):
    xs = list(xs)  # copy
    rng = random.Random(seed)
    rng.shuffle(xs)

    n_train = int(train_frac * len(xs))
    train = xs[:n_train]
    test = xs[n_train:]
    return train, test

def probe_complexity(rule_system, ti, t):
    temp = rule_system.grammar.tokens_to_template(t)
    evl = rule_system.evaluate_complexity(temp)
    return {
        'template_id': ti,
        'n_rules': evl['n_applicable_rules'],
        'rule_ids': evl['which_rules_applied'],
        'aligned': rule_system.evaluate(temp)['overall_ok']
    }

def generate_single_rule_probe(rule, n_features, distractor_features, grammar, init_str=None, seed=0):

    """Generate a token sequence for a single rule.

    Produces a sequence of the form:
        <bos> init_str [preamble_i feature_i ...] preamble_target target_val <eos>
    where condition features and distractor features are shuffled, and the
    target preamble and value at the end.

    Args:
        rule: Rule with .conditions (dict) and .target (single-item dict).
        n_features: Total number of features in the sequence (including conditions and target).
        distractor_features: Pool of feature keys to sample distractors from.
        grammar: Grammar object with .init_str, .variables, .preambles, .stoi.
        init_str: Optional init string token; sampled from grammar if None.
        seed: Random seed for reproducibility.

    Returns:
        List of token indices forming the probe sequence.
    """
       
    rng = random.Random(seed)

    assert len(rule.target) == 1, f'Only one target permitted, but {len(rule.target)} found'
    rule_target_key = next(iter(rule.target))
    rule_target_val = next(iter(rule.target.values()))

    n_distractors = n_features - len(rule.conditions) - 1
    assert n_distractors <= len(distractor_features), \
        f'Insufficient distractor features ({len(distractor_features)}), require at least {n_distractors}'

    if init_str is None:
        init_str = rng.choice(grammar.init_str)

    distractor_keys = rng.sample(distractor_features, n_distractors)
    probe_features = {**rule.conditions, **{k: rng.choice(grammar.variables[k]) for k in distractor_keys}}

    keys = list(probe_features)
    rng.shuffle(keys)

    probe = [grammar.stoi['<bos>'], grammar.stoi[init_str]]
    for feat in keys:
        probe += [grammar.stoi[rng.choice(grammar.preambles[feat])], grammar.stoi[probe_features[feat]]]
    probe.append(grammar.stoi[rng.choice(grammar.preambles[rule_target_key])])

    probe.append(grammar.stoi[rule_target_val])
    probe.append(grammar.stoi['<eos>'])

    return probe


def generate_competition_rule(rule1, rule2, n_features, distractor_features, grammar, init_str=None, seed=0):
    
    """Generate a token sequence probing competition between two rules.

    Both rules must target the same feature key but specify different values.
    The sequence contains all conditions from both rules plus distractors,
    shuffled randomly. The correct target value is determined by priority.

    Sequence format:
        <bos> init_str [preamble_i feature_i ...] preamble_target target_val <eos>

    Args:
        rule1: First competing rule.
        rule2: Second competing rule. Must target the same key as rule1,
               with disjoint conditions and different priority.
        n_features: Total number of features in the sequence (excluding bos/init/eos).
        distractor_features: Pool of feature keys to sample distractors from.
        grammar: Grammar object with .init_str, .variables, .preambles, .stoi.
        init_str: Optional init string token; sampled from grammar if None.
        seed: Random seed for reproducibility.
    
    Returns:
        List of token indices: <bos> init_str [preamble feat]* preamble_target target <eos>
    """

    rng = random.Random(seed)

    assert (len(rule1.target) == 1) and (len(rule2.target) == 1),\
          f'Only one target permitted, but lengths: {len(rule1.target)} , {len(rule2.target)} found'
    assert not rule1.conditions.keys() & rule2.conditions.keys(), \
        f"Rules share condition keys: {rule1.conditions.keys() & rule2.conditions.keys()}"


    rule1_target_key = next(iter(rule1.target))
    rule2_target_key = next(iter(rule2.target))
    assert rule1_target_key == rule2_target_key, f"These rules are not in competition"

    assert rule1.priority != rule2.priority, "Rules must have different priorities to determine winner"
    if rule1.priority > rule2.priority:
        template_target = next(iter(rule1.target.values()))
    else:
        template_target = next(iter(rule2.target.values()))

    n_distractors = n_features - len(rule1.conditions) - len(rule2.conditions) - 1

    assert n_distractors <= len(distractor_features), \
        f'Insufficient distractor features ({len(distractor_features)}), require at least {n_distractors}'

    if init_str is None:
        init_str = rng.choice(grammar.init_str)

    distractor_keys = rng.sample(distractor_features, n_distractors)
    probe_features = {**rule1.conditions, **rule2.conditions, **{k: rng.choice(grammar.variables[k]) for k in distractor_keys}}

    keys = list(probe_features)
    rng.shuffle(keys)

    probe = [grammar.stoi['<bos>'], grammar.stoi[init_str]]
    for feat in keys:
        probe += [grammar.stoi[rng.choice(grammar.preambles[feat])], grammar.stoi[probe_features[feat]]]
    probe.append(grammar.stoi[rng.choice(grammar.preambles[rule1_target_key])])

    probe.append(grammar.stoi[template_target])
    probe.append(grammar.stoi['<eos>'])

    return probe

def generate_composition_probe(rules, n_features, distractor_features, grammar, init_str=None, seed=0):
    
    """Generate a token sequence probing composition of two rules.

    Rules have non overlapping conditions and targets.

    Sequence format:
        <bos> init_str [preamble_i feature_i ...] preamble_target target_val <eos>

    Args:
        rules: List of rules
        n_features: Total number of features in the sequence (excluding bos/init/eos).
        distractor_features: Pool of feature keys to sample distractors from.
        grammar: Grammar object with .init_str, .variables, .preambles, .stoi.
        init_str: Optional init string token; sampled from grammar if None.
        seed: Random seed for reproducibility.
    
    Returns:
        List of token indices: <bos> init_str [preamble feat]* preamble_target target <eos>
    """

    rng = random.Random(seed)

    for rule in rules:
        assert (len(rule.target) == 1),\
          f'Only one target permitted, but lengths: {len(rule.target)} targets found for rule: {rule}'
        
    all_condition_keys = [k for rule in rules for k in rule.conditions.keys()]
    all_target_keys = [k for rule in rules for k in rule.target.keys()]
    all_keys = all_condition_keys + all_target_keys

    counts = Counter(all_keys)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    assert not duplicates, f"Feature keys appear more than once across rules: {duplicates}"

    ## randomly select a rule to be the last token
    final_token_rule_idx = rng.randint(0, len(rules)-1)
    final_token_target = rules[final_token_rule_idx].target
    final_token_key, final_token_val = next(iter(final_token_target.items()))

    probe_features = {}
    for ri, rule in enumerate(rules):
        probe_features.update(rule.conditions)
        if ri != final_token_rule_idx:
            probe_features.update(rule.target)
    
    if init_str is None:
        init_str = rng.choice(grammar.init_str)

    n_distractors = n_features - len(probe_features) - 1
    assert n_distractors <= len(distractor_features), \
        f'Insufficient distractor features ({len(distractor_features)}), require at least {n_distractors}'
    
    if n_distractors < 0:
        print('For this rule combination: there are too many conditions for the sequence length')
        return None

    if n_distractors >= 1:
        distractor_keys = rng.sample(distractor_features, n_distractors)
        probe_features = {**probe_features, **{k: rng.choice(grammar.variables[k]) for k in distractor_keys}}

    keys = list(probe_features)
    rng.shuffle(keys)

    probe = [grammar.stoi['<bos>'], grammar.stoi[init_str]]
    for feat in keys:
        probe += [grammar.stoi[rng.choice(grammar.preambles[feat])], grammar.stoi[probe_features[feat]]]
    probe.append(grammar.stoi[rng.choice(grammar.preambles[final_token_key])])

    probe.append(grammar.stoi[final_token_val])
    probe.append(grammar.stoi['<eos>'])

    return probe

def generate_exception_probe(base_rule, exception_rule, n_features, distractor_features, grammar, init_str=None, seed=0):
    """Generate a token sequence probing rule exception resolution.

    An exception pair has the exception rule's conditions as a strict superset
    of the base rule's conditions, targeting the same key with a different value.
    When all exception conditions are present, the exception (higher priority) wins.

    Sequence format:
        <bos> init_str [preamble_i feature_i ...] preamble_target target_val <eos>

    Args:
        base_rule: The general rule (lower priority, fewer conditions).
        exception_rule: The exception rule (higher priority, strict superset of conditions).
        n_features: Total number of features in the sequence (excluding bos/init/eos).
        distractor_features: Pool of feature keys to sample distractors from.
        grammar: Grammar object with .init_str, .variables, .preambles, .stoi.
        init_str: Optional init string token; sampled from grammar if None.
        seed: Random seed for reproducibility.

    Returns:
        List of token indices: <bos> init_str [preamble feat]* preamble_target target <eos>
    """
    rng = random.Random(seed)

    base_conds = set(base_rule.conditions.items())
    exc_conds = set(exception_rule.conditions.items())
    assert base_conds < exc_conds, "exception_rule conditions must be a strict superset of base_rule conditions"
    assert exception_rule.priority > base_rule.priority, "exception_rule must have higher priority"

    exc_target_key = next(iter(exception_rule.target))
    assert next(iter(base_rule.target)) == exc_target_key, "Rules must target the same key"
    assert exception_rule.target != base_rule.target, "Rules must specify different target values"

    n_distractors = n_features - len(exception_rule.conditions) - 1
    assert n_distractors <= len(distractor_features), \
        f'Insufficient distractor features ({len(distractor_features)}), require at least {n_distractors}'

    if init_str is None:
        init_str = rng.choice(grammar.init_str)

    distractor_keys = rng.sample(distractor_features, n_distractors)
    probe_features = {**exception_rule.conditions, **{k: rng.choice(grammar.variables[k]) for k in distractor_keys}}

    keys = list(probe_features)
    rng.shuffle(keys)

    probe = [grammar.stoi['<bos>'], grammar.stoi[init_str]]
    for feat in keys:
        probe += [grammar.stoi[rng.choice(grammar.preambles[feat])], grammar.stoi[probe_features[feat]]]
    probe.append(grammar.stoi[rng.choice(grammar.preambles[exc_target_key])])
    probe.append(grammar.stoi[next(iter(exception_rule.target.values()))])
    probe.append(grammar.stoi['<eos>'])

    return probe


def generate_unique_probes(probe_fn, n_probes, n_tries, test_frac=0.5, filter_fn=None, label=None, **probe_kwargs):
    """Generate a set of unique probes by repeated sampling, then split into train/test.

    Args:
        probe_fn: Function to generate a single probe. Must accept a `seed` kwarg.
        n_probes: Target number of unique probes to generate.
        n_tries: Maximum number of attempts.
        test_frac: Fraction of probes to use as test set.
        filter_fn: Optional callable that takes a probe (list) and returns True to include it.
        label: Optional label for print messages (e.g. rule index or pair tuple).
        **probe_kwargs: Additional kwargs passed to probe_fn (excluding seed).

    Returns:
        dict with keys 'train' and 'test', each a list of token sequences.
    """
    seen = set()

    for seed in range(n_tries):
        probe = probe_fn(seed=seed, **probe_kwargs)
        if probe is None:
            print(f'Could not generate sequences for combination: {label}')
            return
        if filter_fn is None or filter_fn(probe):
            seen.add(tuple(probe))
        if len(seen) == n_probes:
            print(f'{len(seen)} unique probes{f" for {label}" if label is not None else ""}')
            break
    else:
        print(f'Only found {len(seen)}/{n_probes} unique probes in {n_tries} tries{f" for {label}" if label else ""}')

    train, test = train_test_split_list(list(seen), test_frac)
    return {'train': train, 'test': test}

