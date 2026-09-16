import numpy as np
import random
import pandas as pd
from collections import defaultdict
import re
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from torch.utils.data import DataLoader, Dataset, random_split
import string
import itertools
import torch

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
### DATASET ----------------------------
########################################


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
        rule_str = ['if ']
        for k, v in self.conditions.items():
            rule_str.append(k + ' is ' + v)
        target_key, target_val = next(iter(self.target.items()))
        rule_str.append(' then ' + target_key + ' MUST be ' + target_val)
        rule_str.append(f'  (priority {self.priority})')

        return ''.join(rule_str)

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

    # ---------- basic accessors ----------

    def add(self, rule: Rule):
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

