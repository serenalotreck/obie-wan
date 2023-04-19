"""
Script to evaluate the prediction output against the gold standard for models
obtaining triples from text.

Code adapted from
serenalotreck/pickle-corpus-code/models.evaluate_model_output.py unless
otherwise indicated.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import json
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_CI(prec_samples, rec_samples, f1_samples):
    """
    Calculates CI from bootstrap samples using the percentile method with
    alpha = 0.05 (95% CI).

    parameters:
        prec_samples, list of float: list of precision values for bootstraps
        rec_samples, list of float: list of recall values for bootstraps
        f1_samples, list of float: list of f1 values for bootstraps

    returns list with elements:
        prec_CI, tuple of float: CI for precision
        rec_CI, tuple of float: CI for recall
        f1_CI, tuple of float: CI for F1 score
    """
    alpha = 0.05
    CIs = {}
    for name, samp_set in {'prec_samples':prec_samples,
                           'rec_samples':rec_samples,
                           'f1_samples':f1_samples}.items():
        lower_bound = np.percentile(samp_set, 100*(alpha/2))
        upper_bound = np.percentile(samp_set, 100*(1-alpha/2))
        name = name.split('_')[0] + '_CI'
        CIs[name] = (lower_bound, upper_bound)

    return [CIs['prec_CI'], CIs['rec_CI'], CIs['f1_CI']]


def safe_div(num, denom):
    """
    Function from https://www.github.com/dwadden/dygiepp/dygie/training/f1.py
    """
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    """
    Function from https://www.github.com/dwadden/dygiepp/dygie/training/f1.py
    """
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def remove_trip_dups(trips, check_rel_labels, sym_labs):
    """
    Remove duplicates from a list of triples. If check_rel_labels is False,
    triples with inverted order are considered equivalent and only one is kept.
    If check_rel_labels is True, only relation types in sym_labs are considered
    equivalent if inverted, and all others are only removed if an exact
    duplicate is found.

    parameters:
        trips, list of list: tokenized triples
        check_rel_labels, bool: whether or not to consider relation labels
        sym_labs, list of str: labels that are symmetric if check_rel_labels is
            True

    returns:
        unduped_trips, list of list: untokenized triples with duplicates removed
    """
    unduped_trips = []
    for t in trips:
        # If it's not already in our list,
        if t not in unduped_trips:
            # check whether or not we care about the label
            if check_rel_labels:
                # If we do care, check whether or not it's a symmetric label
                if t[1][0] in sym_labs: ## CAUTION will fail if multiple tokens
                    # If it is, check for the inverted version as well
                    if t[::-1] not in unduped_trips:
                        # If neither version is there, add it
                        unduped_trips.append(t)
                    # Otherwise, we do nothing
                # If it's not a symmetric label, add it
                else:
                    unduped_trips.append(t)
            # If we don't care about the relation label,
            else:
                # then we want to check symmetrically
                if t[::-1] not in unduped_trips:
                    # and add it if it's not there
                    unduped_trips.append(t)
        # And if it is already in our list, we move on

    return unduped_trips


def get_doc_trip_counts(trips, gold_trips, check_rel_labels, nlp, pos_neg,
        sym_labs):
    """
    Update the pos_neg dictionary with true & false positives & negatives.

    parameters:
        trips, list of list: each inner list is a triple whose elements are
            strings
        gold_trips, list of list: same format as predicted triples
        check_rel_labels, bool: whether or not to check the identity of the
            middle element of the triple
        nlp, spacy nlp object: tokenizer to use to compare elements
        pos_neg, dict: positive negative dictionary
        sym_labs, list of str: list of symmetric relation types for which order
            doesn't matter. Only used if check_rel_labels = True

    returns:
        pos_neg, dict: updated pos_neg dictionary
    """
    # Count original numbers of each set
    incoming_nums = (len(trips), len(gold_trips))

    # Tokenize the gold standard and remove duplicates
    gold_toks = []
    for g in gold_trips:
        if check_rel_labels:
            gold_trip_list = []
            for elt in g:
                doc = nlp(elt)
                tok_list = [tok.text for sent in doc.sents for tok in sent]
                gold_trip_list.append(tok_list)
            gold_toks.append(gold_trip_list)
        else:
            gold_trip_list = []
            for elt in [g[0], g[2]]:
                doc = nlp(elt)
                tok_list = [tok.text for sent in doc.sents for tok in sent]
                gold_trip_list.append(tok_list)
            gold_toks.append(gold_trip_list)
    gold_toks = remove_trip_dups(gold_toks, check_rel_labels, sym_labs)

    # Tokenize each triple and remove duplicates
    trip_toks = []
    for t in trips:
        if check_rel_labels:
            trip_list = []
            for elt in t:
                doc = nlp(elt)
                tok_list = [tok.text for sent in doc.sents for tok in sent]
                trip_list.append(tok_list)
            trip_toks.append(trip_list)
        else:
            trip_list = []
            for elt in [t[0], t[2]]:
                doc = nlp(elt)
                tok_list = [tok.text for sent in doc.sents for tok in sent]
                trip_list.append(tok_list)
            trip_toks.append(trip_list)
    trip_toks = remove_trip_dups(trip_toks, check_rel_labels, sym_labs)

    # Report the number of dropped triples in each set
    outgoing_nums = (len(trips), len(gold_trips))
    print(f'After deduplication, {incoming_nums[0] - outgoing_nums[0]} '
            'triples have been removed from the predicted triples, and '
            f'{incoming_nums[1] - outgoing_nums[1]} triples have been removed '
            'from the gold standard.')

    # Check if predicted triples are in the gold standard
    for trip_list in trip_toks:
        if check_rel_labels:
            # When checking relation labels, order matters for all but symmetric labels
            if trip_list in gold_toks:
                pos_neg['tp'] += 1
            else:
                if trip_list[1][0] in sym_labs: ## CAUTION: Will fail if relation
                                                ## tokenizes to more than one token
                    if trip_list[::-1] in gold_toks: # Allow order reversal
                        pos_neg['tp'] += 1
                    else:
                        pos_neg['fp'] += 1
                else:
                    pos_neg['fp'] += 1
        else:
            # Order agnostic for all triples
            if (trip_list in gold_toks) or (trip_list[::-1] in gold_toks):
                pos_neg['tp'] += 1
            else:
                pos_neg['fp'] += 1

    # Check for gold standard for false negatives
    false_negs = []
    for g in gold_toks:
        if check_rel_labels:
            if g[1][0] in sym_labs: ## CAUTION: Will fail if relation
                                    ## tokenize to more than one token
                if not ((g in trip_toks) or (g[::-1] in trip_toks)):
                    pos_neg['fn'] += 1
                    false_negs.append(g)
            else:
                if g not in trip_toks:
                    pos_neg['fn'] += 1
                    false_negs.append(g)
        else:
            if not ((g in trip_toks) or (g[::-1] in trip_toks)):
                pos_neg['fn'] += 1
                false_negs.append(g)
    untoked_false_negs = []
    for neg in false_negs:
        untoked = [' '.join(elt) for elt in neg]
        untoked_false_negs.append(untoked)
    print(f'false neg triples: {untoked_false_negs}')
    print(f'gold trips: {gold_trips}')
    print(f'pos neg before return: {pos_neg}')

    return pos_neg


def get_f1_input(preds, gold, nlp, sym_labs, check_rel_labels=False):
    """
    Get the number of true and false postives and false negatives for the
    model to calculate the following inputs for compute_f1 for both entities
    and relations:
        predicted = true positives + false positives
        gold = true positives + false negatives
        matched = true positives

    parameters:
        preds, dict: predictions, keys are doc_key and values are lists of
            triples
        gold, dict: gold std annotations, keys are doc_key and values are
            lists of triples
        nlp, spacy nlp object: tokenizer to use for evaluation
        sym_labs, list of str: list of symmetric relation types for which order
            doesn't matter. Only used if check_rel_label = True
        check_rel_labels, bool: whether or not to cehck the identity of
            relation labels when evaluating performance

    returns:
        predicted, int
        gold, int
        matched, int
    """
    pos_neg = {'tp':0, 'fp':0, 'fn':0}

    # Go through the docs
    for doc_key, trips in preds.items():
        # Get the corresponding gold standard
        gold_trips = gold[doc_key]
        # Get tp/fp/fn counts for this document
        pos_neg = get_doc_trip_counts(trips, gold_trips, check_rel_labels, nlp,
                pos_neg, sym_labs)

    # Add to get f1 inputs
    predicted = pos_neg['tp'] + pos_neg['fp']
    gold = pos_neg['tp'] + pos_neg['fn']
    matched = pos_neg['tp']

    return (predicted, gold, matched)


def draw_boot_samples(preds, gold, nlp, sym_labs, num_boot=500,
        check_rel_labels=False):
    """
    Draw bootstrap samples.

    parameters:
        preds, dict: model predictions
        gold, dict: gold standard annotations
        nlp, spacy nlp object: tokenizer to use
        num_boot, int: number of bootstrap samples to draw
        check_rel_labels, bool: whether or not to check the identity of
            relation text when evaluating performance
        sym_labs, list of str: list of symmetric relation types for which order
            doesn't matter. Only used if check_rel_label = True, default is
            None

    returns:
        prec_samples, list of float: list of precision values for bootstraps
        rec_samples, list of float: list of recall values for bootstraps
        f1_samples, list of float: list of f1 values for bootstraps
    """
    prec_samples = []
    rec_samples = []
    f1_samples = []

    # Draw the boot samples
    verboseprint('\nDrawing boot samples...')
    for _ in tqdm(range(num_boot)):
        # Sample prediction dicts with replacement
        samp_ids = np.random.choice(list(preds.keys()),
                size=len(preds.keys()), replace=True)
        pred_samp = {k:v for k,v in preds.items() if k in samp_ids}
        gold_samp = {k:v for k,v in gold.items() if k in samp_ids}
        # Calculate performance for the sample
        pred, golds, match = get_f1_input(pred_samp, gold_samp,
                nlp, sym_labs, check_rel_labels)
        prec, rec, f1 = compute_f1(pred, golds, match)
        # Append each of the performance values to their respective sample lists
        prec_samples.append(prec)
        rec_samples.append(rec)
        f1_samples.append(f1)

    return (prec_samples, rec_samples, f1_samples)


def main(preds, gold, out_loc, out_prefix, check_rel_labels, num_boot,
        bootstrap, sym_labs):

    # Read in the preds and gold
    verboseprint('\nReading in documents...')
    with open(preds) as myf:
        preds = json.load(myf)
    with open(gold) as myf:
        gold = json.load(myf)

    # Compare and calculate performance
    verboseprint('\nCalculating performance...')
    nlp = spacy.load("en_core_sci_sm")
    if bootstrap:
        prec_samples, rec_samples, f1_samples = draw_boot_samples(preds, gold,
                nlp, sym_labs, num_boot, check_rel_labels)
        prec_CI, rec_CI, f1_CI = calculate_CI(prec_samples, rec_samples,
                f1_samples)
        perf_dict = {
                'precision': [np.mean(prec_samples)],
                'precision_CI': [prec_CI],
                'recall': [np.mean(rec_samples)],
                'recall_CI': [rec_CI],
                'F1': [np.mean(f1_samples)],
                'F1_CI': [f1_CI]
                }
        perf_df = pd.DataFrame.from_dict(perf_dict)

    else:
        pred, golds, match = get_f1_input(preds, gold,
                nlp, sym_labs, check_rel_labels)
        prec, rec, f1 = compute_f1(pred, golds, match)
        perf_df = pd.DataFrame.from_dict(
                {
                    'precision': [prec],
                    'recall': [rec],
                    'F1': [f1]
                    }
                )

    # Save output
    verboseprint('\nSaving output...')
    out_name = f'{out_loc}/{out_prefix}_performance.csv'
    perf_df.to_csv(out_name, index=False)
    verboseprint(f'Saved output as {out_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract triples')

    parser.add_argument('preds', type=str,
            help='json file where keys are doc keys and values are lists of '
            'triples')
    parser.add_argument('gold', type=str,
            help='Path to file with same format as preds containing gold '
            'standard')
    parser.add_argument('-num_boot', type=int, default=500,
            help='Number of bootstrap samples to use')
    parser.add_argument('-out_loc', type=str,
            help='Path to save the output')
    parser.add_argument('-out_prefix', type=str,
            help='String to prepend to output file names')
    parser.add_argument('--bootstrap', action='store_true',
            help='Whether or not to bootstrap performance values')
    parser.add_argument('--check_rel_labels', action='store_true',
            help='Whether or not to include the identity of relations when '
            'evaluating performance')
    parser.add_argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')
    parser.add_argument('-sym_labs', nargs='+', default=None,
            help='Symmetrical relations to evaluate order-agnostically')
    args = parser.parse_args()

    args.preds = abspath(args.preds)
    args.gold = abspath(args.gold)
    args.out_loc = abspath(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None
    sym_labs = [] if args.sym_labs is None else args.sym_labs

    main(args.preds, args.gold, args.out_loc, args.out_prefix,
            args.check_rel_labels, args.num_boot, args.bootstrap, sym_labs)
