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


def get_doc_trip_counts(trips, gold_trips, check_rel_label, nlp, pos_neg):
    """
    Update the pos_neg dictionary with true & false positives & negatives.

    parameters:
        trips, list of list: each inner list is a triple whose elements are
            strings
        gold_trips, list of list: same format as predicted triples
        check_rel_label, bool: whether or not to check the identity of the
            middle element of the triple
        nlp, spacy nlp object: tokenizer to use to compare elements
        pos_neg, dict: positive negative dictionary

    returns:
        pos_neg, dict: updated pos_neg dictionary
    """
    # Tokenize the gold standard
    gold_toks = []
    for g in gold_trips:
        if check_rel_label:
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

    # Tokenize each triple and see if it's in the gold standard
    trip_toks = []
    for t in trips:
        if check_rel_label:
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
        if trip_list in gold_toks:
            pos_neg['tp'] += 1
        else:
            pos_neg['fp'] += 1

    # Check for gold standard for false negatives
    for g in gold_toks:
        if g not in trip_toks:
            pos_neg['fn'] += 1

    return pos_neg


def get_f1_input(preds, gold, nlp, check_rel_labels=False):
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
                pos_neg)

    # Add to get f1 inputs
    predicted = pos_neg['tp'] + pos_neg['fp']
    gold = pos_neg['tp'] + pos_neg['fn']
    matched = pos_neg['tp']

    return (predicted, gold, matched)


def draw_boot_samples(preds, gold, num_boot=500, check_rel_labels=False):
    """
    Draw bootstrap samples.

    parameters:
        preds, dict: model predictions
        gold, dict: gold standard annotations
        num_boot, int: number of bootstrap samples to draw
        check_rel_labels, bool: whether or not to check the identity of
            relation text when evaluating performance

    returns:
        prec_samples, list of float: list of precision values for bootstraps
        rec_samples, list of float: list of recall values for bootstraps
        f1_samples, list of float: list of f1 values for bootstraps
    """
    prec_samples = []
    rec_samples = []
    f1_samples = []

    nlp = spacy.load("en_core_sci_sm")

    # Draw the boot samples
    for _ in range(num_boot):
        # Sample prediction dicts with replacement
        samp_ids = np.random.choice(preds.keys(),
                size=len(preds.keys()), replace=True)
        pred_samp = {k:v for k,v in preds.items() if k in samp_ids}
        gold_samp = {k:v for k,v in gold.items() if k in samp_ids}
        # Calculate performance for the sample
        pred, gold, match = get_f1_input(pred_samp, gold_samp,
                check_rel_labels, nlp)
        prec, rec, f1 = compute_f1(pred, gold, match)
        # Append each of the performance values to their respective sample lists
        prec_samples.append(prec)
        rec_samples.append(rec)
        f1_samples.append(f1)

    return (prec_samples, rec_samples, f1_samples)


def main(preds, gold, out_loc, out_prefix, check_rel_labels, num_boot):

    # Read in the preds and gold
    with open(preds) as myf:
        preds = json.load(myf)
    with open(gold) as myf:
        gold = json.load(myf)

    # Compare and calculate performance
    prec_samples, rec_samples, f1_samples = draw_boot_samples(preds, gold,
            num_boot, check_rel_labels)
    prec_CI, rec_CI, f1_CI = calculate_CI(prec_samples, rec_samples,
            f1_samples)
    perf_dict = {
            'precision': np.mean(prec_samples),
            'precision_CI': prec_CI,
            'recall': np.mean(rec_samples),
            'recall_CI': rec_CI,
            'F1': np.mean(f1_samples),
            'F1_CI': f1_CI
            }
    perf_df = pd.DataFrame.from_dict(perf_dict)

    # Save output
    out_name = f'{out_loc}/{out_prefix}_performance.csv'
    perf_df.to_csv(out_name)
    print(f'Saved output as {out_name}')


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
    parser.add_argument('--check_rel_labels', action='store_true',
            help='Whether or not to include the identity of relations when '
            'evaluating performance')

    args = parser.parse_args()

    args.preds = abspath(args.preds)
    args.gold = abspath(args.gold)
    args.out_loc = abspath(args.out_loc)

    main(args.preds, args.gold, args.out_loc, args.out_prefix,
            args.check_rel_labels, args.num_boot)
