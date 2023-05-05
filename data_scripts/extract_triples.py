"""
Extracts triples in the form ("entity 1", "relation label", "entity 2") from a
dygiepp-formatted representation of entity and relation annotations.

This can be used for predictions that come in the dygiepp-format, as well as to
prepare the gold standard for input into evaluate_triples.py.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import json
import jsonlines


def get_doc_triples(doc, dygiepp):
    """
    Extracts triples form a dygiepp-formatted doc. Ignores entities
    that are not involved in a relation. If "predicted_relations" is
    present, extracts triples from there, ignoring "relations".

    parameters:
        doc, dygiepp-formatteed doc: triples to extract
        dygiepp, bool: If True, indicates that there are logit and softmax 
            scores at the end of the relation list that need to be dropped
            before processing

    returns:
        triples, list of tuple: extracted triples
    """
    full_doc_toks = []
    triples = []
    for sent_idx in range(len(doc["sentences"])):
        full_doc_toks.extend(doc["sentences"][sent_idx])
        try:
            for rel in doc["predicted_relations"][sent_idx]:
                # Drop the logit and sofmax scores
                if dygiepp:
                    rel = rel[:-2]
                ent1_txt = ' '.join(full_doc_toks[rel[0]:rel[1]+1])
                rel_label = rel[-1]
                ent2_txt = ' '.join(full_doc_toks[rel[2]:rel[3]+1])
                trip = (ent1_txt, rel_label, ent2_txt)
                triples.append(trip)
        except KeyError:
            for rel in doc["relations"][sent_idx]:
                ent1_txt = ' '.join(full_doc_toks[rel[0]:rel[1]+1])
                rel_label = rel[-1]
                ent2_txt = ' '.join(full_doc_toks[rel[2]:rel[3]+1])
                trip = (ent1_txt, rel_label, ent2_txt)
                triples.append(trip)

    return triples


def main(input_file, out_loc, out_prefix, dygiepp):

    # Read in the file
    docs = []
    with jsonlines.open(input_file) as reader:
        for obj in reader:
            docs.append(obj)

    # Extract triples
    triples = {}
    for doc in docs:
        doc_key = doc["doc_key"]
        trips = get_doc_triples(doc, dygiepp)
        triples[doc_key] = trips

    # Save file
    out_name = f'{out_loc}/{out_prefix}_per_doc_triples.json'
    with open(out_name, 'w') as myf:
        json.dump(triples, myf)
    print(f'Saved triples as {out_name}.')

    print('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract triples')

    parser.add_argument('input_file', type=str,
            help='Dygiepp-formatted file with "ner" and "relation" or '
            '"predicted_ner" and "predicted_relation" fields.')
    parser.add_argument('-out_loc', type=str,
            help='Path to save the output')
    parser.add_argument('-out_prefix', type=str,
            help='String to prepend to output file names')
    parser.add_argument('--dygiepp', action='store_true',
            help='If specified, the elements in the optional '
            'predicted_relations field contain a logit and softmax score')

    args = parser.parse_args()

    args.input_file = abspath(args.input_file)
    args.out_loc = abspath(args.out_loc)

    main(args.input_file, args.out_loc, args.out_prefix, args.dygiepp)
