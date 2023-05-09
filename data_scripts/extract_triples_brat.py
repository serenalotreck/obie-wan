"""
Pulls unbound triples from brat documents. Allows disjoint entity annotations.
Ignores all annotations not beginning with T or R.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, splitext, isfile
from os import listdir
import json


def get_brat_trips(brat_lines):
    """
    Get triples from a brat document.

    parameters:
        brat_lines, list of str: each element is one line of a brat .ann file

    returns:
        trips, list of list of str: each internal list is a triple
    """
    # Turn the list into two dicts, one for T anns and one for R anns
    t_dict = {}
    r_dict = {}
    for line in brat_lines:
        id = line.split('\t')[0]
        if id[0] == 'T':
            ent_txt = line.split('\t')[-1]
            t_dict[id] = ent_txt
        elif id[0] == 'R':
            rel_line = line.split('\t')[-1]
            r_dict[id] = rel_line

    # For each rel, get the ent text and rel label
    trips = []
    for r_id, rel_line in r_dict.items():
        split_line = rel_line.split(' ')
        r_type = split_line[0]
        r_arg1_id = split_line[1][5:].strip()
        r_arg2_id = split_line[2][5:].strip()
        arg1_text = t_dict[r_arg1_id]
        arg2_text = t_dict[r_arg2_id]
        trip = [arg1_text, r_type, arg2_text]
        trips.append(trip)

    return trips


def main(brat_dir, out_loc, out_prefix):

    # Read in brat .ann files
    verboseprint('\nReading in brat files...')
    anns = {}
    for f in listdir(brat_dir):
        full_ap = f'{brat_dir}/{f}'
        if isfile(full_ap) and splitext(f)[-1] == '.ann':
            with open(full_ap) as myf:
                brat_lines = myf.readlines()
                brat_lines = [l.strip() for l in brat_lines]
                doc_key = splitext(f)[0]
                anns[doc_key] = brat_lines

    # Get trips
    verboseprint('\nPulling triples...')
    triples = {}
    for doc, brat_lines in anns.items():
        trips = get_brat_trips(brat_lines)
        triples[doc] = trips

    # Save
    verboseprint('\nSaving...')
    out_path = f'{out_loc}/{out_prefix}_per_doc_unbound_triples.json'
    with open(out_path, 'w') as myf:
        json.dump(triples, myf)
    verboseprint(f'Saved output as {out_path}')

    verboseprint('\nDone!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Get triples from brat')

    parser.add_argument('brat_dir', type=str,
            help='Path to directory with brat annotations')
    parser.add_argument('out_loc', type=str,
            help='Path to save output files')
    parser.add_argument('out_prefix', type=str,
            help='String to prepend to output file names')
    parser.add_argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')

    args = parser.parse_args()

    args.brat_dir = abspath(args.brat_dir)
    args.out_loc = abspath(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None

    main(args.brat_dir, args.out_loc, args.out_prefix)
