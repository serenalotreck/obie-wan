"""
Remove triples that didn't properly tokenize to 3 elements from a GPT
prediction set, as well as "No relation" in preparation for evaluation.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, splitext
import json
from collections import defaultdict


def main(to_clean):

    # Read in the file
    with open(to_clean) as myf:
        data = json.load(myf)

    # Go through and drop any non len() 3 triples
    clean_data = defaultdict(list)
    num_dropped = defaultdict(int)
    for doc_key, doc_trips in data.items():
        for trip in doc_trips:
            if isinstance(trip, list):
                if len(trip) != 3:
                    num_dropped[doc_key] += 1
                else:
                    # Account for the fact that the elements of the triple may also be lists
                    list_elts = [isinstance(e, list) for e in trip]
                    if True in list_elts:
                        num_dropped[doc_key] += 1
                    else:
                        clean_data[doc_key].append(trip)
            else: num_dropped[doc_key] += 1

    # Save and print results
    print('\nNumber of triples with a non-3 length that were dropped:')
    print(num_dropped)
    path_and_prefix, ext = splitext(to_clean)
    new_name = f'{path_and_prefix}_CLEAN{ext}'
    with open(new_name, 'w') as myf:
        json.dump(clean_data, myf)
    print(f'Saved cleaned triples to {new_name}')
    print('\nDone!\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Remove malformed triples')

    parser.add_argument('to_clean', type=str,
            help='Path to file to clean. Will be saved to the same path plus '
            'the string CLEAN appended to the end')

    args = parser.parse_args()

    args.to_clean = abspath(args.to_clean)

    main(args.to_clean)
