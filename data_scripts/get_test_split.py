"""
Using sklearn tools, get a subset of a jsonl formatted dataset to use
as a test set and a train set.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import jsonlines
from sklearn.model_selection import train_test_split


def main(dataset, out_loc, out_prefix, fraction, random_state):

    # Read in the dataset
    print('\nReading in the data...')
    with jsonlines.open(dataset) as reader:
        docs = []
        for obj in reader:
            docs.append(obj)

    # Perform train test split
    print('\nPerforming train test split...')
    train_docs, test_docs = train_test_split(docs, test_size=fraction,
            random_state=random_state)
    print(f'Relative lengths of the train and test sets are {len(train_docs)}, '
            f'{len(test_docs)}')

    # Save out the docs
    train_out_name = f'{out_loc}/{out_prefix}_TRAIN.jsonl'
    test_out_name = f'{out_loc}/{out_prefix}_TEST.jsonl'

    with jsonlines.open(train_out_name, 'w') as writer:
        writer.write_all(train_docs)
    with jsonlines.open(test_out_name, 'w') as writer:
        writer.write_all(test_docs)

    print(f'\nSaved data to {out_loc} as {out_prefix}_TRAIN.jsonl and '
            f'{out_prefix}_TEST.jsonl.')

    print('\nDone!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get test set')

    parser.add_argument('dataset', type=str,
            help='Path to jsonlines formatted dataset')
    parser.add_argument('out_loc', type=str,
            help='Path to save the outputs')
    parser.add_argument('out_prefix', type=str,
            help='String to prepend to file names')
    parser.add_argument('-fraction', type=float, default=0.1,
            help='Fraction of the dataset to withold for testing, '
            'default is 0.1')
    parser.add_argument('-random_state', type=int, default=1234,
            help='Random state to ensure reproducibility on the same '
            'dataset. Default is 1234')

    args = parser.parse_args()

    args.dataset = abspath(args.dataset)
    args.out_loc = abspath(args.out_loc)

    main(args.dataset, args.out_loc, args.out_prefix,  args.fraction, args.random_state)
