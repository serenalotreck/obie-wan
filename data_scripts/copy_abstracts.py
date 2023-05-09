"""
Script to create a new directory and copy the abstracts from a jsonl file into
the new directory.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, exists, basename
from os import makedirs
import subprocess
import jsonlines


def main(dataset, all_abstract_dir, new_abstract_dir, kind):

    # Read in jsonlines file
    with jsonlines.open(dataset) as reader:
        dset = []
        for obj in reader:
            dset.append(obj)

    # Get doc keys
    keys = [d["doc_key"] for d in dset]

    # Make them into the filepaths to copy
    if kind == 'all':
        copy_keys = []
        for k in keys:
            txt_path = f'{all_abstract_dir}/{k}.txt'
            ann_path = f'{all_abstract_dir}/{k}.ann'
            copy_keys.extend([txt_path, ann_path])
    elif kind == 'txt':
        copy_keys = [all_abstract_dir + '/' + k + '.txt' for k in keys]

    # Check if target directory exists
    if not exists(new_abstract_dir):
        makedirs(new_abstract_dir)
        print('Target directory didn\'t exist, new directory created at '
            f'{new_abstract_dir}')

    # Build the copy command
    copy_cmd = ['cp'] + copy_keys + [new_abstract_dir]

    # Perform operation
    subprocess.run(copy_cmd)

    print(f'\nSnapshot of the abstracts copied to {new_abstract_dir}:')
    snap_keys = [basename(k) for k in copy_keys]
    print(snap_keys[:5])
    print('\nDone!\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy abstracts')

    parser.add_argument('dataset', type=str,
            help='Path to jsonlines file with abstracts to copy')
    parser.add_argument('all_abstract_dir', type=str,
            help='Path to directory from which to copy abstracts')
    parser.add_argument('new_abstract_dir', type=str,
            help='Path to which abstracts will be copied. Will be created if '
            'it doesn\'t already exist')
    parser.add_argument('-kind', type=str,
            help='Whether or not to copy all file extensions ("all") or only '
            'those with .txt extensions ("txt").')

    args = parser.parse_args()

    args.dataset = abspath(args.dataset)
    args.all_abstract_dir = abspath(args.all_abstract_dir)
    args.new_abstract_dir = abspath(args.new_abstract_dir)


    main(args.dataset, args.all_abstract_dir, args.new_abstract_dir, args.kind)
