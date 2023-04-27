"""
Converts the Huggingface version of the BioInfer dataset to brat standoff. Does
both train and test and saves as separate files. Maintains disjoint entities.

Link to dataset: https://huggingface.co/datasets/bigbio/bioinfer

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, exists
from os import mkdir
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm


def replace_ids(ent_anns, rel_anns):
    """
    Removes the original annotation ID's and joins annotations into a formatted
    string for saving.

    parameters:
        ent_anns, dict: keys are original ent ID's, values are ent anns
        rel_anns, dict: keys are original rel ID's, values are rel anns

    returns:
        anns_to_save, str: formatted string
    """
    ann_list = []
    for ent_key, ent in ent_anns.items():
        ann_list.append(ent)
    for rel_key, rel in rel_anns.items():
        ann_list.append(rel)
    anns_to_save = '\n'.join(ann_list)

    return anns_to_save


def get_joined_text(docs):
    """
    Gets the full joined text of a document.

    parameters:
        docs, list of dict: doc objects to join

    returns:
        text, str: full doc text
    """
    text = ''
    for doc in docs:
        if text != '':
            text += ' ' + doc['text']
        else:
            text += doc['text']

    return text


def get_rel_anns(docs, ent_anns):
    """
    Map relations onto renamed entities and format for brat.

    parameters:
        docs, list of dict: docs form which to get relations
        ent_anns, dict: keys are original entity ID's, values are brat entities

    returns:
        rel_anns, dict: keys are old rel ID's (for consistency with
            get_ent_anns), and values are brat-formatted relations
    """
    rel_anns = {}
    r_num = 1
    for doc in docs:
        for rel in doc['relations']:
            rel_key = rel['id']
            r_key = f'R{r_num}'
            rel_type = rel['type']
            arg1_orig_id = rel['arg1_id']
            arg2_orig_id = rel['arg2_id']
            arg1_tid = ent_anns[arg1_orig_id].split('\t')[0]
            arg2_tid = ent_anns[arg2_orig_id].split('\t')[0]
            full_ann = f'{r_key}\t{rel_type} Arg1:{arg1_tid} Arg2:{arg2_tid}'
            rel_anns[rel_key] = full_ann
            r_num += 1

    return rel_anns


def get_ent_anns(docs):
    """
    Format entity annotations into brat format, but leave original entity ID as
    the key of the output dictionary for alter mapping onto relations.

    parameters:
        docs, list of dict: docs from which to get entities

    returns:
        ent_anns, dict: keys are original entity ID's, values are brat strings
    """
    ent_anns = {}
    t_num = 1
    prev_len = 0
    for doc in docs:
        for ent in doc['entities']:
            ent_key = ent['id']
            t_key = f'T{t_num}'
            ent_type = ent['type']
            # Allows disjoint entities
            if len(ent['offsets']) > 1:
                offset_str_list = []
                for offset_pair in ent['offsets']:
                    offset_pair = [str(o) for o in offset_pair]
                    offset_str_list.append(' '.join(offset_pair))
                offset_str = ';'.join(offset_str_list)
                # Check that text always has multiples
                assert len(ent['text']) > 1
                txt = ' '.join(ent['text'])
            else:
                start = ent['offsets'][0][0] + prev_len
                end = ent['offsets'][0][1] + prev_len
                offset_str = f'{start} {end}'
                txt = ent['text'][0]
            full_ann = f'{t_key}\t{ent_type} {offset_str}\t{txt}'
            ent_anns[ent_key] = full_ann
            t_num += 1
        prev_len += len(doc['text']) + 1 # For the space

    return ent_anns


def join_docs(dset):
    """
    Indexes the dataset by document number, not sentence, and joins sentences
    from the same doc into one list.

    parameters:
        dset, Huggingface Dataset instance: dataset to join

    returns:
        joined_dset, dict: values are doc ID's and keys are lists of doc
            dictionaries
    """
    joined_dset = defaultdict(list)
    for item in dset:
        # Make sure they're all sentences
        assert item['type'] == 'Sentence'
        # Make sure the doc ID splits correctly
        assert len(item['document_id'].split('.')) == 3
        # Get shortened doc ID
        new_id = '.'.join(item['document_id'].split('.')[:2])
        # Add to joined list
        joined_dset[new_id].append(item)

    return joined_dset


def huggingface2brat(dataset, out_loc, out_prefix):
    """
    Convert the huggingface bioinfer dataset to brat format.

    parameters:
        dataset, Huggingface Dataset instance: dataset to convert
        out_loc, str: directory to save output
        out_prefix, str: prefix, if any, to prepend to file names

    returns: None
    """
    # Join sentences from the same abstract into the same document
    joined_dset = join_docs(dataset)

    # Get the docs to save
    for doc_key, docs in tqdm(joined_dset.items()):
        # Get the annotations in brat format but leave original ID's for now
        ent_anns = get_ent_anns(docs)
        rel_anns = get_rel_anns(docs, ent_anns)
        # Get text
        text = get_joined_text(docs)
        # Replace original ID's with sequential brat standoff ID's
        anns_to_save = replace_ids(ent_anns, rel_anns)
        # Get save names
        txt_save = f'{out_loc}/{out_prefix}{doc_key}.txt'
        ann_save = f'{out_loc}/{out_prefix}{doc_key}.ann'
        # Save docs
        with open(txt_save, 'w') as myf:
            myf.write(text)
        with open(ann_save, 'w') as myf:
            myf.write(anns_to_save)


def main(out_loc, out_prefix, split):

    # Check existence of out_loc
    verboseprint('\nChecking existence of output directory...')
    if not exists(out_loc):
        mkdir(out_loc)
        verboseprint(f'New directory created at {out_loc}')

    # Read in both datasets
    verboseprint('\nReading in dataset...')
    dataset = load_dataset('bigbio/bioinfer', split=split)

    # Convert to brat
    verboseprint('\nConverting datasets...')
    huggingface2brat(dataset, out_loc, out_prefix)

    verboseprint('\nDone!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert BioInfer')

    parser.add_argument('out_loc', type=str,
            help='Path to directory to save output. Will be created if it '
            'doesn\'t already exist')
    parser.add_argument('split', type=str,
            help='Either "train" or "test", which split of the data to '
            'convert')
    parser.add_argument('-out_prefix', type=str, default='',
            help='Prefix to prepend to output files')
    parser.add_argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')

    args = parser.parse_args()

    args.out_loc = abspath(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None

    main(args.out_loc, args.out_prefix, args.split)

