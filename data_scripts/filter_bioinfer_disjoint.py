"""
Generate a list of documents that need to be witheld from BioInfer for a given
split based on the presence of disjoint entities & resulting dropped relations.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, splitext
from os import listdir
import jsonlines
import json


def filter_bioinfer(drop_dict, jsonl):
    """
    Function to remove problem docs from the jsonl for a given bioinfer split.

    parameters:
        drop_dict, dict: keys are none/disjoint/other_dropped, values are the
            doc_keys for the documents in each category
        jsonl, list of dict: the jsonl representation of the dataset from which
            to drop documents

    returns:
        filtered_jsonl, list of dict: jsonl list without problem docs
    """
    # Combine ones to drop
    to_drop = drop_dict['disjoint_dropped'] + drop_dict['other_dropped']

    # Drop
    filtered_jsonl = []
    for doc in jsonl:
        if doc['doc_key'] not in to_drop:
            filtered_jsonl.append(doc)

    return filtered_jsonl


def main(brat_dir, jsonl, out_loc, out_prefix):
   
   # Read in brat ann files
    brat_strs = {}
    for f in listdir(brat_dir):
        if splitext(f)[-1] == '.ann':
            with open(f'{brat_dir}/{f}') as myf:
                brat_str = myf.read()
                brat_strs[splitext(f)[0]] = brat_str

    # Read in the jsonl and organize by doc_key
    with jsonlines.open(jsonl) as reader:
        jsonl_data = []
        for obj in reader:
            jsonl_data.append(obj)
    jsonl_docs = {}
    for jsl in jsonl_data:
        k = jsl['doc_key']
        jsonl_docs[k] = jsl

    # Check docs
    none_dropped = []
    disjoint_dropped = []
    other_dropped = []
    for doc_key, brat_doc in brat_strs.items():
        brat_parts = brat_doc.split('\n')
        # First check for disjoint, as this automatically disqualifies the doc
        num_ents_disjoint = sum([1 for l in brat_parts if ';' in l])
        if num_ents_disjoint != 0:
            disjoint_dropped.append(doc_key)
        else:
            num_ents_orig = sum([1 for l in brat_parts if l[0] == 'T'])
            num_rels_orig = sum([1 for l in brat_parts if l[0] == 'R'])
            jsl = jsonl_docs[doc_key]
            jsl_ents = 0
            jsl_rels = 0
            for i, sent in enumerate(jsl['sentences']):
                jsl_ents += len(jsl['ner'][i])
                jsl_rels += len(jsl['relations'][i])
            if (num_ents_orig != jsl_ents) or (num_rels_orig != jsl_rels):
                other_dropped.append(doc_key)
            else:
                none_dropped.append(doc_key)

    out_content = {'none_dropped': none_dropped,
                    'disjoint_dropped': disjoint_dropped,
                    'other_dropped': other_dropped}
    with open(f'{out_loc}/{out_prefix}_dropped.json', 'w') as myf:
        json.dump(out_content, myf)
    
    print(f'Output saved to {out_loc}/{out_prefix}_dropped.json')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Filter bioinfer')
    
    parser.add_argument('brat_dir', type=str,
            help='Path to directory with brat files')
    parser.add_argument('jsonl', type=str,
            help='Path to jsonl version of the dataset')
    parser.add_argument('out_loc', type=str,
            help='Path to save output file')
    parser.add_argument('out_prefix', type=str,
            help='Path to prepend to output filename')
    
    args = parser.parse_args()
    
    args.brat_dir = abspath(args.brat_dir)
    args.jsonl = abspath(args.jsonl)
    args.out_loc = abspath(args.out_loc)
    
    main(args.brat_dir, args.jsonl, args.out_loc, args.out_prefix)