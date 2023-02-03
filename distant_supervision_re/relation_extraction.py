"""
Script to extract relations from text using a distantly-supervised
approach based on constituency parsing & BERT embeddings.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import json
import jsonlines
import pandas as pd
from tqdm import tqdm
import spacy
import benepar
from abstract import Abstract
import bert_embeddings as be


def main(documents_path, label_path, bert_name, out_loc, out_prefix):

    # Read in the documents
    verboseprint('\nReading in documents...')
    docs = []
    with jsonlines.open(documents_path) as reader:
        for obj in reader:
            docs.append(obj)

    # Load spacy model for constituency parsing
    verboseprint('\nLoading spacy constituency parser...')
    nlp = spacy.load('en_core_sci_sm')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    # Create Abstract instances for each document
    verboseprint('\nGenerating abstract objects for each document...')
    abstracts = []
    for doc in tqdm(docs):
        abst = Abstract.parse_pred_dict(doc, nlp)
        abstracts.append(abst)

    # Load BERT model
    verboseprint('\nLoading BERT model...')
    verboseprint(f'Model name: {bert_name}')
    tokenizer, model = be.load_model(pretrained=bert_name)

    # Embed relation labels
    verboseprint('\nEmbedding relation labels...')
    with open(label_path) as infile:
        label_dict = json.load(infile)
    label_embed_dict = be.embed_labels(label_dict, tokenizer, model)
    label_df = pd.DataFrame.from_dict(label_embed_dict, orient='index')

    # Perform relation extraction
    verboseprint('\nPerforming relation extraction...')
    pred_output = []
    skipped = 0
    totals = 0
    skipped_cats = {}
    for abst in tqdm(abstracts):
        skip, total = abst.extract_rels(tokenizer, model, label_df)
        skipped += skip
        totals += total
        skipped_cats[abst] = abst.skipped_cats
        output = abst.rels_to_dygiepp()
        pred_output.append(output)
    skipped_cat_out = f'{out_loc}/{out_prefix}_skipped_sentence_cats.json'
    verboseprint(f'{skipped} of {totals} candidate sentences  were dropped '
            'due to tokenization mismatches. Dropped sentence categorizations '
            f'are being saved to {skipped_cat_out}')

    # Save out the output
    verboseprint('\nSaving results...')
    with open(skipped_cat_out, "w") as outfile:
        json.dump(skipped_cats, outfile)
    out_path = f'{out_loc}/{out_prefix}_rels.jsonl'
    with jsonlines.open(out_path, 'w') as writer:
        writer.write_all(pred_output)

    verboseprint('\nDone!\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract relations')

    parser.add_argument('documents_path', type=str,
            help='DyGIE++ formatted jsonl file with entity annotations '
            'or predictions for all documents')
    parser.add_argument('label_path', type=str,
            help='Path to a json file where keys are the relation '
            'labels, and values are lists of strings, where each string '
            'is an example sentence that literally uses the relation label.')
    parser.add_argument('-bert_name', type=str,
            default="alvaroalon2/biobert_genetic_ner",
            help='Name of pretrained BERT model from the huggingface '
            'transformers library. Default is the BioBERt genetic NER '
            'model.')
    parser.add_argument('-out_loc', type=str,
            help='Path to save the output')
    parser.add_argument('-out_prefix', type=str,
            help='Prefix to add to saved files')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Whether or not to print updates')

    args = parser.parse_args()

    verboseprint = print if args.verbose else lambda *a, **k: None

    args.documents_path = abspath(args.documents_path)
    args.label_path = abspath(args.label_path)
    args.out_loc = abspath(args.out_loc)

    main(args.documents_path, args.label_path, args.bert_name,
            args.out_loc, args.out_prefix)
