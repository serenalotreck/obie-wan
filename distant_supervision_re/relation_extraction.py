"""
Script to extract relations from text using a distantly-supervised
approach based on constituency parsing & BERT embeddings.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import json
import jsonlines

from abstract import Abstract
import bert_embeddings as be


def main(documents_path label_path, bert_name, out_loc, out_prefix):

    # Read in the documents
    verboseprint('\nReading in documents...')
    docs = []
    with jsonlines.open(documents_path) as reader:
        for obj in reader:
            docs.append(obj)

    # Create Abstract instances for each document
    verboseprint('\nGenerating abstract objects for each document...')
    abstracts = []
    for doc in docs:
        abst = Abstract.parse_pred_dict(doc)
        abstracts.append(abst)

    # Load BERT model
    verboseprint('\nLoading BERT model...')
    tokenizer, model = be.load_model(pretrained=bert_name)

    # Embed relation labels
    verboseprint('\nEmbedding relation labels...')
    label_dict = json.load(label_path)
    label_embed_dict = be.embed_labels(label_dict, tokenizer, model)
    label_df = pd.DataFrame.from_dict(label_embed_dict, orient='index')

    # Perform relation extraction
    verboseprint('\nPerforming relation extraction...')
    pred_output = []
    for abst in abstracts:
        abst.extract_rels(tokenizer, model, label_df)
        output = abst.rels_to_dygiepp()
        pred_output.append(output)

    # Save out the output
    verboseprint('\nSaving results...')
    out_path = f'{out_loc}/{out_prefix}_rels.jsonl'
    with jsonlines.open(out_path, 'w') as writer:
        writer.write_all(pred_output)

    verboseprint('\nDone!\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract relations')

    parser.add_argument('-documents_path', type=str,
            help='DyGIE++ formatted jsonl file with entity annotations '
            'or predictions for all documents')
    parser.add_argument('-label_path', type=str,
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

    verboseprint = print if verbose else lambda *a, **k: None

    args.documents_path = abspath(args.documents_path)
    args.label_path = abspath(args.label_path)
    args.out_loc = abspath(args.out_loc)

    main(args.documents_path, args.label_path, args.bert_name,
            args.out_loc, args.out_prefix)
