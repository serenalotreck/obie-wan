"""
Script to perform open relation extraction on text using Stanford OpenIE and
convert the output to the dygiepp format.

Author: Serena G. Lotreck
Originally written for the serenalotreck/knowledge-graph repository
"""
import argparse
from os import listdir
from os.path import abspath, join, splitext, basename
from collections import OrderedDict

import jsonlines
import json
import pandas as pd
import stanza
stanza.install_corenlp()
from stanza.server import CoreNLPClient
from openie import StanfordOpenIE

import sys
sys.path.append('distant_supervision_re/')
import bert_embeddings as be


def embed_relation(sent, rel, label_df, model, tokenizer):
    """
    Performs the same relation embedding as relation_embedding.py to generate a
    standardized relation label for a given triple.

    parameters:
        sent, list of str: sentence from which the triple was extracted
        rel, str: the relation to embed
        label_df, pandas df: rows are labels, columns are vector elements
        model, huggingface BERT model: model to use
        tokenizer, huggingface BERT tokenizer: tokenizer to use
    """
    sent_text = ' '.join(sent)
    embedding = be.get_phrase_embedding(sent_text, rel,
        tokenizer, model)
    label = Abstract.compute_label(label_df, embedding)
    if label == '':
        return None
    else:
        return label


def get_doc_rels(ann, embed_rels, label_path, model, tokenizer):
    """
    Get relation annotations from openie objects. Embeds relations to get
    standardized relation labels is embeds_relations is true.

    parameters:
        ann, output from OpenIE
        embed_rels, bool: True if relations should be converted to standard
            labels by embedding
        label_path, str: path to a file with relation labels and example
            sentences containing them
        model, HuggingFace BERT model: model to use for relations embeddings
        tokenizer, HuggingFace BERT tokenizer: tokenizer to use for relations
            embeddings

    returns:
        rels, list of list of lists: one list per sentence, one list per
            relation annotation in sentence
    """
    prev_toks = 0 # Keep track of the number of tokens we've seen
    rels = []
    for sent in ann["sentences"]:
        sent_rels = []
        triples = sent["openie"]
        for triple in triples:
            # Dygiepp token indices are for the whole document, while stanford
            # openIE indices are on a sentence-level. Need to add the previous
            # number of tokens we've seen in order to get document-level
            # indices. The end indices in dygiepp are inclusive, whereas they
            # are not in stanford openIE, so also need to subtract 1 from end
            # idx
            if embed_rels:
                label = embed_relation(sent, triple["relation"], label_df,
                        model, tokenizer)
                rel = [triple["subjectSpan"][0] + prev_toks,
                        triple["subjectSpan"][1] + prev_toks - 1,
                        triple["objectSpan"][0] + prev_toks,
                        triple["objectSpan"][1] + prev_toks - 1,
                        label]
            else:
                rel = [triple["subjectSpan"][0] + prev_toks,
                        triple["subjectSpan"][1] + prev_toks - 1,
                        triple["objectSpan"][0] + prev_toks,
                        triple["objectSpan"][1] + prev_toks - 1,
                        triple["relation"]]
            sent_rels.append(rel)
        rels.append(sent_rels)
        prev_toks += len(sent["tokens"])

    return rels


def get_doc_ents(ann):
    """
    Get entity annotations from ann openie objects.

    parameters:
        ann, output from OpenIE

    returns:
        ents, list of list of lists: one list per sentence, one list per
            entity annotation in sentence
    """
    prev_toks = 0 # Keep track of the number of tokens we've seen
    ents = []
    for sent in ann["sentences"]:
        sent_ents = []
        triples = sent["openie"]
        for triple in triples:
            # Dygiepp token indices are for the whole document, while stanford
            # openIE indices are on a sentence-level. Need to add the previous
            # number of tokens we've seen in order to get document-level
            # indices. The end indices in dygiepp are inclusive, whereas they
            # are not in stanford openIE, so also need to subtract 1 from end
            # idx
            for part in ['subject', 'object']:
                ent = [triple[f"{part}Span"][0] + prev_toks,
                       triple[f"{part}Span"][1] + prev_toks - 1,
                       'ENTITY']
                sent_ents.append(ent)
        # Remove duplicate entities that participated in multiple relations
        sent_ents = [tuple(x) for x in sent_ents]
        sent_ents = list(OrderedDict.fromkeys(sent_ents))
        sent_ents = [list(x) for x in sent_ents]

        ents.append(sent_ents)
        prev_toks += len(sent["tokens"])

    return ents


def get_doc_sents(ann):
    """
    Reconstruct sentences from the index object in the ann object.

    parameters:
        ann, output from OpenIE

    returns:
        sents, list of lists: one list per sentence containing tokens as
            elements
    """
    sents = []
    for sent in ann["sentences"]:
        sent_list = []
        for idx in sent["tokens"]:
            sent_list.append(idx["originalText"])
        sents.append(sent_list)

    return sents


def openie_to_dygiepp(ann, doc_key, embed_rels, label_path, model, tokenizer):
    """
    Convert the output of StanfordNLP OpenIE to dygiepp format. Subtracts 1
    from all token offsets to convert 1-indexted token offsets to 0-indexed
    token offsets.

    parameters:
        ann, output form stanza.server.CoreNLPClient annotate function:
            annotation to convert
        doc_key, str: string to identify the document
        embed_rels, bool: True if relations should be converted to standard
            labels by embedding
        label_path, str: path to a file with relation labels and example
            sentences containing them
        model, HuggingFace BERT model: model to use for relations embeddings
        tokenizer, HuggingFace BERT tokenizer: tokenizer to use for relations
            embeddings

    returns:
        json, dict: dygiepp formatted json for the annotated file
    """
    # Get doc sentences
    sents = get_doc_sents(ann)

    # Get doc ents
    ents = get_doc_ents(ann)

    # Get doc rels
    rels = get_doc_rels(ann, embed_rels, label_path, model, tokenizer)

    # Make the json
    json = {}
    json["doc_key"] = doc_key
    json["dataset"] = 'scierc' # Doesn't matter what this is for eval
    json["sentences"] = sents
    json["predicted_ner"] = ents
    json["predicted_relations"] = rels

    return json


def main(data_dir, to_annotate, affinity_cap, embed_rels, label_path,
        bert_name, out_loc, out_prefix):

    properties = {'openie.affinity_probability_cap': affinity_cap}

    # Load models for relations embedding outside of loop, if required
    if embed_rels:
        # Load the BERT model, and embed relation labels
        verboseprint('\nLoading BERT model...')
        verboseprint(f'Model name: {bert_name}')
        tokenizer, model = be.load_model(pretrained=bert_name)

        # Embed relation labels
        verboseprint('\nEmbedding relation labels...')
        with open(label_path) as infile:
            label_dict = json.load(infile)
        label_embed_dict = be.embed_labels(label_dict, tokenizer, model)
        label_df = pd.DataFrame.from_dict(label_embed_dict, orient='index')
    else:
        tokenizer = None
        model = None
        label_df = None

    dygiepp_jsonl = []
    with CoreNLPClient(annotators=["openie"], output_format="json") as client:
        for doc in to_annotate:

            # Get the doc_key
            doc_key = splitext(basename(doc))[0]

            # Read in the text
            with open(doc) as f:
                text = " ".join(f.read().split('\n'))

            # Perform OpenIE
                ann = client.annotate(text)

            # Convert output to dygiepp format
            jsonl_anns = openie_to_dygiepp(ann, doc_key, embed_rels,
                label_df, model, tokenizer)
            dygiepp_jsonl.append(jsonl_anns)

    # Write out dygiepp-formatted output
    output_name = f'{out_loc}/{out_prefix}_openie_preds.jsonl'
    with jsonlines.open(output_name, 'w') as writer:
        writer.write_all(dygiepp_jsonl)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Use Stanford OpenIE for '
            'relation extraction')

    parser.add_argument('data_dir', type=str, help='Path to directory with '
            'files to annotate.')
    parser.add_argument('-affinity_cap', type=float, help='"Hyperparameter '
            'denoting the min fraction ofd the time an edge should occur in a '
            'context in order to be considered unremoveable from the graph", '
            'in the original OpenIE experiments this is set to 1/3.',
            default=1/3)
    parser.add_argument('--embed_rels', action='store_true',
            help='Whether or not to embed relations to get standardized '
            'labels. If passed, must also pass label_path')
    parser.add_argument('-label_path', type=str, default='',
            help='Path to a json file where keys are the relation '
            'labels, and values are lists of strings, where each string '
            'is an example sentence that literally uses the relation label.')
    parser.add_argument('-bert_name', type=str,
            default="alvaroalon2/biobert_genetic_ner",
            help='Name of pretrained BERT model from the huggingface '
            'transformers library. Default is the BioBERt genetic NER '
            'model.')
    parser.add_argument('-out_loc', type=str,
            help='Path to save output')
    parser.add_argument('-out_prefix', type=str,
            help='String to prepend to save file names')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Whether or not to print output.')

    args = parser.parse_args()

    if args.embed_rels:
        assert args.label_path != '', ('--embed_rels was specified, -label_path '
                'must be provided.')
        args.label_path = abspath(args.label_path)
    args.data_dir = abspath(args.data_dir)
    args.out_loc = abspath(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None
    print('\n\n\n\n START OF SCRIPT')
    print(f'value of --verbose: {args.verbose}')

    to_annotate = [join(args.data_dir, f) for f in listdir(args.data_dir) if
            f.endswith('.txt')]

    main(args.data_dir, to_annotate, args.affinity_cap, args.embed_rels,
            args.label_path, args.bert_name, args.out_loc, args.out_prefix)
