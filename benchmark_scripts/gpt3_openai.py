"""
Uses OpenAI's API to generate triples for abstracts.

Author: Serena G. Lotreck
"""
import argparse
from os import getenv, listdir
from os.path import abspath, isfile
from ast import literal_eval
import spacy
import openai
from tqdm import tqdm
import jsonlines
import json
import sys
sys.path.append('../distant_supervision_re/')
import bert_embeddings as be
from abstract import Abstract


def embed_relation(sent_tok, rel, label_df, model, tokenizer):
    """
    TODO adapt this to work without sentence indices
    initial thought is to use the entire abstract instead of the sentence


    Performs the same relation embedding as relation_embedding.py to generate a
    standardized relation label for a given triple.

    parameters:
        sent_tok, list of str: tokenized sentence
        rel, str: relation to embed
        label_df, pandas df: rows are labels, columns are vector elements
        model, huggingface BERT model: model to use
        tokenizer, huggingface BERT tokenizer: tokenizer to use
    """
    sent_text = ' '.join(self.sentences[sent_tok])
    embedding = be.get_phrase_embedding(sent_text, rel,
        tokenizer, model)
    label = Abstract.compute_label(label_df, embedding)
    if label == '':
        return None
    else:
        return label


def process_preds(abstracts, raw_preds, bert_name, label_path, embed_rels=False):
    """
    Pulls text output from predictions, and calls helpers to embed relations if
    embed_rels=True. TODO implement relation embeddings

    parameters:
        abstracts, dict: keys are filenames, values are strings with abstracts
        raw_preds, dict: keys are filenames, values are raw output of gpt3
        bert_name, str: name of BERT model to use to embed relations
        label_path, str: path to a file with relation labels and example
            sentences containing them
        embed_rels, bool: True if relations should be converted to standard
            labels by embedding

    returns:
        trip_preds, dict of list: keys are filenames and values are lists of
            the triple tuples
    """
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

    # Format preds
    trip_preds = {}
    for fname, abstract_pred in tqdm(raw_preds.items()):
        # Check that there's only one prediction
        assert len(abstract_pred['choices']) == 1
        # Pull out the text from the prediction
        pred_text = abstract_pred['choices'][0]['text']
        # Read the output literally to get triples
        doc_triples = [literal_eval(t) for t in pred_text.split('\n')
                if t != '']
        if embed_rels:
            pass
            ## TODO fix the embed rels function to work without having
            ## sentences associated with triples
        # Add to preds dict
        trip_preds[fname] = doc_triples

    return trip_preds


def gpt3_predict(abstracts, prompt, fmt=False, model='text-davinci-003',
        max_tokens=2048, temp=0, stop='\n'):
    """
    Passes each abstract to gpt3 and returns the raw output.

    parameters:
       abstracts, dict: keys are file names and values are lines from the
            abstracts
        prompt, str: string to pass as the prompt. If it includes the substring
            "{abstract_txt}" in order to be able to use with each abstract
            separately, must provide fmt=True
        fmt, bool: whether or not the prompt string needs to be formatted
        model, str: name of the model to use, default is the full GPT3 model
        max_tokens, int: maximum number of tokens to request from model. Will
            change based on what model is requested, and must be larger than
            (number of tokens in request + number of tokens in respose)
        temp, int: temperature parameter to pass to gpt3, default is 0
        stop, str: stop character to pass to gpt3

    returns:
        raw_preds, dict: keys are file names and values are the output of gpt3
            predictions
    """
    raw_preds = {}
    for fname, abstract_txt in tqdm(abstracts.items()):
        # Add the abstract text to the prompt if required
        if fmt:
            formatted_prompt = prompt.format(abstract_txt=abstract_txt)
        else:
            formatted_prompt = prompt
        verboseprint(f'Prompt for {fname}:\n{formatted_prompt}')

        # Get predictions
        response = openai.Completion.create(
            model=model,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temperature=temp,
            stop=stop
                )

        # Add to list
        raw_preds[fname] = response
        verboseprint('\n----------------- Predictions -------------------------\n')
        verboseprint(response['choices'][0]['text'])
        verboseprint('\n---------------------------------------------------\n')

    return raw_preds


def main(text_dir, embed_rels, label_path, bert_name, out_loc, out_prefix):

    # Read in the abstracts
    verboseprint('\nReading in abstracts...')
    abstracts = {}
    for f in tqdm(listdir(text_dir)):
        if isfile(f'{text_dir}/{f}'):
            path = f'{text_dir}/{f}'
            with open(path) as myf:
                abstract =[l.strip() for l in  myf.readlines()]
                abstracts[f] = abstract

    # Prompt GPT3 for each abstract
    verboseprint('\nGenerating predictions...')
    prompt = ('Extract the biological relationships from '
     'the following text as (Subject, Predicate, Object) triples in the format ("Subject", "predicate", "Object"): '
     '{abstract_txt[0]}')
    raw_preds = gpt3_predict(abstracts, prompt, fmt=True, model='text-davinci-003',
            max_tokens=1000, stop=["\\n"])

    # Format output in dygiepp format
    verboseprint('\nFormatting predictions...')
    final_preds = process_preds(abstracts, raw_preds, bert_name, label_path,
            embed_rels)

    # Save output
    verboseprint('\nSaving output...')
    out_path = f'{out_loc}/{out_prefix}_gpt3_preds.json'
    with open(out_path, 'w') as myf:
        json.dump(final_preds, myf)
    verboseprint(f'Predictions saved to {out_path}')

    print('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get GPT3 triples')

    parser.add_argument('text_dir', type=str,
            help='Path to directory where abstracts to be used are stored')
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
        assert label_path != '', ('--embed_rels was specified, -label_path '
                'must be provided.')
        args.label_path = abspath(args.label_path)
    args.text_dir = abspath(args.text_dir)
    args.out_loc = abspath(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None

    openai.api_key = getenv("OPENAI_API_KEY")

    main(args.text_dir, args.embed_rels, args.label_path, args.bert_name,
            args.out_loc, args.out_prefix)
