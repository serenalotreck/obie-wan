"""
Uses OpenAI's API to generate triples for abstracts.

Author: Serena G. Lotreck
"""
import argparse
from os import getenv, listdir
from os.path import abspath, isfile, splitext
from ast import literal_eval
import spacy
import openai
from tqdm import tqdm
import jsonlines
import json
import pandas as pd
import sys
sys.path.append('../distant_supervision_re/')
import bert_embeddings as be
import phrase_utils as pu


def process_preds(abstracts, raw_preds):
    """
    Pulls text output from predictions.

    parameters:
        abstracts, dict: keys are filenames, values are strings with abstracts
        raw_preds, dict: keys are filenames, values are raw output of gpt3

    returns:
        trip_preds, dict of list: keys are filenames and values are lists of
            the triple tuples
    """
    # Format preds
    trip_preds = {}
    for fname, abstract_pred in tqdm(raw_preds.items()):
        # Check that there's only one prediction
        assert len(abstract_pred['choices']) == 1
        # Pull out the text from the prediction
        pred_text = abstract_pred['choices'][0]['message']['content']
        # Read the output literally to get triples
        try:
            doc_triples = [literal_eval(t) for t in pred_text.split('\n')
                if t != '']
            doc_triples[0][1]
        except IndexError:
            spl = pred_text.split(", \n")
            doc_triples = [literal_eval(t) for t in pred_text.split(', \n')]
        except SyntaxError:
            doc_triples = abstract_pred['choices'][0]['message']['content'] # For consistency with manual prompts code
        except ValueError:                                                 
            doc_triples = abstract_pred['choices'][0]['message']['content'] 
        trip_preds[fname] = doc_triples

    return trip_preds


def gpt3_predict(abstracts, prompt, fmt=False, model='text-davinci-003',
        max_tokens=2048, temp=0, stop='\n', print_preds=False):
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
        print_preds, bool: whether or not to print predictions

    returns:
        raw_preds, dict: keys are file names and values are the output of gpt3
            predictions
    """
    if print_preds:
        verboseprint = print
    else:
        verboseprint = lambda *a, **k: None

    raw_preds = {}
    for fname, abstract_txt in tqdm(abstracts.items()):
        # Add the abstract text to the prompt if required
        if fmt:
            formatted_prompt = prompt.format(abstract_txt=abstract_txt)
        else:
            formatted_prompt = prompt
        verboseprint(f'Prompt for {fname}:\n{formatted_prompt}')

        # Get predictions
        predicted = False
        num_fails = 0
        while not predicted:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    max_tokens=max_tokens,
                    temperature=temp,
                    stop=stop
                    )
                predicted = True
            except openai.error.RateLimitError:
                num_fails += 1
                print(f'On attempt {num_fails} for doc {fname}')

        # Add to list
        raw_preds[fname] = response
        verboseprint('\n----------------- Predictions -------------------------\n')
        verboseprint(response['choices'][0]['message']['content'])
        verboseprint('\n---------------------------------------------------\n')

    return raw_preds


def main(text_dir, prompt_file, out_loc, out_prefix):

    # Read in the abstracts
    verboseprint('\nReading in abstracts...')
    abstracts = {}
    for f in tqdm(listdir(text_dir)):
        if isfile(f'{text_dir}/{f}'):
            path = f'{text_dir}/{f}'
            with open(path) as myf:
                abstract = [l.strip() for l in  myf.readlines()]
                fname = splitext(f)[0]
                abstracts[fname] = abstract

    # Read in the prompt
    verboseprint('\nReading in the requested prompt...')
    with open(prompt_file) as myf:
        prompt = myf.read()
    verboseprint(f'Snapshot of requested prompt: {prompt}')

    # Prompt GPT3 for each abstract
    verboseprint('\nGenerating predictions...')
    raw_preds = gpt3_predict(abstracts, prompt, fmt=True, model='gpt-3.5-turbo',
            max_tokens=1000, stop=["\\n"], print_preds=True)

    # Format output in dygiepp format
    verboseprint('\nFormatting predictions...')
    final_preds = process_preds(abstracts, raw_preds)

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
    parser.add_argument('prompt_file', type=str,
            help='Path to txt file with prompt string')
    parser.add_argument('out_loc', type=str,
            help='Path to save output')
    parser.add_argument('out_prefix', type=str,
            help='String to prepend to save file names')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Whether or not to print output.')

    args = parser.parse_args()

    args.text_dir = abspath(args.text_dir)
    args.prompt_file = abspath(args.prompt_file)
    args.out_loc = abspath(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None

    openai.api_key = getenv("OPENAI_API_KEY")

    main(args.text_dir, args.prompt_file, args.out_loc, args.out_prefix)
