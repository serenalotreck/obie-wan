"""
Script to run a series of sequential prompts over a set of texts. See
PROMPTS.md for a description of the required prompt format.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, isfile, splitext
from os import getenv, listdir
import json
from ast import literal_eval
from copy import deepcopy
from tqdm import tqdm
import openai


def find_format(prompt):
    """
    Finds the index of the message requiring string formatting.

    parameters:
        current_prompt, list of dict: each dict is a message item, and only one
            message item contains the substring '{text}'

    returns:
        idx, int: index of the message that needs changing
    """
    for idx, msg in enumerate(prompt):
        if '{text}' in msg["content"]:
            return idx


def gpt_predict(abstracts, prompts, model='gpt-3.5-turbo'):
    """
    Builds sequential predictions based on prompts.

    parameters:
        abstracts, dict: keys are file basenames, values are text
        prompts, dict: keys are the integer number of the
            prompt, and values are message-formatted arrays, with the final
            prompt being the only one that needs string formatting, and
            contains the variable {text}
        model, str: model to use, default is GPT-3.5

    returns:
        preds, dict: keys are abstract file basenames, values are lists of
            tuples of triple predictions.
    """
    preds = {}
    for doc_name, doc in tqdm(abstracts.items()):
        prev_out = doc
        commonsenseprint(f'\nInitial doc:\n{doc}')
        for prompt_num, prompt in prompts.items():

            commonsenseprint(f'\nCurrently on prompt {prompt_num}: {prompt}')

            # If the previous output is a list, we need to iterate over the list
            new_prev_out = []
            if isinstance(prev_out, list):

                commonsenseprint(f'\nPrevious output is a list:\n{prev_out}')

                for i in prev_out:
                    # Put the previous prompt's output (or initial doc) into the
                    # query of the current prompt that requires formatting
                    current_prompt = deepcopy(prompt)
                    idx = find_format(current_prompt)
                    current_prompt[idx]["content"] = current_prompt[idx]["content"].format(text=i)

                    commonsenseprint('\nPrompt when formatted with current '
                                        f'element of prev_out:\n{current_prompt}')

                    # Get model predictions for these prompts
                    predicted = False
                    num_fails = 0
                    while not predicted:
                        try:
                            response = openai.ChatCompletion.create(
                                model=model,
                                messages=current_prompt,
                                temperature=0)
                            predicted = True
                        except openai.error.RateLimitError:
                            num_fails += 1
                            print(f'On attempt {num_fails} for doc {doc_name}')

                    commonsenseprint(f'\nRaw response:\n{response}')

                    # Pull out the model response
                    try:
                        literal_response = literal_eval(response['choices'][0]['message']['content'])
                    except SyntaxError:
                        literal_response = response['choices'][0]['message']['content']
                    except ValueError:
                        literal_response = response['choices'][0]['message']['content']
                    commonsenseprint(f'\nLiteral response:\n{literal_response}')

                    # If it's a list, just extend the prev_out instead of
                    # append
                    if isinstance(literal_response, list):
                        new_prev_out.extend(literal_response)
                    else:
                        new_prev_out.append(literal_response)

            else:

                commonsenseprint(f'Previoius output is not a list:\n{prev_out}')

                # Put the previous prompt's output into the last query of the current prompt
                current_prompt = deepcopy(prompt)
                idx = find_format(current_prompt)
                current_prompt[idx]["content"] = current_prompt[idx]["content"].format(text=prev_out)

                commonsenseprint(f'\nPrompt when formatted with prev_out:\n{current_prompt}')

                # Get model predictions for these prompts
                predicted = False
                num_fails = 0
                while not predicted:
                    try:
                        response = openai.ChatCompletion.create(
                            model=model,
                            messages=current_prompt,
                            temperature=0)
                        predicted = True
                    except openai.error.RateLimitError:
                        num_fails += 1
                        print(f'On attempt {num_fails} for doc {doc_name}')
                commonsenseprint(f'\nRaw response:\n{response}')

                # Pull out the model response
                try:
                    literal_response = literal_eval(response['choices'][0]['message']['content'])
                except SyntaxError:
                    literal_response = response['choices'][0]['message']['content']
                except ValueError:
                    literal_response = response['choices'][0]['message']['content']
                
                commonsenseprint(f'\nLiteral response:\n{literal_response}')

                # If it's a list, just extend the prev_out instead of append
                if isinstance(literal_response, list):
                    new_prev_out.extend(literal_response)
                else:
                    new_prev_out.append(literal_response)

            # Make the new_prev_out into the current prev_out
            prev_out = new_prev_out

            # If this is the final prompt, put the repsonse in the prediction
            # list for this abstract
            if prompt_num == sorted(prompts.keys())[-1]:
                commonsenseprint(f'This was the last prompt. Preds:\n{new_prev_out}')
                preds[doc_name] = new_prev_out

    return preds


def read_abstracts(text_dir):
    """
    Reads in plain text abstracts for prediction.

    parameters:
        text_dir, str: directory where files are located

    returns:
        abstracts, dict: keys are file basenames and values are text
    """
    abstracts = {}
    for f in tqdm(listdir(text_dir)):
        if isfile(f'{text_dir}/{f}'):
            path = f'{text_dir}/{f}'
            with open(path) as myf:
                abstract = myf.read()
                fname = splitext(f)[0]
                abstracts[fname] = abstract
    return abstracts


def main(text_dir, prompts_path, model, out_loc, out_prefix):

    # Read in the abstracts
    verboseprint('\nReading in abstracts...')
    abstracts = read_abstracts(text_dir)
    verboseprint('A snapshot of the abstract names to be used: '
                f'{list(abstracts.keys())[:5]}')

    # Read in the prompts
    verboseprint('\nReading in prompts...')
    with open(prompts_path) as myf:
        prompts = json.load(myf)
    verboseprint(f'The prompts you\'ve requested:\n{prompts}')

    # Make sequential predictions
    verboseprint('\nMaking predictions...')
    preds = gpt_predict(abstracts, prompts, model)

    # Save predictions
    verboseprint('\nSaving predictions...')
    out_path = f'{out_loc}/{out_prefix}_{model}_sequentrial_preds.json'
    with open(out_path, 'w') as myf:
        json.dump(preds, myf)
    verboseprint(f'Predictions saved to {out_path}')

    verboseprint('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get ChatGPT '
            'sequentially-generated triples')

    parser.add_argument('text_dir', type=str,
            help='Path to directory where abstracts to be used are stored')
    parser.add_argument('prompts_path', type=str,
            help='Path to json file with sequential prompts')
    parser.add_argument('-model', type=str, default='gpt-3.5-turbo',
            help='Model to use. Default is GPT-3.5 turbo')
    parser.add_argument('-out_loc', type=str,
            help='Path to save output')
    parser.add_argument('-out_prefix', type=str,
            help='String to prepend to save file names')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Whether or not to print progress.')
    parser.add_argument('-c', '--common_sense', action='store_true',
            help='Whether or not to print all intermediate predictions for a '
            'common-sense check of prompt efficacy')

    args = parser.parse_args()

    args.text_dir = abspath(args.text_dir)
    args.prompts_path = abspath(args.prompts_path)
    args.out_loc = abspath(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None
    commonsenseprint = print if args.common_sense else lambda *a, **k: None

    openai.api_key = getenv("OPENAI_API_KEY")

    main(args.text_dir, args.prompts_path, args.model, args.out_loc, args.out_prefix)
