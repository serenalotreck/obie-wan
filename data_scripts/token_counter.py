"""
Given a GPT prompt file, prints the number of tokens as determined by scispacy
(as a proxy for the GPT-3.5 tokenizer).

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
import json
import spacy


def main(prompt_file):

    # Read in prompts
    with open(prompt_file) as myf:
        prompts = json.load(myf)

    # Build spacy tokenizer
    nlp = spacy.load('en_core_sci_sm')

    # Count each prompt
    tok_counts = {}
    for num, prompt_list in prompts.items():
        total_string = ''
        for part in prompt_list:
            prompt_txt = part['content']
            total_string += '' + prompt_txt
        prompt_doc = nlp(total_string)
        num_spacy_toks = len(prompt_doc)
        num_gpt_toks = num_spacy_toks * (4/3)
        tok_counts[num] = (num_spacy_toks, num_gpt_toks)

    # Print report
    print('|  prompt number  |    spacy tokens   |      GPT tokens      |')
    for num, tok_nums in tok_counts.items():
        print('---------------------------------------------------------')
        print(f'|       {num}         |       {tok_nums[0]}      |        {tok_nums[1]}         ')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Count tokens in prompt')

    parser.add_argument('prompt_file', type=str,
            help='Path to promppt file to count')

    args = parser.parse_args()

    args.prompt_file = abspath(args.prompt_file)

    main(args.prompt_file)