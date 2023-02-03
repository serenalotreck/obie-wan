"""
Takes the manually-pasted output from ChatGPT for semantic triples, where
each document's results are in a text file with one triple per line, and
converts them to DyGEI++-formatted relations.

Author: Serena G. Lotreck
"""
import argparse
from os import listdir
from os.path import abspath, isfile, splitext


def format_preds(gpt_preds, fulltext_strs, fulltext_dir):
    """
    Format the GPT predictions as dygiepp entities and relations

    parameters:
        gpt_preds, dict: keys are file basenames and values are the lines
            from prediction files
        fulltext_strs, dict: keys are file basenames and values are the full
            text abstract strings
        fulltext_dir, str: full path to full text files

    returns:
        formatted_preds, list of dict: list of dygiepp-formatted docs
    """
    dropped_doc_names = []
    formatted_preds = []
    for doc_name, doc in gpt_preds.items():

        # Parse the triples
        triples = []
        for line in doc:
            # If there are not both outer parens, flag as not a triple and skip
            if (line[0] != '(') or (line[-1] != ')'):
                dropped_doc_names.append(doc_name)
            # Otherwise, carry on
            elif (line[0] == '(') and (line[-1] == ')'):
                # Strip outer parens
                text = line[1:-1]
                # Split at commas
                text_list = text.split(', ')
                # If there are more than 3 items after split, prompt user
                if len(text_list) != 3: # In case some have less
                # Turn into tuple

        # Find the corresponding full text

        # Pass to helper to format for dygiepp

def read_fulltext_files(fulltext_dir):
    """
    Read in the full text of the abstacts on which predictions have
    been made

    parameters:
        fulltext_dir, str: path to full text abstracts

    returns:
        fulltext_strs, dict: keys are file basenames, values are strings
            with file contents
    """
    fulltext_strs = {}
    for f in listdir(fulltext_dir):
        if isfile(f):
            name = splitext(f)[0]
            with open(f'{fulltext_dir}/{f}') as myf:
                text = my.read()
                fulltext_strs[name] = text

    return fulltext_strs


def read_pred_files(input_dir):
    """
    Read in the ChatGPT prediction files

    parameters:
        input_dir, str: directory with the files

    returns:
        gpt_preds, dict: keys are file basenames, values are list of the lines
            in the file
    """
    gpt_preds = {}
    for f in listdir(input_dir):
        if isfile(f):
            name = splitext(f)[0]
            with open(f'{input_dir}/{f}') as myf:
                lines = myf.readlines()
                lines = [l.strip() for l in lines]
                gpt_preds[name] = lines

    return gpt_preds


def main(input_dir, fulltext_dir, out_loc, out_prefix):

    # Read in the data files
    gpt_preds = read_pred_files(input_dir)

    # Read in the full text files
    fulltext_strs = read_fulltext_files(fulltext_dirs)

    # Format
    formatted_preds = format_gpt(gpt_preds, fulltext_strs, fulltext_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format ChatGPT')

    parser.add_argument('input_dir', type=str,
            help='Path to directory containing one file per document with '
            'ChatGPT annotations. File names should contain the strings '
            'used as filenames in the fulltext_dir.')
    parser.add_argument('fulltext_dir', type=str,
            help='Path to txt files with full abstracts')
    parser.add_argument('out_loc', type=str,
            help='Path to save the output')
    parser.add_argument('out_prefix', type=str,
            help='String to prepend to output file name')

    args = parser.parse_args()

    args.input_dir = abspath(args.input_dir)
    args.fulltext_dir = abspath(args.fulltext_dir)
    args.out_loc = abspath(args.out_loc)

    main(args.input_dir, args.fulltext_dir, args.out_loc, args.out_prefix)
