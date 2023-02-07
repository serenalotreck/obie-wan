"""
Uses OpenAI's API to generate triples for abstracts.

Author: Serena G. Lotreck
"""
import argparse
from os import getenv
from os.path import abspath

if __name__ = "__main__":
    parser = argparse.ArgumentParser(description='Get GPT3 triples')

    parser.add_argument('text_dir', type=str,
            help='Path to directory where abstracts to be used are stored.')
    parser.add_argument('-out_loc', type=str,
            help='Path to save output.')
    parser.add_argument('-out_prefix', type=str,
            help='String to prepend to save file names.')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Whether or not to print output.')

    args = parser.parse_args()

    args.text_dir = abspath(args.text_dir)
    args.out_loc = abspath(args.out_loc)

    openai.api_key = getenv("OPENAI_API_KEY")

    main(args.text_dir, args.out_loc, args.out_prefix)
