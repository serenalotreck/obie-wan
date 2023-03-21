"""
Script to use GPT-3.5 to simplify abstracts before use in relation extraction.

The major limitation of the implementation of the heuristic distantly
supervised algorithm at the moment is that it requires entity predictions from
an outside source; specifically, from DyGIE++. Therefore, the workflow for
using GPT-3.5 to simplify abstracts is the following:

    1. Simplify plain-text abstracts (this script)
    2. Pass to DyGIE++ for entity prediction (called from this script)
    3. Pass the DyGIE++ predictions to the heuristic algorithm (relation_extraction.py)

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, isfile
from os import listdir
import subprocess
from tqdm import trange
import sys
sys.path.append('../benchmark_scripts')
from gpt3_openai import gpt3_predict


def simplify_abstracts(abstracts, verbose):
    """
    Runs a simplification prompt using GPT-3.5.

    parameters:
        abstracts, dict: keys are file names, values are abstract text
        verbose, bool: whether or not to print the predictions as they're made

    returns:
        simp_absts, dict: keys are file names, values are simplified abstract
            texts
    """
    # Get raw preds
    prompt = ('Rewrite the following text so that each sentence only expresses '
                'one biological relationship: {abstract_txt[0]}')
    raw_preds = gpt3_predict(abstracts, prompt, fmt=True,
            model='gpt-3.5-turbo', max_tokens=2000, stop=["\\n"],
            print_preds=verbose)

    # Format raw preds
    simp_absts = {}
    for abst_name, pred in raw_preds.items():
        simp_absts[abst_name] = pred['choices'][0]['message']['content']

    return simp_absts


def main(abstract_dir, dygiepp_top_dir, run_dygiepp_path, dygiepp_path, out_loc,
        out_prefix, verbose):

    # Read in abstracts
    verboseprint('\nReading in abstracts...')
    abstracts = {}
    for f in listdir(abstract_dir):
        if isfile(f'{abstract_dir}/{f}'):
            with open(f'{abstract_dir}/{f}') as myf:
                abstracts[f] = [l.strip() for l in myf.readlines()]

    # Simplify abstracts and save out
    verboseprint('\nSimplifying abstracts...')
    simple_absts = simplify_abstracts(abstracts, verbose)
    for abst_name, abst in simple_absts.items():
        out_name = f'{out_loc}/{abst_name}'
        with open(out_name, 'w') as myf:
            myf.write(abst)

    # Pass simplified abstracts to dygiepp
    verboseprint('\nCalling dygiepp...')
    verb = "-v" if verbose else None
    dygiepp_command = [
            "python", run_dygiepp_path, dygiepp_top_dir, out_prefix,
            dygiepp_path, "--format_data", out_loc, '',
            "-models_to_run", "scierc", verb
            ]
    subprocess.run(dygiepp_command)

    verboseprint('\nDone!\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run dygiepp models")

    parser.add_argument('abstract_dir', type=str,
        help='Path to abstracts to simplify')
    parser.add_argument('dygiepp_top_dir', type=str,
        help='Path to save the dygiepp predictions. A new filetree will '
        'be created here that includes sister dirs "formatted_data", '
        '"model_predictions", and "performance". The directory specified '
        'here can already exist, but will be created if it does not.')
    parser.add_argument('run_dygiepp_path', type=str,
        help='Path to the run_dygiepp.py script in the pickle-corpus-code '
        'repo')
    parser.add_argument('dygiepp_path',type=str,
        help='Path to dygiepp repository.')
    parser.add_argument('-out_loc', type=str, default='',
        help='Path to save the simplified abstracts')
    parser.add_argument('-out_prefix', type=str, default='',
        help='Prefix to prepend to all output files from this script.')
    parser.add_argument('-v', '--verbose', action='store_true',
        help='Whether or not to print updates as the script runs.')

    args = parser.parse_args()

    args.abstract_dir = abspath(args.abstract_dir)
    args.dygiepp_top_dir = abspath(args.dygiepp_top_dir)
    args.run_dygiepp_path = abspath(args.run_dygiepp_path)
    args.dygiepp_path = abspath(args.dygiepp_path)
    args.out_loc = abspath(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None

    main(args.abstract_dir, args.dygiepp_top_dir, args.run_dygiepp_path,
            args.dygiepp_path, args.out_loc, args.out_prefix, args.verbose)
