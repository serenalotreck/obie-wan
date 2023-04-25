"""
Converts the Huggingface version of the BioInfer dataset to brat standoff. Does
both train and test and saves as separate files.

Link to dataset: https://huggingface.co/datasets/bigbio/bioinfer

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
from datasets import load_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert BioInfer')

    parser.add_argument('out_loc', type=str,
            help='Path to directory to save output')
    parser.add_argument('-out_prefix', type=str, default='',
            help='Prefix to prepend to output files')
    parser.add_argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')

