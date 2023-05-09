"""
Automates evaluating the output of GPT manual prompts.

Input requirements:
* A list of datasets to run, from the list: scierc, bioinfer, cdr, pickle
* A path to the labeled data, in a directory structured as:

 labeled_data
     ├─── <dataset shorthand>_<DEV/TRAIN/TEST>_<optional additional information>_per_doc_triples.json
     └─── etc for each dataset being evaluated

    An exception will be raised if there is more than one file with the same
    dataset shorthand
* A template job submission file, with XXXX in the job and output/error file
    names
* An output directory location for the results, will be created if it doesn't
    exist
* A date, to be appended to the output file names
* A path to a directory with model predictions, where each output file begins
    with the dataset shorthand name
* Path to a json file where keys are dataset shorthands, values are lists of
    strings containing the symmetric labels for the dataset

Writes job scripts (and will have job outputs dumped) in the CWD.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, isfile, splitext, isdir, exists, basename
from os import listdir, makedirs
from collections import defaultdict
import json
from ast import literal_eval
import subprocess


def main(datasets, labeled_data, job_template, out_loc, date, model_preds,
            sym_rels_path):

    # Read in labeled data
    verboseprint('\nReading in gold standard filenames...')
    gold_stds = {}
    for f in listdir(labeled_data):
        full_lp = f'{labeled_data}/{f}'
        if isfile(full_lp) and splitext(f)[-1] == '.json':
            dset = f.split('_')[0]
            if dset in datasets:
                assert dset not in gold_stds.keys() # Will need to make an exception for bioinfer
                gold_stds[dset] = full_lp

    # Read in the symmetric labels
    verboseprint('\nReading in symmetric relation labels...')
    with open(sym_rels_path) as myf:
        sym_rels = json.load(myf)
    sym_rels = {k: literal_eval(v) for k,v in sym_rels.items()}

    # Read in predictions
    verboseprint('\nReading in model prediction filenames...')
    preds = defaultdict(list)
    for f in listdir(model_preds):
        full_pp = f'{model_preds}/{f}'
        if isfile(full_pp) and splitext(f)[-1] == '.json':
            dset = f.split('_')[0]
            preds[dset].append(full_pp)

    # Determine if the output directory exists and make if not
    verboseprint('\nDetermining if output directory exists...')
    if not exists(out_loc):
        makedirs(out_loc)
        verboseprint(f'Output dir did not exist, was created at {out_loc}')
    else:
        verboseprint('Output dir already exists.')

    # Generate and submit eval jobs
    verboseprint('\nAssembling and submitting job scripts...')
    for dset in datasets:
        for pred_set in preds[dset]:
            # Build the out prefix
            base_pred = splitext(basename(pred_set))[0]
            base_data = splitext(basename(gold_stds[dset]))[0]
            out_prefix = f'{base_pred}_{base_data}_{date}'
            # Read in the prompt template
            with open(job_template) as myf:
                job = myf.readlines()
            # Replace the job and out/error file names
            for i, line in enumerate(job):
                if 'XXXX' in line:
                    replaced = line.replace('XXXX', out_prefix)
                    job[i] = replaced
            # Build and add the command line strings
            additions = []
            clean_cmd = (f'python remove_malformed_gpt_trips.py {pred_set}')
            additions.extend([clean_cmd, '\n'])
            cd_cmd = ('cd ../evaluation')
            additions.extend([cd_cmd, '\n'])
            cleaned_name = splitext(basename(pred_set))[0] + '_CLEAN.json'
            eval1_out_prefix = f'{out_prefix}_with_relation_labels'
            sym_rel_str = ' '.join(sym_rels[dset])
            eval_cmd1 = (f'python evaluate_triples.py {pred_set} '
                        f'{gold_stds[dset]} -out_loc {out_loc} -out_prefix '
                        f'{eval1_out_prefix} --bootstrap --check_rel_labels '
                        f'-sym_labs {sym_rel_str} -v')
            additions.extend([eval_cmd1, '\n'])
            eval2_out_prefix = f'{out_prefix}_withOUT_relation_labels'
            eval_cmd2 = (f'python evaluate_triples.py {pred_set} '
                        f'{gold_stds[dset]} -out_loc {out_loc} -out_prefix '
                        f'{eval2_out_prefix} --bootstrap -v')
            additions.extend([eval_cmd2, '\n'])
            job.append('\n')
            job.extend(additions)
            # Write out
            job_path = f'{out_prefix}_gpt_manual_eval_job.sb'
            with open(job_path, 'w') as myf:
                myf.writelines(job)
            # Submit job
            subprocess.run(['sbatch', job_path])
            verboseprint(f'Job submitted for dataset {dset}, pred file '
                            f'{basename(pred_set)}')

    verboseprint('\nDone!')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Automate GPT models')
    
    parser.add_argument('datasets', nargs='+',
            help='The datasets to run. Options are scierc, bioinfer, cdr, '
            'pickle')
    parser.add_argument('labeled_data', type=str,
            help='Path to the directory containing per doc triple gold stds')
    parser.add_argument('job_template', type=str,
            help='Path to the file to be used as a template for job scripts')
    parser.add_argument('out_loc', type=str,
            help='Path to directory to store outputs. Will be created if it '
            'doesn\'t exist')
    parser.add_argument('date', type=str,
            help='Date in the format 01Jan2023. Will be appended to output '
            'file names')
    parser.add_argument('model_preds', type=str,
            help='Path to a directory containing model prediction files. File '
            'names must begin with the dataset\'s shorthand name')
    parser.add_argument('sym_rels_path', type=str,
            help='Path to a file containing the symmetric rel labels for each '
            'dataset')
    parser.add_argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')
    
    args = parser.parse_args()
    
    args.labeled_data = abspath(args.labeled_data)
    args.job_template = abspath(args.job_template)
    args.out_loc = abspath(args.out_loc)
    args.model_preds = abspath(args.model_preds)
    args.sym_rels_path = abspath(args.sym_rels_path)
    
    verboseprint = print if args.verbose else lambda *a, **k: None
    
    main(args.datasets, args.labeled_data, args.job_template, args.out_loc,
            args.date, args.model_preds, args.sym_rels_path)