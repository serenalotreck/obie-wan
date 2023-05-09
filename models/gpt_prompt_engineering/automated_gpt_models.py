"""
Automates running GPT manual prompts.

Input requirements:
* A list of datasets to run, from the list: scierc, bioinfer, cdr, pickle
* A prompt directory, which contains prompt files that begin with the shorthand
    for the dataset name followed by an underscore
* A template job submission file, with XXXX in the job and output/error file
    names
* An output directory location for the results, will be created if it doesn't
    exist
* A date, to be appended to the output file names
* A path to a directory containing the unlabeled txt files on which to apply the
    prompts, where each directory of txt files begins with the dataset's
        shorthand name and an underscore

Writes job scripts (and will have job outputs dumped) in the CWD.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, isfile, splitext, isdir, exists, basename
from os import listdir, makedirs
from collections import defaultdict
import subprocess


def main(datasets, prompt_path, job_template, out_loc, date, apply_data):
    
    # Read in the prompt file paths and sort by dataset
    verboseprint('\nDeterminig which prompts to use based on datasets...')
    prompts = defaultdict(list)
    for prompt_item in listdir(prompt_path):
        full_pp = f'{prompt_path}/{prompt_item}'
        if isfile(full_pp) and (splitext(prompt_item)[-1].lower() == '.json'):
            dset = prompt_item.split('_')[0]
            if dset in datasets:
                prompts[dset].append(full_pp)
    
    # Read in the dataset directories and sort
    verboseprint('\nObtaining paths to datasets to use...')
    dataset_dirs = {}
    for dataset_dir in listdir(apply_data):
        full_dp = f'{apply_data}/{dataset_dir}'
        if isdir(full_dp):
            dset = dataset_dir.split('_')[0]
            if dset in datasets:
                # Enforce only one directory per dataset
                assert dset not in dataset_dirs.keys()
                dataset_dirs[dset] = full_dp
    
    # Determine if the output directory exists and make if not
    verboseprint('\nDetermining if output directory exists...')
    if not exists(out_loc):
        makedirs(out_loc)
        verboseprint(f'Output dir did not exist, was created at {out_loc}')
    else:
        verboseprint('Output dir already exists.')
    
    # Run models
    verboseprint('\nAssembling and submitting job scripts...')
    for dset in datasets:
        for prompt in prompts[dset]:
            # Build the out prefix
            base_prompt = splitext(basename(prompt))[0]
            base_data = basename(dataset_dirs[dset])
            out_prefix = f'{base_prompt}_{base_data}_{date}'
            # Read in the prompt template
            with open(job_template) as myf:
                job = myf.readlines()
            # Replace the job and out/error file names
            for i, line in enumerate(job):
                if 'XXXX' in line:
                    replaced = line.replace('XXXX', out_prefix)
                    job[i] = replaced
            # Build and add the command line string
            cmd = (f'python run_prompts.py {dataset_dirs[dset]} {prompt} '
                    f'-out_loc {out_loc} -out_prefix {out_prefix} -v -c')
            job.append('\n')
            job.append(cmd)
            # Write out
            job_path = f'{out_prefix}_gpt_manual_job.sb'
            with open(job_path, 'w') as myf:
                myf.writelines(job)
            # Submit job
            subprocess.run(['sbatch', job_path])
            verboseprint(f'Job submitted for dataset {dset}, prompt '
                            f'{basename(prompt)}')

    verboseprint('\nDone!')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Automate GPT models')
    
    parser.add_argument('datasets', nargs='+',
            help='The datasets to run. Options are scierc, bioinfer, cdr, '
            'pickle')
    parser.add_argument('prompt_path', type=str,
            help='Path to the directory containing GPT prompt files')
    parser.add_argument('job_template', type=str,
            help='Path to the file to be used as a template for job scripts')
    parser.add_argument('out_loc', type=str,
            help='Path to directory to store outputs. Will be created if it '
            'doesn\'t exist')
    parser.add_argument('date', type=str,
            help='Date in the format 01Jan2023. Will be appended to output '
            'file names')
    parser.add_argument('apply_data', type=str,
            help='Path to a directory containing a subdirectory of unlabeled '
            'txt files for each dataset. Subdirectory names must begin with '
            'the dataset\'s shorthand name')
    parser.add_argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')
    
    args = parser.parse_args()
    
    args.prompt_path = abspath(args.prompt_path)
    args.job_template = abspath(args.job_template)
    args.out_loc = abspath(args.out_loc)
    args.apply_data = abspath(args.apply_data)
    
    verboseprint = print if args.verbose else lambda *a, **k: None
    
    main(args.datasets, args.prompt_path, args.job_template, args.out_loc,
            args.date, args.apply_data)