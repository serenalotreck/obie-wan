"""
Script to extract non-text-bound relation triples from the BioCreative V CDR
datset in XML format. Ignores titles and operates on abstracts. Unique document
ID's are the numerical document ID encoded in the BioC dataset. Additionally
saves out the text of each abstract as a separate document for use in
prediction.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath, exists
from os import mkdir
import json
from tqdm import tqdm
from bioc import biocxml


def get_trips(doc):
    """
    Pulls unbound triples from a BioCDocument.

    parameters:
        doc, BioCDocument: document to pull

    returns:
        trips, list of tuple: triples
    """
    trips = []
    assert len(doc.passages) == 2 # Want to make sure none break the rule
    all_anns = []
    for passage in doc.passages:
        anns = passage.annotations
        all_anns.extend(anns)
    for rel in doc.relations:
        # Get relation type
        rel_type = rel.infons['relation']
        assert rel_type == 'CID', f'Different rel type: {rel_type}' # Want to
                                                                    # make sure
                                                                    # they're
                                                                    # all the
                                                                    # same
        # Get entity MESH id's
        ent_ids = (rel.infons['Chemical'], rel.infons['Disease']) # Doesn't
                                                                  # maintain order
        # Get the MESH id's text names
        ent_txt_dict = {}
        for ann in all_anns:
            ann_txt = ann.text
            ann_ids = ann.infons['MESH'].split('|') # There can be multiples separated by pipes
            for ind_id in ann_ids:
                ent_txt_dict[ind_id] = ann_txt
        # Match MESH id's to entity text
        ent1 = ent_txt_dict[ent_ids[0]]
        ent2 = ent_txt_dict[ent_ids[1]]
        # Build triple and add to list
        trip = (ent1, rel_type, ent2)
        trips.append(trip)

    return trips


def get_abstract(doc, doc_key, txt_out_loc):
    """
    Extracts and saves the abstract of a BioCDocument as a txt file.

    parameters:
        doc, BioCDocument instance: document to save
        doc_key, int: unique document ID to use as file name
        txt_out_loc, str: file path to save output

    returns: None
    """
    assert len(doc.passages) == 2 # Want to make sure none break the rule
    txt_to_save = ''
    for passage in doc.passages: # Doesn't enforce title first
        text = passage.text
        txt_to_save += text + ' '
    out_path = f'{txt_out_loc}/{doc_key}.txt'
    with open(out_path, 'w') as myf:
        myf.write(txt_to_save)


def main(bioc_dset, overall_out_loc, txt_out_loc, out_prefix):

    # Read in dataset
    verboseprint('\nReading in BioC dataset...')
    with open(bioc_dset, 'r') as fp:
        collection = biocxml.load(fp)

    # Make new dataset
    verboseprint('\nConverting to unbound triples...')
    doc_trips = {}
    for doc in tqdm(collection.documents):
        # Get unique doc id
        doc_key = doc.id

        # Get and save the abstract text
        text = get_abstract(doc, doc_key, txt_out_loc)

        # Pull triples from abstract
        trips = get_trips(doc)

        # Add to dataset
        doc_trips[doc_key] = trips

    # Save
    verboseprint('\nSaving...')
    out_path = f'{overall_out_loc}/{out_prefix}_unbound_triples.json'
    with open(out_path, 'w') as myf:
        json.dump(doc_trips, myf)
    verboseprint(f'Saved triples to {out_path}')
    verboseprint(f'Abstract txt files were saved to {txt_out_loc}')

    verboseprint('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get non-text bound triples')

    parser.add_argument('bioc_dset', type=str,
            help='Path to BioC formatted CDR dataset')
    parser.add_argument('overall_out_loc', type=str,
            help='Path to save triples')
    parser.add_argument('txt_out_loc', type=str,
            help='Specific directory for saving txt files of abstracts. Will '
            'be created if it doesn\' exist')
    parser.add_argument('out_prefix', type=str,
            help='Prefix to prepend to output triple file name')
    parser.add_argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')

    args = parser.parse_args()

    args.bioc_dset = abspath(args.bioc_dset)
    args.overall_out_loc = abspath(args.overall_out_loc)
    args.txt_out_loc = abspath(args.txt_out_loc)

    if not exists(args.txt_out_loc):
        mkdir(args.txt_out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None

    main(args.bioc_dset, args.overall_out_loc, args.txt_out_loc, args.out_prefix)
