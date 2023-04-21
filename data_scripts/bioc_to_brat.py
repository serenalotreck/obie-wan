"""
Script to convert BioC data from the CDR corpus to brat standoff format.

This should be possible within the bioc package (brat2bioc module), but it
doesn't appear that the package allows bioc --> brat, only brat --> bioc.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath
from tqdm import tqdm
from bioc import biocxml
from intervaltree import Interval, IntervalTree


def convert_entity_anns(passage, title_len, text, start_id=0):
    """
    Take a BioC passage and creates and entity annotation for each instance.
    Adds the title_len offset to account for merging the title and abstract in
    the final document.

    parameters:
        passage, BioCPassage instance: the passage to convert
        title_len, int: additional offset from merge
        text, str: full document text (all passages combined)
        start_id, int: number to start at for T# indices

    returns:
        ents, list of BratEntity instances: entities
    """
    ents = []
    current_id = start_id
    for ent in passage.annotations:
        # Need to iterate through in case it's in more than one location
        for loc in ent.locations:
            # Get offsets from BioC
            start = loc.offset
            end = start + loc.length

            # Convert with title_len
            start += title_len
            end += title_len

            # Make entity and add to list
            entity = BratEntity()
            entity.id = f'T{current_id}'
            entity.text = text
            entity.add_span(start, end)
            ents.append(entity)

            # Augment id counter
            current_id += 1

    return ents


def bioc_to_brat(doc, title_len, text, doc_key, out_loc):
    """
    Turns BioC format to brat standoff format.

    parameters:
        doc, BioCDocument instance: document to save
        title_len, int: number of characters prepended to the original abstract
        text, str: fulll text of title and abstract combined
        doc_key, int: unique document ID to use as file name
        out_loc, str: file path to save output

    returns: None
    """
    assert len(doc.passages) == 2 # Want to make sure none break the rule
    ent_anns = []
    key_order = [p.infons['type'] for p in doc.passages]
    assert key_order == ['title', 'abstract']
    for passage in docs.passages:
        if passage.infons['type'] == 'title':
            # Make entity annotations for title text
            ent_anns = convert_entity_anns(passage, 0, 0)
        else:
            # Make entity annotations for abstract text
            prev_anns = len(ent_anns)
            ent_anns = convert_entity_anns(passage, title_len,
                    start_id=prev_anns)
        # Add to list
        ent_anns += ent_anns
    # Locate relations
    rels = convert_rel_anns(passage, ent_anns)
    # Combine into one list
    all_anns = ent_anns + rel_anns
    # Make brat document
    brat_doc = BratDocument(doc_key, full_text, all_anns)
    out_prefix = f'{out_loc}/{doc_key}'
    with open(f'{out_prefix}.ann', 'w') as ann_fp, open(
            f'{out_prefix_.txt', 'w') as text_fp:
        brat.dump(brat_doc, text_fp, ann_fp)


def get_abstract(doc, doc_key, txt_out_loc):
    """
    Extracts and saves the abstract of a BioCDocument as a txt file.

    parameters:
        doc, BioCDocument instance: document to save
        doc_key, int: unique document ID to use as file name
        txt_out_loc, str: file path to save output

    returns:
        title_len, int: number of characters in title, including additional
            punctuation and space
        text, str: full text of combined title and abstract
    """
    assert len(doc.passages) == 2 # Want to make sure none break the rule
    txt_to_save = ''
    title_len = 0
    key_order = [p.infons['type'] for p in doc.passages]
    assert key_order == ['title', 'abstract'] # Make sure title is first
    for passage in doc.passages:
        text = passage.text
        if passage.infons['type'] == 'title':
            if text[-1] not in ['.', '?']:
                txt_to_save += text + '. '
            else:
                txt_to_save += text + ' '
            title_len += len(txt_to_save)
        else:
            txt_to_save += text

    # Save document
    out_path = f'{txt_out_loc}/{doc_key}.txt'
    with open(out_path, 'w') as myf:
        myf.write(txt_to_save)

    return title_len, text


def main(bioc_dset, out_loc, out_prefix):

    # Read in dataset
    verboseprint('\nReading in BioC dataset...')
    with open(bioc_dset, 'r') as fp:
        collection = biocxml.load(fp)

    # Make new dataset
    verboseprint('\nConverting to brat standoff...')
    for doc in tqdm(collection.documents):
        # Get unique doc id
        doc_key = doc.id

        # Get and save the abstract text
        title_len, text = get_abstract(doc, doc_key, out_loc)

        # Convert annotations to brat
        bioc_to_brat(doc, title_len, text, doc_key, out_loc)

    verboseprint(f'Brat files were saved to {out_loc}')

    verboseprint('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get non-text bound triples')

    parser.add_argument('bioc_dset', type=str,
            help='Path to BioC formatted CDR dataset')
    parser.add_argument('out_loc', type=str,
            help='Specific directory for saving txt and ann files of '
            'abstracts. Will be created if it doesn\'t exist')
    parser.add_argument('--verbose', '-v', action='store_true',
            help='Whether or not to print updates')

    args = parser.parse_args()

    args.bioc_dset = abspath(args.bioc_dset)
    args.out_loc = abspath(args.out_loc)

    if not exists(args.out_loc):
        mkdir(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None

    main(args.bioc_dset, args.out_loc, args.out_prefix)
