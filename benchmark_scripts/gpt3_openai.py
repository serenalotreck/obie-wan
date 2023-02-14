"""
Uses OpenAI's API to generate triples for abstracts.

Author: Serena G. Lotreck
"""
import argparse
from os import getenv, listdir
from os.path import abspath, isfile
from ast import literal_eval
import spacy
import openai
from tqdm import tqdm
import jsonlines
import sys
sys.path.append('../distant_supervision_re/')
import bert_embeddings as be
from abstract import Abstract


def embed_relation(sent_tok, rel, label_df, model, tokenizer):
    """
    Performs the same relation embedding as relation_embedding.py to generate a
    standardized relation label for a given triple.

    parameters:
        sent_tok, list of str: tokenized sentence
        rel, str: relation to embed
        label_df, pandas df: rows are labels, columns are vector elements
        model, huggingface BERT model: model to use
        tokenizer, huggingface BERT tokenizer: tokenizer to use
    """
    sent_text = ' '.join(self.sentences[sent_tok])
    embedding = be.get_phrase_embedding(sent_text, rel,
        tokenizer, model)
    label = Abstract.compute_label(label_df, embedding)
    if label == '':
        return None
    else:
        return label


def find_sent(dygiepp_doc, triple_tok):
    """
    Get the index of the sentence that contains a given triple

    parameters:
        dygiepp_doc, dict: dygiepp-formatted dict with sentences filled in
        triple_tok, list of list of str: tokenized triple to locate

    returns:
        sent_idx, int: index of the sentence that the triple is in
    """
    sent_cands = []
    for sent_idx, sent in enumerate(dygiepp_doc['sentences']):
        sublist_idxs = []
        for elt in triple_tok:
            idxs = be.find_sub_list(elt, sent)
            sublist_idxs.append(idxs)
        all_present = [True if sli != (None, None) else False
                for sli in sublist_idxs]
        if (len(set(all_present)) == 1) & (list(set(all_present))[0]):
            sent_cands.append(sent_idx)
    print(f'sent_cands: {sent_cands}')
    assert len(sent_cands) == 1 # Need to know if this method doens't uniquely
                                # identify the sentence from whence it came
    sent_idx = sent_cands[0]
    return sent_idx


def format_dygiepp_doc(doc_key, abstract_txt, triples, nlp, embed_rels=False,
        label_df=None, model=None, tokenizer=None):
    """
    Uses scispacy to tokenize the original text and align the triples with
    the tokenization to produce the dygiepp-formatted version of the
    predictions.

    parameters:
        doc_key, str: document identifier
        abstract_txt, str: original abstract text
        triples, list of tuple: tuples to map
        nlp, spacy NLP object: tokenizer to use for dygiepp formatting
        embed_rels, bool: true if prediciton labels should be standardized by
            embedding
        label_df, pandas df or None: if embed_labels, rows are labels, columns
            are elements of the emebedding for the labels
        model, huggingface BERT model or None: model to use for relation
            embedding if embed_rels
        tokenizer, huggingface BERT tokenizer or None: tokenier to use for
            relation embedding if embed_rels

    returns:
        dygiepp_doc, dict: dygiepp-formatted predictions
    """
    # Initialize the dygiepp-formatted doc
    dygiepp_doc = {'doc_key': doc_key,
            'dataset': 'scierc', # For consistency with data format
            'sentences': [],
            'ner': [],
            'relations': []}

    # Tokenize the abstract text
    doc = nlp(abstract_txt[0])
    dygiepp_doc['sentences'] = [[tok.text for tok in sent] for sent in doc.sents]
    dygiepp_doc['ner'] = [[] for i in range(len(dygiepp_doc['sentences']))]
    dygiepp_doc['relations'] = [[] for i in range(len(dygiepp_doc['sentences']))]
    for triple in triples:
        # Tokenize the triple's components
        triple_tok = [nlp(part) for part in triple]
        triple_tok = [[tok.text for tok in part_doc] for part_doc in triple_tok]
        # Align the entities with the tokenization
        # First, figure out what sentence we're in
        sent_idx = find_sent(dygiepp_doc, triple_tok)
        # Then use document-wide tokenization indices to format the relation
        # Get the number of tokens up to the sentence of interest
        prev_tok_num = sum([len(dygiepp_doc['sentences'][i]) for i in
            range(sent_idx)])
        # For both entities, get the document wide indices
        entities = [] # Keep here to use for relations
        for ent in [triple_tok[0], triple_tok[2]]:
            sent_text = dygiepp_doc["sentences"][sent_idx]
            # Check to see if we already have an entity that starts with the
            # same text in this sentence
            iters_prev = 0
            for p_ent in dygiepp_doc["ner"][sent_idx]:
                p_toks = sent_text[p_ent[0] - prev_tok_num]
                if p_toks == ent[0]:
                    iters_prev += 1
            # Look for the instance of the first entity that corresponds with
            # the number of the same start tokens we've previously seen
            start_idx = [i for i, n in enumerate(sent_text)
                    if n == ent[0]][iters_prev] + prev_tok_num
            # Then starting at the start token, look for the first instance of
            # the end token
            end_idx = [i for i, n in enumerate(sent_text)
                    if n == ent[-1] and i + prev_tok_num >= start_idx
                    ][0] + prev_tok_num
            entities.append([start_idx, end_idx, 'ENTITY'])
        # Place entities in the correct sentence
        dygiepp_doc['ner'][sent_idx].extend(entities)
        # Get the relation label
        if embed_rels:
            label = embed_relation(dygiepp_doc['ner'][sent_idx], triple_tok[1],
                    label_df)
            if label is None:
                continue # Skip this relation if it doesn't get a label
        else:
            label = triple[1]
        # Format the relation using the entities
        relation = [entities[0][0], entities[0][1], entities[1][0],
                entities[1][1], label]
        # Place relation in correct sentence
        dygiepp_doc['relations'][sent_idx].append(relation)

    return dygiepp_doc


def process_preds(abstracts, raw_preds, bert_name, label_path, embed_rels=False):
    """
    Pulls text output from predictions, and calls helpers to format the
    predictions for dygiepp.

    parameters:
        abstracts, dict: keys are filenames, values are strings with abstracts
        raw_preds, dict: keys are filenames, values are raw output of gpt3
        bert_name, str: name of BERT model to use to embed relations
        label_path, str: path to a file with relation labels and example
            sentences containing them
        embed_rels, bool: True if relations should be converted to standard
            labels by embedding

    returns:
        formatted_preds, list of dict: dygiepp formatted data
    """
    if embed_rels:
        # Load the BERT model, and embed relation labels
        verboseprint('\nLoading BERT model...')
        verboseprint(f'Model name: {bert_name}')
        tokenizer, model = be.load_model(pretrained=bert_name)

        # Embed relation labels
        verboseprint('\nEmbedding relation labels...')
        with open(label_path) as infile:
            label_dict = json.load(infile)
        label_embed_dict = be.embed_labels(label_dict, tokenizer, model)
        label_df = pd.DataFrame.from_dict(label_embed_dict, orient='index')
    else:
        tokenizer = None
        model = None
        label_df = None

    # Load the scispacy model to use for tokenization
    nlp = spacy.load("en_core_sci_sm")

    # Format preds
    formatted_preds = []
    for fname, abstract_pred in tqdm(raw_preds.items()):
        # Check that there's only one prediction
        assert len(abstract_pred['choices']) == 1
        # Pull out the text from the prediction
        pred_text = abstract_pred['choices'][0]['text']
        # Read the output literally to get triples
        print(f'abstract_pred:\n{abstract_pred}')
        print(f'pred_text:\n{pred_text}')
        doc_triples = [literal_eval(t) for t in pred_text.split('\n')
                if t != '']
        # Pass triples and abstract text to helper to get tokenizations
        doc_dygiepp = format_dygiepp_doc(fname, abstracts[fname], doc_triples,
                nlp, embed_rels, label_df, tokenizer, model)
        # Add to formatted docs
        formatted_preds.append(doc_dygiepp)

    return formatted_preds


def gpt3_predict(abstracts, prompt, fmt=False, model='text-davinci-003',
        max_tokens=2048):
    """
    Passes each abstract to gpt3 and returns the raw output.

    parameters:
       abstracts, dict: keys are file names and values are lines from the
            abstracts
        prompt, str: string to pass as the prompt. If it includes the substring
            "{abstract_txt}" in order to be able to use with each abstract
            separately, must provide fmt=True
        fmt, bool: whether or not the prompt string needs to be formatted
        model, str: name of the model to use, default is the full GPT3 model
        max_tokens, int: maximum number of tokens to request from model. Will
            change based on what model is requested, and must be larger than
            (number of tokens in request + number of tokens in respose)

    returns:
        raw_preds, dict: keys are file names and values are the output of gpt3
            predictions
    """
    raw_preds = {}
    for fname, abstract_txt in tqdm(abstracts.items()):
        if fmt:
            prompt = prompt.format(abstract_txt=abstract_txt)
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0
                )
        raw_preds[fname] = response
        print(f'\nResponse and prediction for {fname}:')
        print('------------------------------------------')
        print(response)
        print('------------------------------------------')
        print(response['choices'][0]['text'])
    return raw_preds


def main(text_dir, embed_rels, label_path, bert_name, out_loc, out_prefix):

    # Read in the abstracts
    verboseprint('\nReading in abstracts...')
    abstracts = {}
    for f in tqdm(listdir(text_dir)):
        if isfile(f'{text_dir}/{f}'):
            path = f'{text_dir}/{f}'
            with open(path) as myf:
                abstract =[l.strip() for l in  myf.readlines()]
                abstracts[f] = abstract

    # Prompt GPT3 for each abstract
    verboseprint('\nGenerating predictions...')
    prompt = ('Extract the biological relationships from '
    'the following text as (Subject, Predicate, Object) triples, and '
    'include the index of the sentence from which the triple is extracted: '
    '{abstract_txt}.\nExample relationship triple:\n("Protein 1", '
    '"regulates", "Protein 2"), "Sentence 0"')
    raw_preds = gpt3_predict(abstracts, prompt)

    # Format output in dygiepp format
    verboseprint('\nFormatting predictions...')
    #final_preds = process_preds(abstracts, raw_preds, bert_name, label_path,
    #        embed_rels)

    # Save output
    verboseprint('\nSaving output...')
    out_path = f'{out_loc}/{out_prefix}_gpt3_preds.jsonl'
    with jsonlines.open(out_path, 'w') as writer:
        writer.write_all(final_preds)
    verboseprint(f'Predictions saved to {out_path}')

    print('\nDone!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get GPT3 triples')

    parser.add_argument('text_dir', type=str,
            help='Path to directory where abstracts to be used are stored')
    parser.add_argument('--embed_rels', action='store_true',
            help='Whether or not to embed relations to get standardized '
            'labels. If passed, must also pass label_path')
    parser.add_argument('-label_path', type=str, default='',
            help='Path to a json file where keys are the relation '
            'labels, and values are lists of strings, where each string '
            'is an example sentence that literally uses the relation label.')
    parser.add_argument('-bert_name', type=str,
            default="alvaroalon2/biobert_genetic_ner",
            help='Name of pretrained BERT model from the huggingface '
            'transformers library. Default is the BioBERt genetic NER '
            'model.')
    parser.add_argument('-out_loc', type=str,
            help='Path to save output')
    parser.add_argument('-out_prefix', type=str,
            help='String to prepend to save file names')
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Whether or not to print output.')

    args = parser.parse_args()

    if args.embed_rels:
        assert label_path != '', ('--embed_rels was specified, -label_path '
                'must be provided.')
        args.label_path = abspath(args.label_path)
    args.text_dir = abspath(args.text_dir)
    args.out_loc = abspath(args.out_loc)

    verboseprint = print if args.verbose else lambda *a, **k: None

    openai.api_key = getenv("OPENAI_API_KEY")

    main(args.text_dir, args.embed_rels, args.label_path, args.bert_name,
            args.out_loc, args.out_prefix)
