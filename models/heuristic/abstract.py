"""
Defines the Abstract class.

Author: Serena G. Lotreck
"""
from collections import OrderedDict
import benepar, spacy
from spacy.tokens import Doc
import bert_embeddings as be
import numpy as np
from collections import defaultdict
import phrase_utils as pu


class Abstract():
    """
    Contains the text and annotations for one abstract.
    """
    def __init__(self, dygiepp=None, text='', sentences=None,
            entities=None, cand_sents=None, const_parse=None,
            nlp=None, spacy_doc=None, relations=None):

        self.dygiepp = {} if dygiepp is None else dygiepp
        self.text = text
        self.sentences = [] if sentences is None else sentences
        self.doc_tok = [tok for sent in self.sentences for tok in sent]
        self.entities = [] if entities is None else entities
        self.cand_sents = [] if cand_sents is None else cand_sents
        self.const_parse = [] if const_parse is None else const_parse
        self.nlp = nlp
        self.spacy_doc = spacy_doc
        self.relations = [] if relations is None else relations
        self.skipped_sents = {'parse':[], 'phrase':[]}
        self.success_sents = {'parse':[], 'phrase':[]}

    def __eq__(self, other):
        """
        Allows instances to be compared by the == operator, for
        enabling tests.
        """
        assert isinstance(other, Abstract)

        checks = [self.dygiepp == other.dygiepp,
                self.text == other.text,
                self.sentences == other.sentences,
                self.doc_tok == other.doc_tok,
                self.entities == other.entities,
                self.cand_sents == other.cand_sents,
                self.const_parse == other.const_parse,
                #self.spacy_doc == other.spacy_doc, # This doesn't work,
                                # as far as I can tell they haven't overwritten
                                # the __eq__ method, which seems weird to me
                                ## TODO write something to compare two docs
                #self.nlp = other.nlp # This also may not work, haven't
                                      # confirmed
                self.relations == other.relations,
                self.skipped_sents == other.skipped_sents,
                self.success_sents == other.success_sents]
        match = set(checks)
        if (len(match) == 1) and list(match)[0]:
            return True
        else:
            names = ['dygiepp', 'text', 'sentences', 'doc_tok', 'entities',
                    'cand_sents','const_parse', 'nlp', 'spacy_doc',
                    'relations', 'skipped_sents', 'success_sents']
            print('The attributes in disagreement are:')
            print({attr:boolval for attr,boolval in zip(names, checks) if not
                boolval})
            return False

    @classmethod
    def parse_pred_dict(cls, pred_dict, nlp):
        """
        Pulls input from a dictionary in the DyGIE++ prediction format
        (see https://github.com/dwadden/dygiepp/blob/master/doc/data.md)
        and calls constructor to build instance.

        parameters:
            pred_dict, dict: dygiepp formatted dictionary with entity
                predictions
            nlp, spacy model: nlp model with constituency parser loaded

        returns:
            abst, Abstract instance: instance of abstract class
        """
        # Get information to initialize an Abstract instance
        all_tokens = []
        for sent in pred_dict["sentences"]:
            # Replace -LRB- -RRB- with ( and )
            updated_sent = []
            for tok in sent:
                if tok == '-LRB-':
                    updated_sent.append('(')
                elif tok == '-RRB-':
                    updated_sent.append(')')
                else:
                    updated_sent.append(tok)
            all_tokens.extend(updated_sent)
        text = ' '.join(all_tokens)
        sentences = pred_dict["sentences"]

        # Allow documents with both predicted or gold standard NER
        # Use predictions by default, unless there are none
        try:
            entities = pred_dict["predicted_ner"]
        except KeyError:
            entities = pred_dict["ner"]

        # Initialize Abstract instance
        abst = cls(pred_dict, text, sentences, entities)

        # Perform the higher functions of the class
        abst.set_cand_sents()
        abst.set_const_parse_and_spacy_doc(nlp)

        return abst

    def set_cand_sents(self):
        """
        Identifies sentences with two or more entities and creates a list
        of their indices in the sentence list attribute.
        """
        for i in range(len(self.entities)):
            if len(self.entities[i]) >= 2:
                self.cand_sents.append(i)

    def set_const_parse_and_spacy_doc(self, nlp):
        """
        Performs constituency parse using provided model,
        and assigns the parse strings and spacy doc containing parse
        information to attributes.
        """
        self.nlp = nlp
        doc = nlp(self.text)
        # Check for mismatching tokenizers between jsonl and spacy
        if len(list(doc.sents)) == len(self.sentences):
            for sent in doc.sents:
                self.const_parse.append(sent._.parse_string)
            self.spacy_doc = doc
        else:
            print(f'\nDoc {self.dygiepp["doc_key"]}\'s original tokenization '
                    'does not match that generated by en_core_sci_sm.')
            docs_to_join = []
            for sent in self.sentences:
                sent_doc = nlp(' '.join(sent))
                assert len(list(sent_doc.sents)) == 1
                self.const_parse.append(list(sent_doc.sents)[0]._.parse_string)
                docs_to_join.append(sent_doc)
            new_doc = Doc.from_docs(docs_to_join)
            self.spacy_doc = new_doc

    def extract_rels(self, tokenizer, model, label_df):
        """
        For each candidate sentence, calls a helper function to get the
        candidate relation phrase from the constituency parse, and uses
        the sentence to generate a contextualized embedding for that phrase.
        Using context-averaged representations of the four relation types,
        computes distance between phrase embedding and label embeddings,
        assigning a label based on which label embedding is closer for each
        of the candidate relation phrases. Distance must be less than 0.5 to
        be assigned a label, otherwise no relation is recorded. Sets an
        attribute of DyGIE++-formatted list for relation annotations.

        parameters:
            tokenizer, huggingface Bert tokenizer: the tokenizer to use
                with the model
            model, huggingface Bert Model: the model to use to
                generate embeddings
            label_df, df: index is strings representing the
                relation labels that we want to use, rows are their
                context-averaged embeddings. These should be generated
                elsewhere in order to have consistent embeddings for these
                labels across all abstracts on which this method is used.

        returns:
            skipped, int: number of sentences whose candidate phrases
                couldn't be resolved to their tokenizations
            total, int: the total number of candidate sentences
        """
        phrase_labels = {k:[] for k in self.cand_sents}
        for sent in self.cand_sents:

            # Use helper to get the phrase to embed
            try:
                phrases = self.pick_phrase(sent)
            # Happens when:
            # it's not a complete sentence, or
            # the identified phrase is noncontinuous
            except AssertionError as e:
                self.skipped_sents['parse'].append(self.const_parse[sent])
                if e == 'Noncontinuous span':
                    self.skipped_sents['phrase'].append(f'NO PHRASE: {e}')
                elif e == 'NO PHRASE: Multiple levels with kids':
                    self.skipped_sents['phrase'].append(e)
                elif e == 'NO PHRASE: VP-CC-VP detector issue':
                    self.skipped_sents['phrase'].append(e)
                elif e == 'NO PHRASE: Double label failure':
                    self.skipped_sents['phrase'].append(e)
                else:
                    self.skipped_sents['phrase'].append('NO PHRASE: other '
                            'Assertion error')
                continue

            # Get this embedding out of the BERT output and add to dict
            sent_text = ' '.join(self.sentences[sent])
            for phrase in phrases:
                try:
                    embedding = be.get_phrase_embedding(sent_text, phrase.text,
                        tokenizer, model)
                    # Get distances to choose label
                    label = pu.compute_label(label_df, embedding)
                    if label != '':
                        phrase_labels[sent].append((phrase, label))
                    self.success_sents['parse'].append(self.const_parse[sent])
                    self.success_sents['phrase'].append(phrase.text)
                # Not sure why this TypeError occurs now; it used to be that if
                # there were gaps in the phrase, the tokenization wouldn't
                # align, but I take care of that when subsetting the doc to get
                # the phrase with correct indices now
                ## TODO look into it
                except TypeError:
                    self.skipped_sents['parse'].append(self.const_parse[sent])
                    self.skipped_sents['phrase'].append(phrase.text)
        # Format the relations in DyGIE++ format
        relations = self.format_rels(phrase_labels)
        self.set_relations(relations)

        # Get the total number of candidate sentences and the number that were
        # skipped
        skipped = len(self.skipped_sents['parse'])
        total = len(self.cand_sents)

        return skipped, total

    def pick_phrase(self, sent_idx):
        """
        Uses constituency parse to choose phrases to embed for a given sentence.

        parameters:
            sent_idx, int: sentence from which to choose a phrase
            include_preps, bool: whether or not to include prepositions
                that are part of the final noun phrase in a sentence
                ## TODO implement this

        returns:
            phrases, list of spacy Span objects: the phrases to embed for this sentence
        """
        # Get the spacy object version of the sentence
        sent = list(self.spacy_doc.sents)[sent_idx]

        # Walk the tree and its labels
        first_level_labels = [c._.labels for c in sent._.children]
        # Check for special cases
        check_to_walk = []
        # Check for SBAR clauses
        if 'SBAR' in sent._.parse_string:
            check_to_walk.extend(pu.subset_tree(sent, 'SBAR', highest=False))
        # Check for directly nested sentence annotations
        elif sent._.parse_string.count('(S ') >=2:
            check_to_walk.extend(pu.subset_tree(sent, 'S', highest=True,
                ignore_root=True))
        # Another possibility is that we have multiple VP connected by a CC
        elif (sent._.parse_string.count('(VP ') >=2) and (
                sent._.parse_string.count('(CC ') >= 1):
            check_to_walk.extend(pu.subset_tree(sent, 'VP', highest=False))
        # If it's not a special case, just add to the list
        else:
            check_to_walk.append(sent)
        # Now, go through these to get what we should walk
        to_walk = []
        for cand in check_to_walk:
            # We want to find the next VP
            w = pu.subset_tree(cand, 'VP', highest=True)
            to_walk.extend(w)

        # Then, we walk to build the phrases
        phrases = []
        for walk in to_walk:
            # Get phrase Span components
            phrase = pu.walk_VP([], walk)
            assert phrase != [], ('NO PHRASE: VP-CC-VP detector issue')
            assert phrase != 'NO PHRASE: Multiple levels with kids', (
                                'NO PHRASE: Multiple levels with kids')
            # Reconstruct one Span for the phrase
            phrase_span = phrase[0].doc[phrase[0].start:phrase[-1].end]
            # Assert that the span is continuous
            phrase_idxs = [p.start for p in phrase]
            assert len(phrase_idxs) == phrase_idxs[-1]-phrase_idxs[0]+1, (
                    'Noncontinuous span')
            # If they are, add to the phrases list
            phrases.append(phrase_span)

        return phrases

    def format_rels(self, phrase_labels):
        """
        Formats relations in DyGIE++ format.

        parameters:
            phrase_labels, dict of list: keys are sentence indices,
                values are lists of tuples of the form (phrase, label)

        returns:
            relations, list of list: DyGIE++ formatted relations
        """
        relations = []
        for i, sent in enumerate(self.sentences):
            sent_rels = []
            if i in phrase_labels.keys():
                for p_l_pair in phrase_labels[i]:
                    phrase = p_l_pair[0]
                    label = p_l_pair[1]
                    try:
                        start_ent, end_ent = self.choose_ents(phrase, i)
                    except ValueError:
                        continue
                    rel = [start_ent[0], start_ent[1], end_ent[0], end_ent[1],
                            label]
                    sent_rels.append(rel)
            relations.append(sent_rels)

        return relations

    def choose_ents(self, phrase, sent_idx):
        """
        Choose an entity for a relation. Meant to ensure that the first
        entity in the relation is before the relation's VP, and the second
        is after.

        parameters:
            phrase, spacy Span object: relation phrase
            sent_idx, int: sentence index

        returns:
            entities, list of list: two DyGIE++ formatted entities
        """
        # Get the sentence's entities
        sent_ents = self.entities[sent_idx]
        # Locate phrase in the document
        start_idx, end_idx = phrase.start, phrase.end - 1 # So that its the
                                                          # last token
        # Figure out what entities are closest on either side
        subj_cands = [e[1] for e in sent_ents]
        # Gets the closest ent ending on either side of the relation start
        subj_cand_idx = (np.abs(np.asarray(subj_cands) - start_idx)).argmin()
        subj_end_idx = subj_cands[subj_cand_idx]
        # So we want to make sure it's before the relation start
        if subj_end_idx >= start_idx:
            move = (subj_end_idx - start_idx) + 1
            subj_cand_idx -= move
            if subj_cand_idx < 0:
                # - numbers are valid indices so it will return the wrong entity if
                # allowed to pass through
                return []
            else:
                subj = sent_ents[subj_cand_idx]
        else:
            subj = sent_ents[subj_cand_idx]
        # Get the closest ent starting on either side of the relation end
        obj_cands = [e[0] for e in sent_ents]
        obj_cand_idx = (np.abs(np.asarray(obj_cands) - end_idx)).argmin()
        obj_start_idx = obj_cands[obj_cand_idx]
        # Make sure it's after the ending
        if obj_start_idx <= end_idx:
            move = (end_idx - obj_start_idx) + 1
            obj_cand_idx += move
            try:
                obj = sent_ents[obj_cand_idx]
            except IndexError:
                return []
        else:
            obj = sent_ents[obj_cand_idx]

        return [subj, obj]

    def set_relations(self, relations):
        """
        Set the output of the relation extraction process as an attribute.
        """
        self.relations = relations

    def rels_to_dygiepp(self):
        """
        Adds relation predictions to DyGIE++ formatted version of the
        abstract. Returns the new dict in addition to updating the
        attribute.
        """
        self.dygiepp["predicted_relations"] = self.relations

        return self.dygiepp
