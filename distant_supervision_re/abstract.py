"""
Defines the Abstract class.

Author: Serena G. Lotreck
"""
from collections import OrderedDict
import benepar, spacy
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
            all_tokens.extend(sent)
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
        for sent in doc.sents:
            self.const_parse.append(sent._.parse_string)
        self.spacy_doc = doc

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
            # There are sub-nodes with S labels, or ## TODO deal with this
            # there are sibling SBAR annotations, or ## TODO deal with this
            # it's not a complete sentence
            except AttributeError or AssertionError:
                self.skipped_sents['parse'].append(self.const_parse[sent])
                if self.const_parse[sent].count('S') > 2:
                    self.skipped_sents['phrase'].append('NO PHRASE: Multiple '
                    'nested sentence annotations')
                elif self.const_parse[sent].count('SBAR') > 2:
                    self.skipped_sents['phrase'].append('NO PHRASE: Sibling '
                            'SBAR annotations')
                else:
                    self.skipped_sents['phrase'].append('NO PHRASE: Incomplete '
                    'sentence')
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
                    self.success_sents['phrase'].append(phrase)
                # If there are gaps in the phrase, the tokenization won't align
                except TypeError:
                    self.skipped_sents['parse'].append(self.const_parse[sent])
                    self.skipped_sents['phrase'].append(phrase)
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
            check_to_walk.extend(pu.parse_sbar(sent))
        # Check for directly nested sentence annotations
        elif sent._.parse_string.count('S') >=2:
            check_to_walk.extend(pu.parse_mult_S(sent))
        # Another possibility is that we have multiple VP connected by a CC
        ## TODO deal with it
        # If it's not a special case, just add to the list
        else:
            check_to_walk.append(sent)
        # Now, go through these to get what we should walk
        to_walk = []
        for cand in check_to_walk:
            # We want to find the VP on the level right below the top S
            for child, label in zip(sent._.children, first_level_labels):
                if (len(label) == 1) and (label[0] == 'VP'):
                    to_walk.append(child)

        # Then, we walk to build the phrases
        phrases = []
        for walk in to_walk:
            # Get phrase Span components
            phrase = pu.walk_VP('', walk)
            # Reconstruct one Span for the phrase
            phrase_span = phrase[0].doc[phrase[0].start:phrase[-1].end]
            # Assert that the span is continuous
            phrase_idxs = [p.start for p in phrase]
            assert len(phrase_idxs) == phrase_idxs[-1]-phrase_idxs[0]+1
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
            for p_l_pair in phrase_labels[i]:
                phrase = p_l_pair[0]
                label = p_l_pair[1]
                start_ent, end_ent = self.choose_ents(phrase, i)
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
            phrase, str: relation phrase
            sent_idx, int: sentence index

        returns:
            entities, list of list: two DyGIE++ formatted entities
        """
        # Get the sentence tokenization
        sent = self.sentences[sent_idx]
        # Get the sentence's entities
        sent_ents = self.entities[sent_idx]
        # Tokenize the phrase and locate in the sentence
        phrase_doc = self.nlp(phrase)
        phrase_toks = [tok.text for tok in phrase_doc]
        start_idx, end_idx = be.find_sublist(phrase_toks, self.doc_toks)
        # Figure out what entities are closest on either side
        subj_cands = [e[1] for e in sent_ents]
        # Gets the closest ent ending on either side of the relation start
        subj_idx = (np.abs(np.asarray(subj_cands) - start_idx)).argmin()
        # So we want to make sure it's before the relation start
        if subj_idx > start_idx:
            subj_idx -= 1
            # Need to make sure this index exists
            try:
                subj = sent_ents[subj_idx]
            except IndexError:
                return [] # We'll drop this triple if we can't find the right
                          # entities
        else:
            subj = sent_ents[subj_idx]
        # Get the closest ent starting on either side of the relation end
        obj_cands = [e[0] for e in sent_ents]
        obj_idx = (np.abs(np.asarray(obj_cands) - end_idx)).argmin()
        # Make sure it's after the ending
        if obj_idx < end_idx:
            obj_idx += 1
            try:
                obj = sent_ents[obj_idx]
            except IndexError:
                return []
        else:
            obj = sent_ents[obj_idx]

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
