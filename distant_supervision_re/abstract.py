"""
Defines the Abstract class.

Author: Serena G. Lotreck
"""
from collections import OrderedDict
import benepar, spacy
from nltk import ParentedTree
import bert_embeddings as be
import numpy as np
from numpy.linalg import norm
from collections import defaultdict

import traceback

class Abstract():
    """
    Contains the text and annotations for one abstract.
    """
    def __init__(self, dygiepp=None, text='', sentences=None,
            entities=None, cand_sents=None, const_parse=None,
            spacy_doc=None, relations=None):

        self.dygiepp = {} if dygiepp is None else dygiepp
        self.text = text
        self.sentences = [] if sentences is None else sentences
        self.entities = [] if entities is None else entities
        self.cand_sents = [] if cand_sents is None else cand_sents
        self.const_parse = [] if const_parse is None else const_parse
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
                self.entities == other.entities,
                self.cand_sents == other.cand_sents,
                self.const_parse == other.const_parse,
                #self.spacy_doc == other.spacy_doc, # This doesn't work,
                                # as far as I can tell they haven't overwritten
                                # the __eq__ method, which seems weird to me
                                ## TODO write something to compare two docs
                self.relations == other.relations]
        match = set(checks)
        if (len(match) == 1) and list(match)[0]:
            return True
        else:
            names = ['dygiepp', 'text', 'sentences','entities',
                    'cand_sents','const_parse', 'spacy_doc',
                    'relations']
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
        phrase_labels = {k:{} for k in self.cand_sents}
        for sent in self.cand_sents:

            # Use helper to get the phrase to embed
            try:
                phrases = self.pick_phrase(sent)
            # Happens when:
            # There are sub-nodes with S labels, or ## TODO deal with this
            # It's not a complete sentence
            except AttributeError:
                self.skipped_sents['parse'].append(self.const_parse[sent])
                if self.const_parse[sent].count('S') > 2:
                    self.skipped_sents['phrase'].append('NO PHRASE: Multiple '
                    'nested sentence annotations')
                else:
                    self.skipped_sents['phrase'].append('NO PHRASE: Incomplete '
                    'sentence')
                continue

            # Get this embedding out of the BERT output and add to dict
            sent_text = ' '.join(self.sentences[sent])
            for phrase in phrases:
                try:
                    embedding = be.get_phrase_embedding(sent_text, phrase,
                        tokenizer, model)
                    # Get distances to choose label
                    label = self.compute_label(label_df, embedding)
                    if label != '':
                        phrase_labels[sent][phrase] = label
                    self.success_sents['parse'].append(self.const_parse[sent])
                    self.success_sents['phrase'].append(phrase)
                # If there are gaps in the phrase, the tokenization won't align
                except TypeError:
                    self.skipped_sents['parse'].append(self.const_parse[sent])
                    self.skipped_sents['phrase'].append(phrase)
        ## TODO add something to keep associating the plain textphrases with
        ## the embedding-derived labels to be able to determine where the phrase
        ## came in the sentence later on
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
            phrases, list of str: the phrases to embed for this sentence
        """
        # Get the spacy object version of the sentence
        sent = list(self.spacy_doc.sents)[sent_idx]

        # Walk the tree and its labels
        first_level_labels = [c._.labels for c in sent._.children]
        # Check for special cases
        check_to_walk = []
        # Check for SBAR clauses
        if 'SBAR' in sent._.parse_string:
            check_to_walk.extend(Abstract.parse_sbar(sent))
        # Check for directly nested sentence annotations
        elif sent._.parse_string.count('S') >=2:
            check_to_walk.extend(Abstract.parse_mult_S(sent))
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
            phrase = Abstract.walk_VP('', walk)
            phrases.append(phrase)

        return phrases

    @staticmethod
    def parse_sbar(sent):
        """
        Function to pull apart sentences containing SBAR annotations into the
        correct parts to pass to walk_VP.

        This function assumes that sentences with SBAR annotations fall into
        one of three categories:
            Class 1: The SBAR node only has a VP child, but no NP child
            Class 2: The SBAR node has both a NP and VP child
            Class 3: There are multiple other SBAR's nested within the
                top-level SBAR node. Child SBAR's can fall into Class 1 or
                Class 2

        TODO decide how to deal with  Class 1 (whether or not to annotate)

        parameters:
            sent, spacy Doc object sentence: sentence to parse

        returns:
            check_to_walk, list of spacy Span objects: phrases to walk,
                starting at the S node below the SBAR annotation(s)
        """
        pass

    @staticmethod
    def parse_mult_S(sent):
        """
        Function to pull apart sentences that have multiple nested S
        annotations.

        parameters:
            sent, spacy Doc object sentence: sentence to parse

        returns:
            check_to_walk, list of spacy Span objects: phrases to walk,
            starting at the nested S annotations
        """
        pass

    @staticmethod
    def walk_VP(phrase, next_child):
        """
        Walk recursively through a VP, adding anything that's not the
        terminal NP to the phrase string.

        parameters:
            phrase, str: phrase to add to
            next_child, spacy Span object: the next child to check

        returns:
            phrase, str: updated phrase
        """
        # Get the labels that are on the next level
        next_labels, child_tups = Abstract.get_child_tups(next_child)
        # Base case
        if 'NP' in next_labels:
            phrase_add = [t[1].text for t in child_tups
                    if (t[0] != 'NP') & (t[0] != 'PP')]
            phrase += ' ' + ' '.join(phrase_add)
            return phrase.strip() # Removes leading whitespace
        # Recursive case
        else:
            # Add anything that doesn't have a child
            # Leaf nodes have no labels in benepar
            phrase_add = [t[1].text for t in child_tups
                if t[0] == 'NO_LABEL']
            phrase += ' ' + ' '.join(phrase_add)
            # Continue down the one that does
            to_walk = [t[1] for t in child_tups if t[0] != 'NO_LABEL']
            if len(to_walk) == 1:
                to_walk = to_walk[0]
                return Abstract.walk_VP(phrase, to_walk)
            elif len(to_walk) == 0:
                return phrase.strip()
            else:
                ## TODO implement dropping leading PP's here
                return 'NO PHRASE: Multiple levels with kids'

    @staticmethod
    def subset_tree(next_child, label):
        """
        Return the benepar-parsed spacy object starting at the node with the
        label label. If this label occurs more than once in the tree, and the

        parameters:
            next_child, spacy Span object: the next child to check
            label, str: parse label to look for

        returns:
            subset_child, spacy Span object: subset child
        """
        # Base case
        if label in next_child._.labels:
            subset_child = next_child
            return subset_child
        # Recursive case
        else:
            # Get the labels that appear on the next level
            _, child_tups = Abstract.get_child_tups(next_child)
            # Check which ones have children
            have_kids = [tup[1] for tup in child_tups if tup[0] != 'NO_LABEL']
            # If multiples have children, only go down the one that contains
            # SBAR
            if len(have_kids) > 1:
                contains_lab = []
                for c in have_kids:
                    if label in c._.parse_string:
                        contains_lab.append(c)
                # The case where multiple occurrences of the target label are
                # nested and we want to stop at the one that's highest in the
                # tree
                if len(contains_lab) == 1:
                    to_walk = contains_lab[0]
                    return Abstract.subset_tree(to_walk, label)

                # The case where the same label are siblings
                ## TODO implement
                elif len(contains_lab) > 1:
                    mults = True
                    try:
                        assert mults, (f'The requested label is sibling with '
                                'itself')
                    except AssertionError as e:
                        print(e)
                        Abstract.visualize_parse(next_child._.parse_string)
            # If only one has children:
            else:
                to_walk = have_kids[0]
                return Abstract.subset_tree(to_walk, label)

    @staticmethod
    def get_child_tups(next_child):
        """
        Returns a list of spacy-object children and their corresponding labels.

        parameters:
            next_child, spacy Span object: span to get children from

        returns:
            next_labels, list of str: labels of the children
            child_tups, list of tuple: spacy-object children with their labels
        """
        next_labels = []
        for c in next_child._.children:
            # Terminal nodes
            if len(c._.labels) == 0:
                next_labels.append('NO_LABEL')
            # Non-terminal nodes with only one label
            elif len(c._.labels) == 1:
                next_labels.append(c._.labels[0])

            elif len(c._.labels) > 1:
                other_labs = list(c._.labels)
                other_labs.remove('S')
                assert len(other_labs) == 1
                next_labels.append(other_labs[0])

        kids = next_child._.children
        child_tups = [(lab, c) for lab, c in zip(next_labels, kids)]

        return next_labels, child_tups

    @staticmethod
    def compute_label(label_df, embedding):
        """
        Computes cosine distance between phrase embedding and all labels,
        choosing the label with the highest similarity as the label for this
        phrase. Must have a similarity of at least 0.5 to get a label.

        parameters:
            label_df, df: index is labels, rows are embeddings
            embedding, vector: embedding for the phrase

        returns:
            label, str: chosen label, empty string if no label chosen
        """
        label_mat = np.asarray(label_df)
        cosine = np.dot(label_mat,
                embedding)/(norm(label_mat, axis=1)*norm(embedding))
        label_idx = np.argmax(cosine)
        assert isinstance(label_idx, np.int64)
        try:
            assert cosine[label_idx] > 0.5
            label = label_df.index.values.tolist()[label_idx]
        except AssertionError:
            label = ''

        return label

    def format_rels(self, phrase_labels):
        """
        Formats relations in DyGIE++ format.

        This is where the choice to only label one relation per sentence is
        implemented, will need to come back here to make this process more
        nuanced.

        parameters:
            phrase_labels, dict of dict : keys are sentence indices,
                values are dictionaries, where keys are the phrase strings
                and values are the corresponding chosen label

        returns:
            relations, list of list: DyGIE++ formatted relations
        """
        # Will need to update to allow multiples
        relations = []
        for i, sent in enumerate(self.sentences):
            try:
                phrase_and_label = phrase_labels[i]
                phrase = list(phrase_and_label.keys())[0]
                label = phrase_and_label[phrase]
                start_ent, end_ent = self.choose_ents(phrase, i)
                rel = [start_ent[0], start_ent[1], end_ent[0], end_ent[1],
                        label]
                sent_rels = [rel]
                relations.append(sent_rels)
            except (KeyError, IndexError): # Catching IndexError is a stopgap
                                            # for the bigger problem of complex
                                            # sentences resulting in mismatches
                                            # of tokenizations (walk_VP bug)
                relations.append([])

        return relations

    def choose_ents(self, phrase, sent_idx):
        """
        Choose an entity for a relation. Meant to ensure that the first
        entity in the relation is before the relation's VP, and the second
        is after.

        NOTE: The current implementation here is to randomly choose two
        entities, in order to simplify the process of choosing relations.
        Therefore, nothing is actually enforced except that the two entities
        be different from one another. This is related to the
        "only one relation per sentence" paradigm, as there's not really a
        good way to choose only one pair if there are multiple relations.
        This also means that relations should be treated as undirected
        (as in my current evaluation paradigm for PICKLE), since the ordering
        of the entities is random.

        TODO improve this

        parameters:
            phrase, str: relation phrase
            sent_idx, int: sentence index

        returns:
            entities, list of list: two DyGIE++ formatted entities
        """
        # Get the sentence's entities
        sent_ents = self.entities[sent_idx]

        # Randomly choose two
        ## TODO improve
        # Importing this here because I won't need it when this isn't random
        from random import sample
        entities = sample(sent_ents, 2)
        return entities


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


    @staticmethod
    def parse_by_level(parse_string):
        """
        Turns a parse string into a dictionary that contains the labels for
        each level. The tree cannot be reconstructed form the output of this
        function, as it doesn't preserve parents within a level.

        parameters:
            parse_string, str: parse string to parse

        returns:
            mylabs, dict of list: keys are level indices, values are lists
                of the labels at that level in the parse tree
        """
        # Make parse tree
        parse_tree = ParentedTree.fromstring(parse_string)

        # Get label dictionary
        mylabs = defaultdict(list)
        for pos in parse_tree.treepositions():
            try:
                mylabs[len(pos)].append(parse_tree[pos].label())
            except AttributeError:
                if parse_tree[pos].isupper():
                    mylabs[len(pos)].append(parse_tree[pos])

        # Check for and remove empty list in last key
        max_key = max(list(mylabs.keys()))
        if mylabs[max_key] == []:
            del mylabs[max_key]

        return mylabs


    @staticmethod
    def visualize_parse(parse_string):
        """
        Pretty-prints the parse tree as rendered by nltk.
        """
        parse_tree = ParentedTree.fromstring('(' + parse_string + ')')
        parse_tree.pretty_print()

