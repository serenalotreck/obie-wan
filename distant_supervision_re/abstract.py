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
        self.skipped_sents = []
        self.success_cats = {}
        self.skipped_cats = {} # keys are sent_idx, values are dicts
                                    # with list of labels at each level. Num
                                    # of levels depends on what's passed to
                                    # set_skip type

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
        skipped = 0
        total = 0
        phrase_labels = {k:{} for k in self.cand_sents}
        for sent in self.cand_sents:

            # Use helper to get the phrase to embed
            try:
                phrase = self.pick_phrase(sent)
                struct_dict = Abstract.parse_by_level(self.const_parse[sent])
                self.success_cats[sent] = struct_dict
            except AttributeError:
                struct_dict = Abstract.parse_by_level(self.const_parse[sent])
                self.skipped_sents.append(sent)
                self.skipped_cats[sent] = struct_dict
                skipped += 1
                total += 1
                continue

            # Get this embedding out of the BERT output and add to dict
            sent_text = ' '.join(self.sentences[sent])
            try:
                embedding = be.get_phrase_embedding(sent_text, phrase,
                    tokenizer, model)
                # Get distances to choose label
                label = self.compute_label(label_df, embedding)
                if label != '':
                    phrase_labels[sent][phrase] = label
            except TypeError:
                struct_dict = Abstract.parse_by_level(self.const_parse[sent])
                self.skipped_sents.append(sent)
                self.skipped_cats[sent] = struct_dict
                skipped += 1 ## See TODO note in pick_phrase


            total += 1

        # Format the relations in DyGIE++ format
        relations = self.format_rels(phrase_labels)
        self.set_relations(relations)

        return skipped, total

    def pick_phrase(self, sent_idx):
        """
        Uses constituency parse to choose phrase to embed.

        The heuristic rule used in this process is based on the observation
        that the benepar constituency parse usually has a verb phrase (VP)
        that encompasses both the verb that constitutes the relation, as well
        as the noun phrase (NP) that contains the second entity. We therefore
        want to get at the part of the VP that contains the verb and any
        prepositions, not including the entity. For simplicity's sake, I am
        starting by including anything that isn't the nested NP; futher
        refinement of this approach may be needed. This approach also relies
        on the assumption that on the first level of the tree, there is only
        one VP and one NP. I haven't reviewed parse trees for enough sentences
        to determine if that is a truly reliable assumption for the scientific
        literature, but my writing intuition says that it is -- TODO verify

        For the moment, I am treating any sentence with more than 2 entities
        as a sentence in which there is only one relation present. I am aware
        that this is a major limitation, as in the corpus there are many
        instances in which there are multiple relations in a sentence,
        especially in cases where a sentence contains a list of entities that
        are related to one or more other entities. However, this is a starting
        point, and I will come back to this after implementing the simplest
        version -- TODO

        parameters:
            sent_idx, int: sentence from which to choose a phrase
            include_preps, bool: whether or not to include prepositions
                that are part of the final noun phrase in a sentence

        returns:
            phrase, str: the phrase to embed
        """
        # Get the spacy object version of the sentence
        sent = list(self.spacy_doc.sents)[sent_idx]

        ## TODO: This fails catastrophically on sentences with multiple
        ## NPs within a VP, as well as sentences whose parse trees have
        ## multiple nodes with children on the same level. Need to refine
        ## this quite a bit, for now I'm just tracking failures and skipping
        ## them
        # Walk the tree and its labels
        # We want everything from the VP besides the nested NP 
        # First, get the VP
        first_level_labels = [c._.labels for c in sent._.children]
        top_vp = None
        for child, label in zip(sent._.children, first_level_labels):
            if (len(label) == 1) and (label[0] == 'VP'):
                top_vp = child
        # Then, we walk, and keep anything that's not the terminal NP
        phrase = Abstract.walk_VP('', top_vp)

        return phrase

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
        ## TODO confirm it's safe to assume there's only one label per tuple
        ### UPDATE: It's not -- need to decide what to do about it, implemented the
        ### elif's below as a stopgap measure
        next_labels = []
        for c in next_child._.children:
            if len(c._.labels) == 0:
                next_labels.append('NO_LABEL')
            elif len(c._.labels) == 1:
                next_labels.append(c._.labels[0])

            ## This is a hideous stopgap measure, TODO fix
            elif len(c._.labels) > 1:
                if 'NP' in c._.labels and 'S' not in c._.labels:
                    next_labels.append('NP')
                elif 'S' in c._.labels:
                    other_labs = list(c._.labels)
                    other_labs.remove('S')
                    assert len(other_labs) == 1
                    next_labels.append(other_labs[0])

        kids = next_child._.children
        child_dict = OrderedDict({lab:c for lab, c in zip(next_labels, kids)})

        # Base case
        if 'NP' in next_labels:
            phrase_add = [child_dict[l].text for l in next_labels if l != 'NP']
            phrase += ' ' + ' '.join(phrase_add)
            return phrase.strip() # Removes leading whitespace
        # Recursive case
        else:
            # Add anything that doesn't have a child
            # Leaf nodes have no labels in benepar
            phrase_add = [child_dict[l].text for l in next_labels
                    if l == 'NO_LABEL']
            phrase += ' ' + ' '.join(phrase_add)
            # Continue down the one that does
            ## TODO what to do if there's more than one on the same level
            ## that has children? Is that possible?
            to_walk = [child_dict[l] for l in next_labels if l != 'NO_LABEL']
            if len(to_walk) == 1:
                to_walk = to_walk[0]
                return Abstract.walk_VP(phrase, to_walk)
            elif len(to_walk) == 0:
                return phrase.strip()
            else:
                print('It is possible for there to be more than one level '
                        'with kids')

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
        print(parse_tree.pretty_print())

