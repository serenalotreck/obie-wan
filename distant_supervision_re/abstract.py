"""
Defines the Abstract class.

Author: Serena G. Lotreck
"""
form collections import OrderedDict
import benepar, spacy
import nltk
from nltk import Tree, ParentedTree


class Abstract():
    """
    Contains the text and annotations for one abstract.
    """
    def __init__(text, sentences, entities):

        self.text = text
        self.sentences = sentences
        self.entities = entities
        self.candidate_sents = []
        self.constituency_parse_strings = []
        self.spacy_doc = None
        self.relations = []

    @staticmethod
    def parse_pred_dict(pred_dict):
        """
        Pulls input from a dictionary in the DyGIE++ prediction format
        (see https://github.com/dwadden/dygiepp/blob/master/doc/data.md)
        and calls constructor to build instance.

        parameters:
            pred_dict, dict: dygiepp formatted dictionary with entity
                predictions

        returns:
            abst, Abstract instance: instance of abstract class
        """
        # Get information to initialize an Abstract instance
        text = ' '.join(pred_dict["sentences"])
        sentences = pred_dict["sentences"])

        # Allow documents with both predicted or gold standard NER
        # Use predictions by default, unless there are none
        try:
            entities = pred_dict["predicted_ner"]
        except KeyError:
            entities = pred_dict["ner"]

        # Initialize Abstract instance
        abst = Abstract(text, sentences, entities)

        # Perform the higher functions of the class
        abst.set_candidate_sents()
        abst.set_const_parse_and_spacy_doc()
        abst.set_relations()

        return abst

    def set_candidate_sents(self):
        """
        Identifies sentences with two or more entities and creates a list
        of their indices in the sentence list attribute.
        """
        for i in len(self.entities):
            if len(self.entities[i]) >= 2:
                self.candidate_sents.append(i)

    def set_const_parse_strings_and_spacy_doc(self):
        """
        Performs constituency parse using the benepar pipe of spacy,
        and assigns the parse strings and spacy doc containing parse
        information to attributes.
        """
        nlp = spacy.load('en_core_sci_sm')
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        doc = nlp(text)
        for sent in doc.sents:
            self.constituency_parse_strings.append(sent._.parse_string)
        self.spacy_doc = doc

    def extract_rels(self, bert_model, label_embeddings):
        """
        For each candidate sentence, calls a helper function to get the
        candidate relation phrase from the constituency parse, and uses
        the sentence to generate a contextualized embedding for that phrase.
        Using context-averaged representations of the four relation types,
        performs clustering to assign each of the candidate relation phrases
        to a label group. Sets an attribute of DyGIE++-formatted list for
        relation annotations.

        parameters:
            bert_model, huggingface BertModel: the model to use to
                generate embeddings
            label_embeddings, dict: keys are strings representing the
                relation labels that we want to use, values are their
                context-averaged embeddings. These should be generated
                elsewhere in order to have consistent embeddings for these
                labels across all abstracts on which this method is used.
        """
        phrase_embeddings = {k:{} for k in self.candidate_sents}
        for sent in self.candidate_sents:

            # Prep the sentence for BERT

            # Then, generate the embeddings

            # Use helper to get the phrase to embed
            phrase = pick_phrase(sent)

            # Get this embedding out of the BERT output and add to dict
            embedding = 
            phrase_embeddings[sent][phrase] = embedding

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

        # Walk the tree and its labels
        # We want everything from the VP besides the nested NP 
        # First, get the VP
        first_level_labels = [c._.labels for c in sent._.children]
        top_vp = None
        for child, label in zip(sent._.children, first_level_labels):
            if (len(label) == 1) and (label[0] == 'VP'):
                top_vp = child
        # Then, we walk, and keep anything that's not the terminal NP
        phrase = walk_VP('', top_vp)

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
        next_labels = []
        for c in next_child._.children:
            if len(c._.labels) == 0:
                next_labels.append('NO_LABEL')
            else:
                assert len(c._.labels) == 1
                next_labels.append(c._.labels[0])
        kids = next_child._.children
        child_dict = OrderedDict({lab:c for lab, c in zip(next_labels, kids)})

        # Base case
        if 'NP' in next_labels:
            phrase_add = [child_dict[l].text for l in next_labels if l != 'NP']
            phrase += ' '.join(phrase_add)
            return phrase
        # Recursive case
        else:
            # Add anything that doesn't have a child
            # Leaf nodes have no labels in benepar
            phrase_add = [child_dict[l] for l in next_labels
                    if l == 'NO_LABEL']
            phrase += ' '.join(phrase_add)
            # Continue down the one that does
            ## TODO what to do if there's more than one on the same level
            ## that has children? Is that possible?
            to_walk = [child_dict[l] for l in next_labels if l != 'NO_LABEL']
            assert len(to_walk) == 1 # Make sure only one has children
            to_walk = to_walk[0]
            walk_VP(phrase, to_walk)

    def set_relations(self, relations):
        """
        Set the output of the relation extraction process as an attribute.
        """
        self.relations = relations

    @staticmethod
    def visualize_parse(parse_string):
        """
        Pretty-prints the parse tree as rendered by nltk.
        """
        parse_tree = ParentedTree.fromstring('(' + parse_string + ')')
        print(parse_tree.pretty_print())

