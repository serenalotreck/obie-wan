"""
Defines the Abstract class.

Author: Serena G. Lotreck
"""

class Abstract():
    """
    Contains the text and annotations for one abstract.
    """
    def __init__(text, sentences, entities):

        self.text = text
        self.sentences = sentences
        self.entities = entities
        self.candidate_sents = []
        self.constituency_parse = []
        self.relations = []


    def parse_pred_dict(pred_dict):
        """
        Pulls input from a dictionary in the DyGIE++ prediction format
        (see https://github.com/dwadden/dygiepp/blob/master/doc/data.md)
        and calls constructor to build instance.

        parameters:
            pred_dict, dict: dygiepp formatted dictionary with entity
                predictions

        returns:
            abs, Abstract instance: instance of abstract class
        """
        pass

    def set_candidate_sents():
        """
        Identifies sentences with two or more entities and creates a list
        of their indices in the sentence list attribute.
        """
        pass

    def set_constituency_parse():
        """
        Performs constituency parse and assigns its output to an attribute.
        """
        pass

    def set_relations(relations):
        """
        Set the output of the relation extraction process as an attribute.
        """
        self.relations = relations
