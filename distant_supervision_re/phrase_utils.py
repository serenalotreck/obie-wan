"""
Static methods for candidate phrase identification.

Author: Serena G. Lotreck
"""
from collections import defaultdict
import numpy as np
from numpy.linalg import norm
from nltk import ParentedTree


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
        Class 4: There are sibling nodes that contain SBAR annotations, which
            can fall into any of the other three classes

    TODO decide how to deal with  Class 1 (whether or not to annotate)
    TODO figure out how to implement this parse for Class 4

    parameters:
        sent, spacy Doc object sentence: sentence to parse

    returns:
        check_to_walk, list of spacy Span objects: phrases to walk,
            starting at the lowest SBAR annotation
    """
    child = subset_tree(sent, 'SBAR', highest=False)
    return [child]


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
    child = subset_tree(sent, 'S')
    return [child]


def walk_VP(phrase, next_child):
    """
    Walk recursively through a VP, adding anything that's not the
    terminal NP to the phrase string.

    parameters:
        phrase, list: list of words that make up the phrase to add to
        next_child, spacy Span object: the next child to check

    returns:
        phrase, list: updated phrase list
    """
    # Get the labels that are on the next level
    next_labels, child_tups = get_child_tups(next_child)
    # Base case
    if 'NP' in next_labels:
        phrase_add = [t[1] for t in child_tups
                if (t[0] != 'NP') & (t[0] != 'PP')]
        phrase.extend(phrase_add)
        return phrase
    # Recursive case
    else:
        # Add anything that doesn't have a child
        # Leaf nodes have no labels in benepar
        phrase_add = [t[1] for t in child_tups
            if t[0] == 'NO_LABEL']
        phrase.extend(phrase_add)
        # Continue down the one that does
        to_walk = [t[1] for t in child_tups if t[0] != 'NO_LABEL']
        if len(to_walk) == 1:
            to_walk = to_walk[0]
            return walk_VP(phrase, to_walk)
        elif len(to_walk) == 0:
            return phrase
        else:
            ## TODO implement dropping leading PP's here
            return 'NO PHRASE: Multiple levels with kids'


def subset_tree(next_child, label, highest=True):
    """
    Return the benepar-parsed spacy object starting at the node with the
    label label.

    parameters:
        next_child, spacy Span object: the next child to check
        label, str: parse label to look for
        highest, bool: if there are multiple of the requested label in the
            tree, whether or not to return the highest one or the one closest
            to the leaves

    returns:
        subset_child, spacy Span object: subset child
    """
    # Base case
    # There are two cases that trigger the base case:
    # 1. The root node has the target label, and we only want the first
    # occurrence so we return, regardless of what's underneath; or
    # 2. The root node has the target label and we only want the lowest, so we
    # only want to return if there are no other occurrences of the label left
    # in the tree
    if label in next_child._.labels and (highest or
            (not highest and next_child._.parse_string.count(label) == 1)):
        subset_child = next_child
        return subset_child
    # Recursive case
    else:
        # Get the labels that appear on the next level
        _, child_tups = get_child_tups(next_child)
        # Check which ones have children
        have_kids = [tup[1] for tup in child_tups if tup[0] != 'NO_LABEL']
        # Check how many times the target label occurs within each child
        contains_lab = defaultdict(list)
        for c in have_kids:
            if label in c._.parse_string:
                count_lab = c._.parse_string.count(label)
                contains_lab[count_lab].append(c) # Categorize children by
                                                  # how many times the
                                                  # target label appears
                                                  # within their trees
        sibling_lens = [len(contains_lab[i]) for i in contains_lab.keys()
                if i != 0 and len(contains_lab[i]) > 1]
        # The case where sibling nodes contain the target label
        if len(sibling_lens) > 0:
            ## TODO figure out how to deal with this in a way that returns the
            ## recursion results of the two siblings separately
            assert True, (f'The requested label is sibling with itself')
        # The case where labels are nested or there is only 1
        else:
            nonzero_keys = [i for i in contains_lab.keys() if i != 0]
            assert len(nonzero_keys) == 1 # This should be true, safety measure
            child_key = nonzero_keys[0]
            to_walk = contains_lab[child_key][0] # We've already checked none of them
                                         # longer than 1
            return subset_tree(to_walk, label, highest)


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


def visualize_parse(parse_string):
    """
    Pretty-prints the parse tree as rendered by nltk.
    """
    parse_tree = ParentedTree.fromstring('(' + parse_string + ')')
    parse_tree.pretty_print()

