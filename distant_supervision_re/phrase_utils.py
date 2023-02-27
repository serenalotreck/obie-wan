"""
Static methods for candidate phrase identification.

Author: Serena G. Lotreck
"""
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
    ## TODO update this to allow the lowest case
    if label in next_child._.labels:
        subset_child = next_child
        return subset_child
    # Recursive case
    else:
        # Get the labels that appear on the next level
        _, child_tups = get_child_tups(next_child)
        # Check which ones have children
        have_kids = [tup[1] for tup in child_tups if tup[0] != 'NO_LABEL']
        # If multiples have children, look for target label
        if len(have_kids) > 1:
            # Check how many times the target label occurs within each child
            contains_lab = defaultdict(list)
            for c in have_kids:
                if label in c._.parse_string:
                    count_lab = c._.parse_string.count(label)
                    contains_lab[count_lab].append(c) # Categorize children by
                                                      # how many times the
                                                      # target label appears
                                                      # within their trees
            lab_counts = list(contains_lab.keys())
            # The case where multiple occurrences of the target label are
            # nested and we want to stop at the one that's highest
            if (lab_counts.max() > 1) and highest:
                to_walk = contains_lab[0]
                return Abstract.subset_tree(to_walk, label)

            # The case where multiple occurrences are nested and we want to get
            # to the lowest one
            if (len(contains_lab) == 1) and not highest:
                pass
                ## TODO
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

