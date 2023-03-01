"""
Spot checks for phrase_utils.

Author: Serena G. Lotreck
"""
import pytest
import pandas as pd
import spacy
import benepar
import sys
sys.path.append('../distant_supervision_re')
import phrase_utils as pu


class TestRelHelpers:
    """
    Class to test the testable helpers for extract_rels. I'm not writing a
    unit test for extract_rels itself, beacuse its functionality relies on
    the code in bert_embeddings.py in addition to its helpers, so it's
    covered by the tests for the bert_embeddings code, as well as the
    tests for its helpers
    """

    ############################ Fixtures ################################

    @pytest.fixture
    def dygiepp(self):
        return {'doc_key': 'doc2',
                'dataset': 'scierc',
                'sentences': [['Jasmonic', 'acid', 'is', 'a', 'hormone', '.'],
                    ['Jasmonic', 'acid', 'upregulates', 'Protein', '1', '.'],
                    ['Jasmonic', 'acid', 'is', 'found', 'in', 'Arabidopsis',
                        'thaliana', '.']],
                'ner': [[[0, 1, 'Plant_hormone']],
                    [[6, 7, 'Plant_hormone'], [9, 10, 'Protein']],
                    [[12, 13, 'Plant_hormone'],
                        [17, 18, 'Multicellular_organism']]]}

    @pytest.fixture
    def nlp(self):
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        return nlp

    @pytest.fixture
    def top_VP_one_word(self, nlp):
        text = 'upregulates Protein 1'
        doc = nlp(text)
        sent = list(doc.sents)[0]
        return sent

    @pytest.fixture
    def top_VP_multi_word(self, nlp):
        text = 'is found in Arabidopsis thaliana'
        doc = nlp(text)
        sent = list(doc.sents)[0]
        return sent

    @pytest.fixture
    def candidate_phrase_one_word(self, nlp):
        return [list(nlp('upregulates').sents)[0]]

    @pytest.fixture
    def candidate_phrase_multi_word(self, nlp):
        return [list(nlp(w).sents)[0] for w in ['is', 'found', 'in']]

    @pytest.fixture
    def label_df(self):
        """
        Dummy embeddings for easy testing
        """
        label_dict = {'activates': [0.5, 0.1],
                'inhibits': [0.1, 0.1],
                'produces': [1, 0.5],
                'interacts': [1, 1],
                'is_in': [0.1, 1]}

        return pd.DataFrame.from_dict(label_dict, orient='index')

    @pytest.fixture
    def embed_gets_label(self):
        return [0.3, 0.1]

    @pytest.fixture
    def embed_no_label(self):
        # I'm not sure this is a legitimate scenario, but I wanted to
        # test the scenario where the similarity is less than 0.5
        return [-0.1, -0.5]

    @pytest.fixture
    def correct_label(self):
        return 'activates'

    @pytest.fixture
    def phrase_labels(self):
        return {1:{'upregulates':'activates'}, 2:{'is found in':'is_in'}}

    ############################### Tests ################################

    def test_walk_VP_one_word(self, top_VP_one_word,
            candidate_phrase_one_word):
        phrase = pu.walk_VP([], top_VP_one_word)

        assert phrase == candidate_phrase_one_word

    def test_walk_VP_multi_word(self, top_VP_multi_word,
            candidate_phrase_multi_word):
        phrase = pu.walk_VP([], top_VP_multi_word)
        assert phrase == candidate_phrase_multi_word

    def pick_phrase_one_word(self, abstract, candidate_phrase_one_word):
        sent_idx = 1
        phrase = abstract.pick_phrase(sent_idx)

        assert phrase == candidate_phrase_one_word

    def pick_phrase_multi_word(self, abstract, candidate_phrase_multi_word):
        sent_idx = 2
        phrase = abstract.pick_phrase(sent_idx)

        assert phrase == candidate_phrase_multi_word

    def test_compute_label_gets_label(self, label_df, embed_gets_label,
            correct_label):
        label = pu.compute_label(label_df, embed_gets_label)
        assert label == correct_label

    def test_compute_label_no_label(self, label_df, embed_no_label):
        label = pu.compute_label(label_df, embed_no_label)

        assert label == ''

class TestSubsetTree:
    """
    Test parse_sbar, parse_mult_S, and the subset_tree function on which both
    rely.

    NOTE: As a result of needing to independently turn the candidate phrase
    into spacy docs, there are differences between the parse trees of the
    correct subset of the full sentence and the "right answer". I am relatively
    confident that this is an artefact of testing and not an issue that would
    occur when just subsetting the tree, so instead of comparing the entire
    parse string in the tests here, I just compare the text of the subset to
    the text of the correct answer.
    """

    ############################ Fixtures ################################

    @pytest.fixture
    def nlp(self):
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        return nlp

    @pytest.fixture
    def class1a_sent(self):
        return ('The signal transduction pathway regulating this response is '
                'not fully understood , but several compounds have been '
                'identified which are capable of inducing proteinase inhibitor '
                'synthesis in tomato and potato leaves .')

    @pytest.fixture
    def class1a_next_child(self, class1a_sent, nlp):
        doc = nlp(class1a_sent)
        next_child = list(doc.sents)[0]
        return next_child

    @pytest.fixture
    def class1a_subset_child(self, nlp):
        doc = nlp('which are capable of inducing proteinase inhibitor '
                    'synthesis in tomato and potato leaves')
        subset_child = list(doc.sents)[0].text
        return [subset_child]

    @pytest.fixture
    def class1b_sent(self):
         return ('The same complexity is evident which the role of '
                'phytoalexin accumulation in resistance is analysed .')

    @pytest.fixture
    def class1b_next_child(self, class1b_sent, nlp):
        doc = nlp(class1b_sent)
        next_child = list(doc.sents)[0]
        return next_child

    @pytest.fixture
    def class1b_subset_child(self, nlp):
        doc = nlp('which the role of phytoalexin accumulation in '
                'resistance is analysed')
        subset_child = list(doc.sents)[0].text
        return [subset_child]

    @pytest.fixture
    def class2_sent(self):
        return ('The separate , and distant , locations of the receptor and '
                'the responsive genes means that the event in which the '
                'signal is perceived by the receptor must be relayed to '
                'the genes by means of a second messenger system .')

    @pytest.fixture
    def class2_next_child(self, class2_sent, nlp):
        doc = nlp(class2_sent)
        next_child = list(doc.sents)[0]
        return next_child

    @pytest.fixture
    def class2_subset_child(self, nlp):
        doc = nlp('in which the signal is perceived by the receptor')
        subset_child = list(doc.sents)[0].text
        return [subset_child]

    @pytest.fixture
    def sbar_class3_sent(self):
        return ('We conclude that plants prioritize resistance of reproductive '
        'tissues over vegetative tissues , and that a chewing herbivore '
        'species is the main driver of responses in flowering B. nigra .')

    @pytest.fixture
    def sbar_class3_next_child(self, sbar_class3_sent, nlp):
        doc = nlp(sbar_class3_sent)
        next_child = list(doc.sents)[0]
        return next_child

    @pytest.fixture
    def sbar_class3_subset_child(self, nlp):
        sbar1 = ('that plants prioritize resistance of reproductive tissues '
                    'over vegetative tissues')
        sbar2 = ('that a chewing herbivore species is the main driver of '
                    'responses in flowering B. nigra')
        subset_children = [list(nlp(s).sents)[0].text for s in [sbar1, sbar2]]
        return subset_children

    @pytest.fixture
    def s_class3_sent(self):
        return (' In this chapter , we describe the properties of systemin '
        'and its precursor prosystemin , and we summarize the evidence '
        'supporting a role for systemin as an initial signal that regulates '
        'proteinase inhibitor synthesis in response to wounding .') 
    ############################### Tests ################################

    def test_subset_tree_class1a(self, class1a_next_child, class1a_subset_child):

        subset = pu.subset_tree(class1a_next_child, 'SBAR', highest=False)
        subset_text = [s.text for s in subset]

        assert subset_text == class1a_subset_child


    def test_subset_tree_class1b(self, class1b_next_child, class1b_subset_child):

        subset = pu.subset_tree(class1b_next_child, 'SBAR', highest=False)
        subset_text = [s.text for s in subset]

        assert subset_text == class1b_subset_child


    def test_subset_tree_class2(self, class2_next_child, class2_subset_child):

        subset = pu.subset_tree(class2_next_child, 'SBAR', highest=False)
        subset_text = [s.text for s in subset]

        assert subset_text == class2_subset_child

    def test_subset_tree_sbar_class3(self, sbar_class3_next_child, sbar_class3_subset_child):

        subset = pu.subset_tree(sbar_class3_next_child, 'SBAR', highest=False)
        subset_text = [s.text for s in subset]

        assert subset_text == sbar_class3_subset_child


class TestCategoryTracking:
    """
    Test the functions that keep track of sentence structures for
    categorization.

    NOTE: I didn't end up using this function, but leaving it in phrase_utils
    as well as this test here in case I want it in the future.
    """

    ############################ Fixtures ################################

    @pytest.fixture
    def nlp(self):
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        return nlp

    @pytest.fixture
    def sent1(self):
        return 'Jasmonic acid is a hormone.'

    @pytest.fixture
    def sent2(self):
        return ('JAs biosynthesis, perception, transport, signal transduction '
                'and action have been extensively investigated.')

    @pytest.fixture
    def sent3(self):
        return ('In particular, as a signaling molecule, JAs can effectively '
                'mediate responses against environmental stresses by inducing '
                'a series of genes expression.')

    @pytest.fixture
    def sent1_cat_dict(self):
        return {
                0: ['S'],
                1: ['NP', 'VP', '.'],
                2: ['JJ', 'NN', 'VBZ', 'NP'],
                3: ['DT', 'NN']
                }

    @pytest.fixture
    def sent2_cat_dict(self):
        return {
                0: ['S'],
                1: ['NP', 'VP', '.'],
                2: ['NNP', 'NN', ',', 'NP', ',', 'NP', ',', 'NP', 'CC', 'NP',
                    'VBP', 'VP'],
                3: ['NN', 'NN', 'NN', 'NN', 'NN', 'VBN', 'ADVP', 'VP'],
                4: ['RB', 'VBN']
                }

    @pytest.fixture
    def sent3_cat_dict(self):
        return {
                0: ['S'],
                1: ['PP', ',', 'PP', ',', 'NP', 'VP', '.'],
                2: ['IN', 'ADJP', 'IN', 'NP', 'NNP', 'MD', 'ADVP', 'VP'],
                3: ['JJ', 'DT', 'NN', 'NN', 'RB', 'VB', 'NP', 'PP'],
                4: ['NP', 'PP', 'IN', 'S'],
                5: ['NNS', 'IN', 'NP', 'VP'],
                6: ['JJ', 'NNS', 'VBG', 'NP'],
                7: ['NP', 'PP'],
                8: ['DT', 'NN', 'IN', 'NP'],
                9: ['NNS', 'NN']
                }

    ############################### Tests ################################

    def test_parse_by_level_sent1(self, nlp, sent1, sent1_cat_dict):

        parse_string = list(nlp(sent1).sents)[0]._.parse_string
        mylabs = pu.parse_by_level(parse_string)

        assert mylabs == sent1_cat_dict

    def test_parse_by_level_sent2(self, nlp, sent2, sent2_cat_dict):

        parse_string = list(nlp(sent2).sents)[0]._.parse_string
        mylabs = pu.parse_by_level(parse_string)

        assert mylabs == sent2_cat_dict

    def test_parse_by_level_sent3(self, nlp, sent3, sent3_cat_dict):

        parse_string = list(nlp(sent3).sents)[0]._.parse_string
        mylabs = pu.parse_by_level(parse_string)

        assert mylabs == sent3_cat_dict
