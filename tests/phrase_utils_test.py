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


class TestWalkVP:
    """
    Class to test the walk_VP function

    NOTE: before calling walk_VP, the sentence has to undergo the same
    preprocessing that happens in pick_phrase (tested in abstract_test.py).
    The preprocessing is designed to make sure that the only thing that gets
    passed to walk_VP is a single, parsable VP. Therefore, many of the tests
    here end up being basically identical in terms of edge cases; however, I've
    left them all in case there are issues with pick_phrase that originate
    here, as they correspond to the test cases in pick_phrase.
    """

    ############################ Fixtures ################################
    @pytest.fixture
    def nlp(self):
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        return nlp


    @pytest.fixture
    def internal_sbar(self, nlp):
        return list(nlp('plays an important role in the SA-mediated '
                        'attenuation of the JA signaling pathway').sents)[0]

    @pytest.fixture
    def vp_w_pp(self, nlp):
        return list(nlp('varied from no effect to the more generally observed '
                        '1.4-to 3.0-fold stimulation').sents)[0]

    @pytest.fixture
    def normal_vp(self, nlp):
        # Have to use whole phrase to build out VP instead of just providing
        # VP, beacuse it changes the parse tree to provide a partial sentence
        # and gives the incorrect answer
        sent = list(nlp('JA inhibited active sucrose uptake in beet '
            'roots').sents)[0]
        vp = pu.subset_tree(sent, 'VP', highest=True)[0]
        return vp

    @pytest.fixture
    def mult_words(self, nlp):
        return list(nlp('did not modify or counteract the auxin effect').sents)[0]

    @pytest.fixture
    def internal_sbar_phrase(self, internal_sbar, nlp):
        phrase = internal_sbar[0:1]
        return [phrase]

    @pytest.fixture
    def vp_w_pp_phrase(self, vp_w_pp, nlp):
        phrase = vp_w_pp[0:1]
        return [phrase]

    @pytest.fixture
    def normal_vp_phrase(self, normal_vp, nlp):
        phrase = normal_vp[0:1]
        return [phrase]

    @pytest.fixture
    def mult_words_phrase(self, mult_words, nlp):
        phrase = [mult_words[i:i+1] for i in range(5)]
        return phrase

    ############################### Tests ################################

    def test_internal_sbar(self, internal_sbar, internal_sbar_phrase):

        phrase = pu.walk_VP([], internal_sbar)

        assert phrase == internal_sbar_phrase

    def test_vp_w_pp(self, vp_w_pp, vp_w_pp_phrase):

        phrase = pu.walk_VP([], vp_w_pp)

        assert phrase == vp_w_pp_phrase

    def test_normal_vp(self, normal_vp, normal_vp_phrase):

        phrase = pu.walk_VP([], normal_vp)

        assert phrase == normal_vp_phrase

    def test_mult_words(self, mult_words, mult_words_phrase):

        phrase = pu.walk_VP([], mult_words)

        assert phrase == mult_words_phrase


class TestGetChildTups:
    """
    Class to test the get_child_tups function
    """

    ############################ Fixtures ################################

    @pytest.fixture
    def nlp(self):
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        return nlp

    @pytest.fixture
    def one_with_child_sent(self, nlp):
        text = ('indicated that ABA signaling positively regulates root '
                'defense to R. solanacearum')
        doc = nlp(text)
        sent = list(doc.sents)[0]
        return sent

    @pytest.fixture
    def mult_with_child_sent(self, nlp):
        text = ('Plant responses to one attacker can interfere with responses '
                'to a second attacker')
        doc = nlp(text)
        sent = list(doc.sents)[0]
        return sent

    @pytest.fixture
    def double_label_sent(self, nlp):
       text = ('which are capable of inducing proteinase inhibitor synthesis in '
               'tomato and potato leaves')
       doc = nlp(text)
       sent = list(doc.sents)[0]
       return sent

    @pytest.fixture
    def one_with_child_tups(self, one_with_child_sent):
        c1_label = 'NO_LABEL'
        c1 = one_with_child_sent[0:1]
        c2_label = 'SBAR'
        c2 = one_with_child_sent[1:]
        return [(c1_label, c1), (c2_label, c2)]

    @pytest.fixture
    def one_with_child_list(self):
        return ['NO_LABEL', 'SBAR']

    @pytest.fixture
    def mult_with_child_tups(self, mult_with_child_sent):
        c1_label = 'NP'
        c1 = mult_with_child_sent[0:5]
        c2_label = 'VP'
        c2 = mult_with_child_sent[5:]
        return [(c1_label, c1), (c2_label, c2)]

    @pytest.fixture
    def mult_with_child_list(self):
        return ['NP', 'VP']

    @pytest.fixture
    def double_label_tups(self, double_label_sent):
        c1_label = 'WHNP'
        c1 = double_label_sent[0:1]
        c2_label = 'VP'
        c2 = double_label_sent[1:]
        return [(c1_label, c1), (c2_label, c2)]

    @pytest.fixture
    def double_label_list(self):
        return ['WHNP', 'VP']

    ############################### Tests ################################

    def test_one_with_child(self, one_with_child_sent, one_with_child_tups,
            one_with_child_list):

        next_labels, child_tups = pu.get_child_tups(one_with_child_sent)

        assert next_labels == one_with_child_list
        assert child_tups == one_with_child_tups

    def test_mult_with_child(self, mult_with_child_sent, mult_with_child_tups,
            mult_with_child_list):

        next_labels, child_tups = pu.get_child_tups(mult_with_child_sent)

        assert next_labels == mult_with_child_list
        assert child_tups == mult_with_child_tups

    def test_double_label(self, double_label_sent, double_label_tups,
            double_label_list):

        next_labels, child_tups = pu.get_child_tups(double_label_sent)

        assert next_labels == double_label_list
        assert child_tups == double_label_tups


class TestSubsetTree:
    """
    Test subset_tree.

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
    def class1a_subset_child(self, class1a_next_child):
        subset_child = class1a_next_child[18:-1]
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
    def class1b_subset_child(self, class1b_next_child):
        subset_child = class1b_next_child[5:-1]
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
    def class2_subset_child(self, class2_next_child):
        subset_child = class2_next_child[18:27]
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
    def sbar_class3_subset_child(self, sbar_class3_next_child):
        sbar1 = sbar_class3_next_child[2:12]
        sbar2 = sbar_class3_next_child[14:-1]
        subset_children = [sbar1, sbar2]
        return subset_children

    @pytest.fixture
    def s_class3_sent(self):
        return ('In this chapter , we describe the properties of systemin '
        'and its precursor prosystemin , and we summarize the evidence '
        'supporting a role for systemin as an initial signal that regulates '
        'proteinase inhibitor synthesis in response to wounding .')

    @pytest.fixture
    def s_class3_next_child(self, nlp, s_class3_sent):
        doc = nlp(s_class3_sent)
        next_child = list(doc.sents)[0]
        return next_child

    @pytest.fixture
    def s_class3_subset_child(self, s_class3_next_child):
        s1 = s_class3_next_child[4:14]
        s2 = s_class3_next_child[16:-1]
        subset_children = [s1, s2]
        return subset_children

    @pytest.fixture
    def vp_class1_sent(self):
        return ('Our study indicated that ABA signaling positively regulates '
                'root defense to R. solanacearum.')

    @pytest.fixture
    def vp_class1_next_child(self, vp_class1_sent, nlp):
        # Note that this is a case where we've already subset by SBAR within
        # pick_phrase, so we will do that here manually
        next_child = list(nlp(vp_class1_sent).sents)[0]
        next_child = next_child[3:-1]
        return next_child

    @pytest.fixture
    def vp_class1_subset_child(self, vp_class1_next_child):
        s1 = vp_class1_next_child[4:]
        subset_children = [s1]
        return subset_children

    @pytest.fixture
    def vp_cc_vp_class3_sent(self):
        return ('Dual attack increased plant resistance to caterpillars, but '
                'compromised resistance to aphids.')

    @pytest.fixture
    def vp_cc_vp_class3_next_child(self, vp_cc_vp_class3_sent, nlp):
        doc = nlp(vp_cc_vp_class3_sent)
        next_child = list(doc.sents)[0]
        return next_child

    @pytest.fixture
    def vp_cc_vp_class3_subset_child(self, vp_cc_vp_class3_next_child):
        s1 = vp_cc_vp_class3_next_child[2:7]
        s2 = vp_cc_vp_class3_next_child[9:13]
        subset_children = [s1, s2]
        return subset_children

    ############################### Tests ################################

    def test_subset_tree_class1a(self, class1a_next_child, class1a_subset_child):

        subset = pu.subset_tree(class1a_next_child, 'SBAR', highest=False)

        assert subset == class1a_subset_child


    def test_subset_tree_class1b(self, class1b_next_child, class1b_subset_child):

        subset = pu.subset_tree(class1b_next_child, 'SBAR', highest=False)

        assert subset == class1b_subset_child


    def test_subset_tree_class2(self, class2_next_child, class2_subset_child):

        subset = pu.subset_tree(class2_next_child, 'SBAR', highest=False)

        assert subset == class2_subset_child

    def test_subset_tree_sbar_class3(self, sbar_class3_next_child, sbar_class3_subset_child):

        subset = pu.subset_tree(sbar_class3_next_child, 'SBAR', highest=False)

        assert subset == sbar_class3_subset_child

    def test_subset_tree_s_class3(self, s_class3_next_child, s_class3_subset_child):

        subset = pu.subset_tree(s_class3_next_child, 'S', highest=True,
                ignore_root=True)

        assert subset == s_class3_subset_child

    def test_subset_tree_vp_class1(self, vp_class1_next_child,
            vp_class1_subset_child):

        subset = pu.subset_tree(vp_class1_next_child, 'VP', highest=True)

        assert subset == vp_class1_subset_child

    def test_subset_tree_vp_cc_vp_class3(self, vp_cc_vp_class3_next_child,
            vp_cc_vp_class3_subset_child):

        subset = pu.subset_tree(vp_cc_vp_class3_next_child, 'VP',
                highest=False)

        assert subset == vp_cc_vp_class3_subset_child


class TestComputeLabel:
    """
    Class to test compute_label
    """

    ############################ Fixtures ################################

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

    ############################### Tests ################################

    def test_compute_label_gets_label(self, label_df, embed_gets_label,
            correct_label):
        label = pu.compute_label(label_df, embed_gets_label)
        assert label == correct_label

    def test_compute_label_no_label(self, label_df, embed_no_label):
        label = pu.compute_label(label_df, embed_no_label)

        assert label == ''


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
