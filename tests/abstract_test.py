"""
Tests for the Abstract class.

Author: Serena G. Lotreck
"""
import pytest
import sys
sys.path.append('../distant_supervision_re')
import pandas as pd
import spacy
import benepar
from abstract import Abstract


class TestAbstractSetup:
    """
    Tests for the setup function & accompanying setters of the Abstract class
    """

    ############################### Fixtures ################################

    @pytest.fixture
    def nlp(self):
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        return nlp

    @pytest.fixture
    def dygiepp_gold(self):
        return {'doc_key':'doc1',
                'dataset':'scierc',
                'sentences':[['Hello', 'world', '.'],
                    ['My', 'name', 'is', 'Sparty', 'and', 'I',
                        'work', 'at', 'MSU', '.']],
                'ner':[[],[[6, 6, 'Person'], [11, 11, 'Organization']]]}

    @pytest.fixture
    def dygiepp_pred(self):
        return {'doc_key':'doc1',
                'dataset':'scierc',
                'sentences':[['Hello', 'world', '.'], # spacy doesn't separate
                                                    # sentences on !
                    ['My', 'name', 'is', 'Sparty', 'and', 'I',
                        'work', 'at', 'MSU', '.']],
                'ner':[[],[]], # PURE requires this field even if there is no
                                # gold standard annotation
                'predicted_ner':[[],[[6, 6, 'Person'],
                    [11, 11, 'Organization']]]}

    @pytest.fixture
    def text(self):
        return 'Hello world . My name is Sparty and I work at MSU .'

    @pytest.fixture
    def sentences(self, dygiepp_gold):
        return dygiepp_gold["sentences"]

    @pytest.fixture
    def entities(self, dygiepp_gold):
        return dygiepp_gold["ner"]

    @pytest.fixture
    def cand_sents(self):
        return [1]

    @pytest.fixture
    def const_parse(self):
        return ['(NP (UH Hello) (NN world) (. .))',
                '(S (S (NP (PRP$ My) (NN name)) (VP (VBZ is) (NP (NNP '
                'Sparty)))) (CC and) (S (NP (PRP I)) (VP (VBP work) (PP '
                '(IN at) (NP (NNP MSU))))) (. .))']

    @pytest.fixture
    def spacy_doc(self, nlp, text):
        return nlp(text)

    @pytest.fixture
    def abst_gold(self, dygiepp_gold, text, sentences, entities,
            cand_sents, const_parse, spacy_doc):
        return Abstract(dygiepp_gold, text, sentences, entities,
                cand_sents, const_parse, spacy_doc)

    @pytest.fixture
    def abst_pred(self, dygiepp_pred, text, sentences, entities,
            cand_sents, const_parse, spacy_doc):
        return Abstract(dygiepp_pred, text, sentences, entities,
                cand_sents, const_parse, spacy_doc)

    @pytest.fixture
    def abst_set_cand_sents(self, dygiepp_gold, text, sentences, entities):
        return Abstract(dygiepp=dygiepp_gold, text=text,
                sentences=sentences, entities=entities)

    ############################### Tests ################################

    def test_parse_pred_dict_gold(self, dygiepp_gold, abst_gold, nlp):
        """
        Test for when the entity key is "ner"
        """
        abst = Abstract.parse_pred_dict(dygiepp_gold, nlp)

        assert abst == abst_gold

    def test_parse_pred_dict_pred(self, dygiepp_pred, abst_pred, nlp):
        """
        Test for when the entity key is "predicted_ner", allowing there
        to also be a key for "ner" that we ignore
        """
        abst = Abstract.parse_pred_dict(dygiepp_pred, nlp)

        assert abst == abst_pred

    def test_set_cand_sents(self, abst_set_cand_sents, cand_sents):
        """
        Make sure that the correct sentence indices are returned
        """
        abst_set_cand_sents.set_cand_sents()

        assert abst_set_cand_sents.cand_sents == cand_sents

    # Note that there's no test for the other method,
    # set_const_parse_and_spacy_doc; this is because to set up
    # the test I would just write the same code as is in the function, which
    # feels purpose defeating

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
    def abstract(self, dygiepp, nlp):
        return Abstract.parse_pred_dict(dygiepp, nlp)

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
    def candidate_phrase_one_word(self):
        return 'upregulates'

    @pytest.fixture
    def candidate_phrase_multi_word(self):
        return 'is found in'

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
        phrase = Abstract.walk_VP('', top_VP_one_word)

        assert phrase == candidate_phrase_one_word

    def test_walk_VP_multi_word(self, top_VP_multi_word,
            candidate_phrase_multi_word):
        phrase = Abstract.walk_VP('', top_VP_multi_word)
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
        label = Abstract.compute_label(label_df, embed_gets_label)
        assert label == correct_label

    def test_compute_label_no_label(self, label_df, embed_no_label):
        label = Abstract.compute_label(label_df, embed_no_label)

        assert label == ''

    def test_choose_ents(self, abstract, candidate_phrase_one_word):
        """
        This test is currently a little wonky, because of the random
        selection of entities I can't just test for the entities that I
        expect to see. Instead, I'm just testing for the presence of two
        non-identical entities in the relation. Same for the test for
        format_rels, this will need to be updated when the function is
        made more sophisticated.
        """
        sent_idx = 1
        ents = abstract.choose_ents(candidate_phrase_one_word, sent_idx)

        assert len(ents) == 2
        assert ents[0] != ents[1]

    def test_format_rels(self, abstract, phrase_labels):
        rels = abstract.format_rels(phrase_labels)

        assert rels[0] == []
        assert (rels[1][0][0], rels[1][0][1]) != (rels[1][0][2], rels[1][0][3])
        assert (rels[2][0][0], rels[2][0][1]) != (rels[2][0][2], rels[2][0][3])

class TestCategoryTracking:
    """
    Test the functions that keep track of sentence structures for
    categorization.
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
        mylabs = Abstract.parse_by_level(parse_string)

        assert mylabs == sent1_cat_dict

    def test_parse_by_level_sent2(self, nlp, sent2, sent2_cat_dict):

        parse_string = list(nlp(sent2).sents)[0]._.parse_string
        mylabs = Abstract.parse_by_level(parse_string)

        assert mylabs == sent2_cat_dict

    def test_parse_by_level_sent3(self, nlp, sent3, sent3_cat_dict):

        parse_string = list(nlp(sent3).sents)[0]._.parse_string
        mylabs = Abstract.parse_by_level(parse_string)

        assert mylabs == sent3_cat_dict
