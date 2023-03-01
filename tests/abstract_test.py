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


class TestTripleChoice:
    """
    Tests for the functions that choose phrases and entities within in the
    abstract class.
    """

    ############################### Fixtures ################################

    @pytest.fixture

    ############################### Tests ################################

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

