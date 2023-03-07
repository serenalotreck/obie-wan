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


class TestPickPhrase:
    """
    Tests pick_phrase method.
    """

    ############################### Fixtures ################################

    @pytest.fixture
    def nlp(self):
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        return nlp

    @pytest.fixture
    def spacy_doc(self, nlp):
        text = ('Indicated that ABA signaling positively regulates root '
                'defense to R. solanacearum. We conclude that plants '
                'prioritize resistance of reproductive tissues over '
                'vegetative tissues, and that a chewing herbivore species '
                'is the main driver of responses in flowering B. nigra.'
                'Flowering Brassica nigra were exposed to either a chewing '
                'caterpillar, a phloem-feeding aphid or a bacterial pathogen, '
                'and plant hormonal responses were compared with dual attack '
                'situations. Dual attack increased plant resistance to '
                'caterpillars, but compromised resistance to aphids. However, '
                'when SA was applied more than 30 h prior to the onset of the '
                'JA response, the suppressive effect of SA was completely '
                'absent.')
        return nlp(text)

    @pytest.fixture
    def abstract(self, spacy_doc):
        return Abstract(spacy_doc=spacy_doc)

    @pytest.fixture
    def simplest_idx(self):
        return 0

    @pytest.fixture
    def simplest_phrases(self, simplest_idx, spacy_doc):
        phrase1 = list(spacy_doc.sents)[simplest_idx][0]
        return [phrase1]

    @pytest.fixture
    def sbar_idx(self):
        return 1

    @pytest.fixture
    def sbar_phrases(self, sbar_idx, spacy_doc):
        phrase1 = list(spacy_doc.sents)[sbar_idx][16]
        phrase2 = list(spacy_doc.sents)[sbar_idx][31]
        return [phrase1, phrase2]

    @pytest.fixture
    def sibling_s_idx(self):
        return 2

    @pytest.fixture
    def sibling_s_phrases(self, sibling_s_idx, spacy_doc):
        phrase1 = list(spacy_doc.sents)[sibling_s_idx][45:47]
        phrase2 = list(spacy_doc.sents)[sibling_s_idx][65:67]
        return [phrase1, phrase2]

    @pytest.fixture
    def vp_cc_vp_idx(self):
        return 3

    @pytest.fixture
    def vp_cc_vp_phrases(self, nlp):
        return 'NO PHRASE: Multiple levels with kids'

    @pytest.fixture
    def noncontinuous_idx(self):
        return 4

    ############################### Tests ################################

    def test_simplest(self, abstract, simplest_idx, simplest_phrases):

        phrases = abstract.pick_phrase(simplest_idx)

        assert phrases == simplest_phrases

    def test_sbar(self, abstract, sbar_idx, sbar_phrases):

        phrases = abstract.pick_phrase(sbar_idx)

        assert phrases == sbar_phrases

    def test_sibling_s(self, abstract, sibling_s_idx, sibling_s_phrases):

        phrases = abstract.pick_phrase(sibling_s_idx)

        assert phrases == sibling_s_phrases

    def test_vp_cc_vp(self, abstract, vp_cc_vp_idx, vp_cc_vp_phrases):

        phrases = abstract.pick_phrase(vp_cc_vp_idx)

        assert phrases == vp_cc_vp_phrases

    def test_noncontinuous(self, abstract, noncontinuous_idx):

        with pytest.raises(Exception) as exc_info:
            phrases = abstract.pick_phrases(noncontinuous_idx)

        assert exc_info.value.args[0] == 'Noncontinuous span'


class TestChooseEntsFormatRels:
    """
    Tests for choose_ents and format_rels methods.
    """

    ############################### Fixtures ################################
    @pytest.fixtures
    def nlp(self):
        nlp = spacy.load("en_core_sci_sm")
        nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
        return nlp

    @pytest.fixture
    def fulldoc_spacy(self, nlp):
        text = ('Hello world my name is Sparty. Jasmonic acid and gibberellic '
                'acid are both found in Arabidopsis thaliana. Flowers tend to '
                'enjoy jasmonic acid and gibberellic acid. Once again my name '
                'is Sparty.')
        doc = nlp(text)
        return doc

    @pytest.fixture
    def dygiepp(self):
        return {
                'doc_key': 'doc1',
                'sentences': [['Hello', 'world', 'my', 'name', 'is', 'Sparty',
                    '.'],
                    ['Jasmonic', 'acid', 'and', 'gibberellic', 'acid',
                    'are', 'both', 'found', 'in', 'Arabidopsis', 'thaliana',
                    '.'],
                    ['Flowers', 'tend', 'to', 'enjoy', 'jasmonic',
                    'acid', 'and', 'gibberellic', 'acid', '.'],
                    ['Once', 'again', 'my', 'name', 'is', 'Sparty', '.']],
                'entities': [[[2, 3, 'ENTITY'], [5, 5, 'ENTITY']],
                    [[7, 8, 'ENTITY'], [10, 11, 'ENTITY'], [16, 17, 'ENTITY']],
                    [[23, 24, 'ENTITY'], [26, 27, 'ENTITY']],
                    [[31, 32, 'ENTITY'], [34, 34, 'ENTITY']]]
                }

    @pytest.fixture
    def abstract(self, dygiepp, nlp):
        abst = Abstract.parse_pred_dict(dygiepp, nlp)
        return abst

    @pytest.fixture
    def one_each_side_phrase(self, fulldoc_spacy):
        start_idx = 4
        end_idx = 5
        return fulldoc_spacy[start_idx:end_idx]

    @pytest.fixture
    def mult_both_sides_phrase(self, fulldoc_spacy):
        start_idx = 12
        end_idx = 16
        return fulldoc_spacy[start_idx:end_idx]

     @pytest.fixture
     def mult_one_side_phrase(self, fulldoc_spacy):
         start_idx = 20
         end_idx = 23
         return fulldoc_spacy[start_idx:end_idx]

     @pytest.fixture
     def overlapping_phrase(self, fulldoc_spacy):
         start_idx = 32
         end_idx = 34
         return fulldoc_spacy[start_idc:end_idx]

     @pytest.fixture
     def one_each_side_answer(self, dygiepp):
         ents = dygiepp['entities'][0]
         return ents

     @pytest.fixture
     def mult_both_sides_answer(self, dygiepp):
         ents = sygiepp['entities'][1][1:]
         return ents

     @pytest.fixture
     def mult_one_side_answer(self):
        return []

    @pytest.fixture
    def overlapping_answer(self):
        return []

    @pytest.fixture
    def phrase_labels(self, one_each_side_phrase, mult_both_sides_phrase,
            mult_one_side_phrase, overlapping_phrase):
        return {
                0: [(one_each_side_phrase, 'activates')],
                1: [(mult_both_sides_phrase, 'deactivates')],
                2: [(mult_one_side_phrase, 'activates')],
                3: [(overlapping_phrase, 'deactivates')]
                }

    @pytest.fixture
    def formatted_rels(self, phrase_labels, one_each_side_answer,
            mult_both_sides_answer):
        sent0_rel = [one_each_side_answer[0][0],
                one_each_side_answer[0][1],
                one_each_side_answer[1][0],
                one_each_side_answer[1][1],
                'activates']
        sent1_rel = [mult_both_sides_answer[0][0],
                mult_both_sides_answer[0][1],
                mult_both_sides_answer[1][0],
                mult_both_sides_answer[1][1],
                'deactivates']
        return [
                [sent0_rel],
                [sent1_rel],
                [],
                []
                ]

    ############################### Tests ################################

    def test_choose_ents_one_each_side(self, abst, one_each_side_phrase,
            one_each_side_answer):

        ents = abst.choose_ents(one_side_each_phrase, 0)

        assert ents == one_each_side_answer

    def test_choose_ents_mult_both_sides(self, abst, mult_both_sides_phrase,
            mult_both_sides_answer):

        ents = abst.choose_ents(mult_both_sides_phrase, 1)

        assert ents == mult_both_sides_answer

    def test_choose_ents_mult_one_side(self, abst, mult_one_side_phrase,
            mult_one_side_answer):

        ents = abst.choose_ents(mult_one_side_phrase, 2)

        assert ents == mult_one_side_answer

    def test_choose_ents_overlapping(self, abst, overlapping_phrase, overlapping_answer):

        ents = abst.choose_ents(overlapping_phrase, 3)

        assert ents == overlapping_answer


    def test_format_rels(self, abst, phrase_labels, formatted_rels):

        rels = abstract.format_rels(phrase_labels)

        assert rels == formatted_rels

