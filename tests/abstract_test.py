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
        yield nlp

    @pytest.fixture
    def dygiepp_gold(self):
        yield {'doc_key':'doc1',
                'dataset':'scierc',
                'sentences':[['Hello', 'world', '.'],
                    ['My', 'name', 'is', 'Sparty', 'and', 'I',
                        'work', 'at', 'MSU', '.']],
                'ner':[[],[[6, 6, 'Person'], [11, 11, 'Organization']]]}

    @pytest.fixture
    def dygiepp_pred(self):
        import inspect
        signature = inspect.signature(Abstract)
        print(f'signature in the dygiepp_pred fixture: {signature}')
        yield {'doc_key':'doc1',
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
        yield 'Hello world . My name is Sparty and I work at MSU .'

    @pytest.fixture
    def sentences(self, dygiepp_gold):
        yield dygiepp_gold["sentences"]

    @pytest.fixture
    def entities(self, dygiepp_gold):
        yield dygiepp_gold["ner"]

    @pytest.fixture
    def candidate_sents(self):
        yield [1]

    @pytest.fixture
    def const_parse_str(self):
        yield ['(NP (UH Hello) (NN world) (. .))',
                '(S (S (NP (PRP$ My) (NN name)) (VP (VBZ is) (NP (NNP '
                'Sparty)))) (CC and) (S (NP (PRP I)) (VP (VBP work) (PP '
                '(IN at) (NP (NNP MSU))))) (. .))']

    @pytest.fixture
    def spacy_doc(self, nlp, text):
        yield nlp(text)

    @pytest.fixture
    def abst_gold(self, dygiepp_gold, text, sentences, entities,
            candidate_sents, const_parse_str, spacy_doc):
        yield Abstract(dygiepp_gold, text, sentences, entities,
                candidate_sents, const_parse_str, spacy_doc)

    @pytest.fixture
    def abst_pred(self, dygiepp_pred, text, sentences, entities,
            candidate_sents, const_parse_str, spacy_doc):
        print('inside pred fixture, heres whats passed as candidate sents:')
        print(candidate_sents)
        yield Abstract(dygiepp_pred, text, sentences, entities,
                candidate_sents, const_parse_str, spacy_doc)

    @pytest.fixture
    def abst_set_cand_sents(self, dygiepp_gold, text, sentences, entities):
        import inspect
        signature = inspect.signature(Abstract.__init__)
        print('Abstract init default args:', signature)
        within_fix = Abstract(dygiepp=dygiepp_gold, text=text,
                sentences=sentences, entities=entities)
        print('Within the fixture, here is the candidate sent attribute:')
        print(within_fix.candidate_sents)
        yield within_fix

    ############################### Tests ################################

    def test_parse_pred_dict_gold(self, dygiepp_gold, abst_gold):
        """
        Test for when the entity key is "ner"
        """
        import inspect
        signature = inspect.signature(Abstract.__init__)
        print('Abstract init default args:', signature)
        abst = Abstract.parse_pred_dict(dygiepp_gold)

        assert abst == abst_gold

    def test_parse_pred_dict_pred(self, dygiepp_pred, abst_pred):
        """
        Test for when the entity key is "predicted_ner", allowing there
        to also be a key for "ner" that we ignore
        """
        import inspect
        signature = inspect.signature(Abstract.__init__)
        print('Abstract init default args:', signature)
        abst = Abstract.parse_pred_dict(dygiepp_pred)

        assert abst == abst_pred

    def test_set_candidate_sents(self, abst_set_cand_sents, candidate_sents):
        """
        Make sure that the correct sentence indices are returned
        """
        print('STARTING TEST for candidate sentence setting')
        print('Candidate sentence attribute before doing anything:')
        print(abst_set_cand_sents.candidate_sents)
        abst_set_cand_sents.set_candidate_sents()
        print('And after setting them:')
        print(abst_set_cand_sents.candidate_sents)

        assert abst_set_cand_sents.candidate_sents == candidate_sents

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
        yield {'doc_key': 'doc2',
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
        yield nlp

    @pytest.fixture
    def abstract(self, dygiepp):
        yield Abstract.parse_pred_dict(dygiepp)

    @pytest.fixture
    def top_VP_one_word(self, nlp):
        text = 'upregulates Protein 1'
        doc = nlp(text)
        sent = list(doc.sents)[0]
        yield sent

    @pytest.fixture
    def top_VP_multi_word(self, nlp):
        text = 'is found in Arabidopsis thaliana'
        doc = nlp(text)
        sent = list(doc.sents)[0]
        yield sent

    @pytest.fixture
    def candidate_phrase_one_word(self):
        yield 'upregulates'

    @pytest.fixture
    def candidate_phrase_multi_word(self):
        yield 'is found in'

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

        yield pd.DataFrame.from_dict(label_dict, orient='index')

    @pytest.fixture
    def embed_gets_label(self):
        yield [0.3, 0.1]

    @pytest.fixture
    def embed_no_label(self):
        # I'm not sure this is a legitimate scenario, but I wanted to
        # test the scenario where the similarity is less than 0.5
        yield [-0.1, -0.5]

    @pytest.fixture
    def correct_label(self):
        yield 'activates'

    @pytest.fixture
    def phrase_labels(self):
        yield {1:{'upregulates':'activates'}, 2:{'is found in':'is_in'}}

    ############################### Tests ################################

    def test_walk_VP_one_word(self, top_VP_one_word,
            candidate_phrase_one_word):
        phrase = Abstract.walk_VP('', top_VP_one_word)

        assert phrase == candidate_phrase_one_word

    def test_walk_VP_multi_word(self, top_VP_multi_word,
            candidate_phrase_multi_word):
        print(f'inside test, top_VP_multi_word is {top_VP_multi_word}')
        phrase = Abstract.walk_VP('', top_VP_multi_word)
        print(f'phrase inside test: {phrase}')
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

