"""
Tests for the output-formatting portion of gpt3_openai.py.

Author: Serena G. Lotreck
"""
import pytest
import spacy
import sys
sys.path.append('../models/')
from gpt3_openai import format_dygiepp_doc, find_sent


@pytest.fixture
def nlp():
    """
    Note: Using en_core_web_sm instead of en_core_sci_sm because the scispacy
    model fails to correctly split sentences in these examples.
    """
    return spacy.load("en_core_web_sm")

@pytest.fixture
def one_sent_text():
    return 'Hello world, I am Sparty.'

@pytest.fixture
def multi_sent_text():
    return ('Hello world, I am Sparty. I work at MSU, and I like football. '
            'I live in East Lansing, Michigan.')

@pytest.fixture
def one_sent_triples():
    return [('I', 'am', 'Sparty')]

@pytest.fixture
def multi_sent_triples():
    return [
            ('I', 'am', 'Sparty'),
            ('I', 'work at', 'MSU'),
            ('I', 'like', 'football'),
            ('I', 'live in', 'East Lansing, Michigan')
            ]

@pytest.fixture
def one_sent_dygiepp():
    return {
            'doc_key': 'doc1',
            'dataset': 'scierc',
            'sentences': [['Hello', 'world', ',', 'I', 'am', 'Sparty', '.']],
            'ner':[[[3, 3, 'ENTITY'], [5, 5, 'ENTITY']]],
            'relations': [[[3, 3, 5, 5, 'am']]]
            }

@pytest.fixture
def multi_sent_dygiepp():
    return {
            'doc_key': 'doc2',
            'dataset': 'scierc',
            'sentences': [['Hello', 'world', ',', 'I', 'am', 'Sparty', '.'],
                ['I', 'work', 'at', 'MSU', ',', 'and', 'I', 'like', 'football',
                    '.'],
                ['I', 'live', 'in', 'East', 'Lansing', ',', 'Michigan', '.']],
            'ner':[[[3, 3, 'ENTITY'], [5, 5, 'ENTITY']],
                [[7, 7, 'ENTITY'], [10, 10, 'ENTITY'], [13, 13, 'ENTITY'],
                    [15, 15, 'ENTITY']],
                [[17, 17, 'ENTITY'], [20, 23, 'ENTITY']]],
            'relations': [[[3, 3, 5, 5, 'am']],
                [[7, 7, 10, 10, 'work at'], [13, 13, 15, 15, 'like']],
                [[17, 17, 20, 23, 'live in']]]
            }

@pytest.fixture
def first_trip_tok():
    return [['I'], ['am'], ['Sparty']]

@pytest.fixture
def last_trip_tok():
    return [['I'], ['live', 'in'], ['East', 'Lansing', ',', 'Michigan']]

@pytest.fixture
def first_sent_idx():
    return 0

@pytest.fixture
def last_sent_idx():
    return 2


def test_format_dygiepp_doc_one_sent(nlp, one_sent_text, one_sent_triples,
        one_sent_dygiepp):

    dygiepp_doc = format_dygiepp_doc('doc1', one_sent_text, one_sent_triples,
            nlp)

    assert dygiepp_doc == one_sent_dygiepp


def test_format_dygiepp_doc_multi_sent(nlp, multi_sent_text, multi_sent_triples,
        multi_sent_dygiepp):

    dygiepp_doc = format_dygiepp_doc('doc2', multi_sent_text, multi_sent_triples,
            nlp)

    assert dygiepp_doc == multi_sent_dygiepp


def test_find_sent_first(multi_sent_dygiepp, first_trip_tok,
        first_sent_idx):

    sent_idx = find_sent(multi_sent_dygiepp, first_trip_tok)

    assert sent_idx == first_sent_idx


def test_find_sent_last(multi_sent_dygiepp, last_trip_tok,
        last_sent_idx):

    sent_idx = find_sent(multi_sent_dygiepp, last_trip_tok)

    assert sent_idx == last_sent_idx
