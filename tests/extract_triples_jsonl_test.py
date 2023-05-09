"""
Spot checks for extract_triples.py.

Author: Serena G. Lotreck
"""
import pytest
import sys
sys.path.append('../data_scripts')
from extract_triples_jsonl import get_doc_triples


@pytest.fixture
def one_sent_triples():
    return [('I', 'am', 'Sparty')]

@pytest.fixture
def multi_sent_triples():
    return [
            ('I', 'am', 'Sparty'),
            ('I', 'work at', 'MSU'),
            ('I', 'like', 'football'),
            ('I', 'live in', 'East Lansing , Michigan')
            ] # Note that joining with spaces may cause problems evaluating
              # entities with commas in them, as GPT3 or other approaches may
              # not include these spaces. Will have to account for this in the
              # evaluation paradigm

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

def test_get_doc_triples_one_sent(one_sent_dygiepp, one_sent_triples):

    trips = get_doc_triples(one_sent_dygiepp)

    assert trips == one_sent_triples

def test_get_doc_triples_multi_sent(multi_sent_dygiepp, multi_sent_triples):

    trips = get_doc_triples(multi_sent_dygiepp)

    assert trips == multi_sent_triples
