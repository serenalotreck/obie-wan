"""
Tests for the Abstract class.

Author: Serena G. Lotreck
"""
import sys
sys.path.append('../distant_supervision_re')
from abstract import Abstract

################################ Fixtures #####################################

# Fixtures for testing the initial setup of an Abstract class instance

@pytest.fixture
def empty_abstract():
    return Abstract()

@pytest.fixture
def dygiepp_gold():
    return {'doc_key':'doc1',
            'dataset':'scierc',
            'sentences':[['Hello', 'world', '!'],
                ['My', 'name', 'is', 'Sparty']],
            'ner':[[],[[6, 6, 'Person']]]}

@pytest.fixture
def dygiepp_pred():
    return {'doc_key':'doc1',
            'dataset':'scierc',
            'sentences':[['Hello', 'world', '!'],
                ['My', 'name', 'is', 'Sparty']],
            'ner':[[],[]] # PURE requires this field even if there is no gold
            'predicted_ner':[[],[[6, 6, 'Person']]]}


################################# Tests #######################################

# Test for setting up an abstract class instance

def test_
