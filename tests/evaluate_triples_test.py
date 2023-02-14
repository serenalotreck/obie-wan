"""
Spot checks for evaluate_triples.py

Author: Serena G. Lotreck
"""
import pytest
import sys
sys.path.append('../evaluation/')
import evaluate_triples as et
import spacy

@pytest.fixture
def no_punc_trips():
    return {'doc1':[("my name", "was", "sparty"),
            ("I", "am", "big")]}

@pytest.fixture
def punc_trips():
    return {'doc2':[("hello, my name", "is", "sparty"),
            ("I", "work", "East Lansing, Michigan"),
            ("I", "am", "big and green")]}

@pytest.fixture
def no_punc_gold_trips():
    return {'doc1':[("my name", "is", "sparty"),
            ("I", "work at", "MSU")]}

@pytest.fixture
def punc_gold_trips():
    return {'doc2':[("hello , my name", "is", "sparty"),
            ("I", "work in", "East Lansing , Michigan")]}

@pytest.fixture
def nlp():
    return spacy.load("en_core_sci_sm")

@pytest.fixture
def no_punc_f1_input_no_rel_lab():
    return (2, 2, 1)

@pytest.fixture
def punc_f1_input_no_rel_lab():
    return (3, 2, 2)

@pytest.fixture
def no_punc_f1_input_rel_lab():
    return (2, 2, 0)

@pytest.fixture
def punc_f1_input_rel_lab():
    return (3, 2, 1)

def test_get_f1_input_no_punc(no_punc_trips, no_punc_gold_trips, nlp,
        no_punc_f1_input_no_rel_lab):

    f1_input = et.get_f1_input(no_punc_trips, no_punc_gold_trips, nlp=nlp)

    assert f1_input == no_punc_f1_input_no_rel_lab

def test_get_f1_input_punc_no_rel_lab(punc_trips, punc_gold_trips, nlp,
        punc_f1_input_no_rel_lab):

    f1_input = et.get_f1_input(punc_trips, punc_gold_trips, nlp=nlp)

    assert f1_input == punc_f1_input_no_rel_lab

def test_get_f1_input_no_punc_rel_lab(no_punc_trips, no_punc_gold_trips, nlp,
        no_punc_f1_input_rel_lab):

    f1_input = et.get_f1_input(no_punc_trips, no_punc_gold_trips, nlp=nlp,
            check_rel_labels=True)

    assert f1_input == no_punc_f1_input_rel_lab

def test_get_f1_input_punc_rel_lab(punc_trips, punc_gold_trips, nlp,
        punc_f1_input_rel_lab):

    f1_input = et.get_f1_input(punc_trips, punc_gold_trips, nlp=nlp,
            check_rel_labels=True)

    assert f1_input == punc_f1_input_rel_lab
