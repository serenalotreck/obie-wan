"""
Spot checks for functions in bert_embeddings.py

Author: Serena G. Lotreck
"""
import pytest
import sys
sys.path.append('../models/distant_supervision_re/')
import bert_embeddings as be

@pytest.fixture
def full_list():
    return ['Hello', 'my', 'name', 'is', 'sparty', '.']

@pytest.fixture
def sub_list_ordered():
    return ['name', 'is']

@pytest.fixture
def sub_list_unordered():
    return ['is', 'name']

@pytest.fixture
def sub_list_not_present():
    return ['East', 'Lansing']


def test_find_sublist_ordered(full_list, sub_list_ordered):
    start, end = be.find_sub_list(sub_list_ordered, full_list)

    assert start == 2
    assert end == 3


def test_find_sublist_unordered(full_list, sub_list_unordered):
    start, end = be.find_sub_list(sub_list_unordered, full_list)

    assert start == None
    assert end == None


def test_find_sublist_not_present(full_list, sub_list_not_present):
    start, end = be.find_sub_list(sub_list_not_present, full_list)

    assert start == None
    assert end == None
