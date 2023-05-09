"""
Spot checks for extract_triples_brat.py

Author: Serena G. Lotreck
"""
import pytest
import sys
sys.path.append('../data_scripts')
from extract_triples_brat import get_brat_trips


@pytest.fixture
def nondisjoint_brat():
    return ['T1\tGene/protein/RNA 27 36\tverprolin',
            'T2\tIndividual_protein 162 166\tArp2',
            'T3\tIndividual_protein 195 200\tactin',
            'T4\tGene/protein/RNA 58 65\tcofilin',
            'R1\tPPI Arg1:T2 Arg2:T3']

@pytest.fixture
def nondisjoint_trips():
    return [['Arp2', 'PPI', 'actin']]

@pytest.fixture
def disjoint_brat():
    return ['T1\tGene/protein/RNA 27 36\tverprolin',
            'T2\tIndividual_protein 162 166\tArp2',
            'T3\tIndividual_protein 195 200\tactin',
            'T4\tGene/protein/RNA 58 65\tcofilin',
            'T5\tIndividual_protein 162 165;167 168\tArp 3',
            'R1\tPPI Arg1:T2 Arg2:T3',
            'R2\tPPI Arg1:T2 Arg2:T5',
            'R3\tPPI Arg1:T3 Arg2:T5']

@pytest.fixture
def disjoint_trips():
    return [['Arp2', 'PPI', 'actin'], ['Arp2', 'PPI', 'Arp 3'],
            ['actin', 'PPI', 'Arp 3']]

def test_get_brat_trips_nondisjoint(nondisjoint_brat, nondisjoint_trips):
    
    trips = get_brat_trips(nondisjoint_brat)
    
    assert trips == nondisjoint_trips

def test_get_brat_trips_disjoint(disjoint_brat, disjoint_trips):
    
    trips = get_brat_trips(disjoint_brat)
    
    assert trips == disjoint_trips