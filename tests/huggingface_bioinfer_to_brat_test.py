"""
Spot checks for converting bioinfer huggingface format to brat.

Author: Serena G. Lotreck
"""
import pytest
import sys
sys.path.append('../data_scripts')
import huggingface_bioinfer_to_brat as hbb


@pytest.fixture
def rep_ids_ent_input():
    return {'origid1': 'T1\tProtein 0 3\tFLO1',
            'origid2': 'T2\tDNA 7 9\tpro'}

@pytest.fixture
def rep_ids_rel_input():
    return {'relorig1': 'R1\tProduces Arg1:T2 Arg2:T1'}

@pytest.fixture
def rep_ids_output():
    return ('T1\tProtein 0 3\tFLO1\nT2\tDNA 7 9\tpro\n'
            'R1\tProduces Arg1:T2 Arg2:T1')

def test_replace_ids(rep_ids_ent_input, rep_ids_rel_input, rep_ids_output):

    anns_to_save = hbb.replace_ids(rep_ids_ent_input, rep_ids_rel_input)

    assert anns_to_save == rep_ids_output


@pytest.fixture
def joined_txt_input_one_sent():
    return [{'document_id': 'doc1', 'text': 'This is a sentence.'}]

@pytest.fixture
def joined_txt_output_one_sent():
    return 'This is a sentence.'

def test_get_joined_text_one_sent(joined_txt_input_one_sent,
        joined_txt_output_one_sent):

    text = hbb.get_joined_text(joined_txt_input_one_sent)

    assert text == joined_txt_output_one_sent


@pytest.fixture
def joined_txt_input_mult_sent():
    return [{'document_id': 'doc1', 'text': 'This is a sentence.'},
            {'document_id': 'doc2', 'text': 'This is another sentence.'},
            {'document_id': 'doc3', 'text': 'And this is another.'}]

@pytest.fixture
def joined_txt_output_mult_sent():
    return 'This is a sentence. This is another sentence. And this is another.'

def test_get_joined_text_mult_sent(joined_txt_input_mult_sent,
        joined_txt_output_mult_sent):

    text = hbb.get_joined_text(joined_txt_input_mult_sent)

    assert text == joined_txt_output_mult_sent


@pytest.fixture
def ent_anns_one_doc_input():
    return [{'document_id': 'BioInfer.d0.s0', 'type': 'Sentence',
            'text': 'alpha-catenin inhibits beta-catenin signaling by '
            'preventing formation of a beta-catenin*T-cell factor*DNA '
            'complex.',
            'entities':[{'id': 'BioInfer.d0.s0.e0', 'offsets': [[88, 101]],
                'text': ['T-cell factor'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e1', 'offsets': [[0, 13]],
                'text': ['alpha-catenin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e2', 'offsets': [[23, 35]],
                'text': ['beta-catenin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e3', 'offsets': [[75, 87]],
                'text': ['beta-catenin'], 'type': 'Individual_protein',
                'normalized': []}]}]

@pytest.fixture
def ent_anns_one_doc_output():
    return {
            'BioInfer.d0.s0.e0': 'T1\tIndividual_protein 88 101\tT-cell factor',
            'BioInfer.d0.s0.e1': 'T2\tIndividual_protein 0 13\talpha-catenin',
            'BioInfer.d0.s0.e2': 'T3\tIndividual_protein 23 35\tbeta-catenin',
            'BioInfer.d0.s0.e3': 'T4\tIndividual_protein 75 87\tbeta-catenin'
            }

def test_get_ent_anns_one_doc(ent_anns_one_doc_input, ent_anns_one_doc_output):

    ent_anns = hbb.get_ent_anns(ent_anns_one_doc_input)

    assert ent_anns == ent_anns_one_doc_output


@pytest.fixture
def ent_anns_mult_doc_input():
    return [{'document_id': 'BioInfer.d0.s0', 'type': 'Sentence',
            'text': 'alpha-catenin inhibits beta-catenin signaling by '
            'preventing formation of a beta-catenin*T-cell factor*DNA '
            'complex.',
            'entities': [{'id': 'BioInfer.d0.s0.e0', 'offsets': [[88, 101]],
                'text': ['T-cell factor'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e1', 'offsets': [[0, 13]],
                'text': ['alpha-catenin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e2', 'offsets': [[23, 35]],
                'text': ['beta-catenin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e3', 'offsets': [[75, 87]],
                'text': ['beta-catenin'], 'type': 'Individual_protein',
                'normalized': []}]},
            # Note that this isn't actually in the same doc, I changed the IDs
            {'document_id': 'BioInfer.d0.s1', 'type': 'Sentence',
            'text': 'A binary complex of birch profilin and skeletal muscle '
            'actin could be isolated by gel chromatography.',
            'entities': [{'id': 'BioInfer.d0.s1.e0', 'offsets': [[26, 34]],
                'text': ['profilin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s1.e1', 'offsets': [[39, 60]],
                'text':['skeletal muscle actin'], 'type': 'Individual_protein',
                'normalized': []}]}]

@pytest.fixture
def ent_anns_mult_doc_output():
    return {
            'BioInfer.d0.s0.e0': 'T1\tIndividual_protein 88 101\tT-cell factor',
            'BioInfer.d0.s0.e1': 'T2\tIndividual_protein 0 13\talpha-catenin',
            'BioInfer.d0.s0.e2': 'T3\tIndividual_protein 23 35\tbeta-catenin',
            'BioInfer.d0.s0.e3': 'T4\tIndividual_protein 75 87\tbeta-catenin',
            'BioInfer.d0.s1.e0': 'T5\tIndividual_protein 141 149\tprofilin',
            'BioInfer.d0.s1.e1': 'T6\tIndividual_protein 154 175\tskeletal muscle actin'
            }

def test_get_ent_anns_mult_doc(ent_anns_mult_doc_input,
        ent_anns_mult_doc_output):

    ent_anns = hbb.get_ent_anns(ent_anns_mult_doc_input)

    assert ent_anns == ent_anns_mult_doc_output


@pytest.fixture
def ent_anns_disjoint_input():
    return [{'document_id': 'BioInfer.d1.s1', 'type': 'Sentence',
        'text': 'Birch profilin increased the critical concentration required '
        'for muscle and brain actin polymerization in a '
        'concentration-dependent manner, supporting the notion of the '
        'formation of a heterologous complex between the plant protein and '
        'animal actin.',
        'entities': [
            {'id': 'BioInfer.d1.s1.e0', 'offsets': [[76, 87]],
                'text': ['brain actin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d1.s1.e1', 'offsets': [[6, 14]],
                'text': ['profilin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d1.s1.e2', 'offsets': [[65, 71], [82, 87]],
                'text': ['muscle', 'actin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d1.s1.e3', 'offsets': [[242, 247]],
                'text': ['actin'], 'type': 'Individual_protein',
                'normalized': []}],
        'relations': [
            {'id': 'BioInfer.d1.s1.i0', 'type': 'PPI',
                'arg1_id': 'BioInfer.d1.s1.e0', 'arg2_id': 'BioInfer.d1.s1.e1',
                'normalized': []},
            {'id': 'BioInfer.d1.s1.i1', 'type': 'PPI',
                'arg1_id': 'BioInfer.d1.s1.e1', 'arg2_id': 'BioInfer.d1.s1.e2',
                'normalized': []},
            {'id': 'BioInfer.d1.s1.i2', 'type': 'PPI',
                'arg1_id': 'BioInfer.d1.s1.e1', 'arg2_id': 'BioInfer.d1.s1.e3',
                'normalized': []}]}]

@pytest.fixture
def ent_anns_disjoint_output():
    return {
            'BioInfer.d1.s1.e0': 'T1\tIndividual_protein 76 87\tbrain actin',
            'BioInfer.d1.s1.e1': 'T2\tIndividual_protein 6 14\tprofilin',
            'BioInfer.d1.s1.e2':
                'T3\tIndividual_protein 65 71;82 87\tmuscle actin',
            'BioInfer.d1.s1.e3': 'T4\tIndividual_protein 242 247\tactin'
            }

def test_get_ent_anns_disjoint(ent_anns_disjoint_input,
        ent_anns_disjoint_output):

    ent_anns = hbb.get_ent_anns(ent_anns_disjoint_input)

    assert ent_anns == ent_anns_disjoint_output


@pytest.fixture
def rel_anns_one_doc_input():
    return [{'document_id': 'BioInfer.d0.s0',
            'relations': [{'id': 'BioInfer.d0.s0.i0', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e1',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i1', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e2',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i2', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i3', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e1', 'arg2_id': 'BioInfer.d0.s0.e2',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i4', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e1', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i5', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e2', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []}]}]

@pytest.fixture
def rel_anns_one_doc_output():
    return {
            'BioInfer.d0.s0.i0': 'R1\tPPI Arg1:T1 Arg2:T2',
            'BioInfer.d0.s0.i1': 'R2\tPPI Arg1:T1 Arg2:T3',
            'BioInfer.d0.s0.i2': 'R3\tPPI Arg1:T1 Arg2:T4',
            'BioInfer.d0.s0.i3': 'R4\tPPI Arg1:T2 Arg2:T3',
            'BioInfer.d0.s0.i4': 'R5\tPPI Arg1:T2 Arg2:T4',
            'BioInfer.d0.s0.i5': 'R6\tPPI Arg1:T3 Arg2:T4'
            }

def test_get_rel_anns_one_doc(ent_anns_one_doc_output, rel_anns_one_doc_input,
        rel_anns_one_doc_output):

    rel_anns = hbb.get_rel_anns(rel_anns_one_doc_input,
            ent_anns_one_doc_output)

    assert rel_anns == rel_anns_one_doc_output


@pytest.fixture
def rel_anns_mult_doc_input():
    return [{'document_id': 'BioInfer.d0.s0',
            'relations': [{'id': 'BioInfer.d0.s0.i0', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e1',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i1', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e2',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i2', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i3', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e1', 'arg2_id': 'BioInfer.d0.s0.e2',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i4', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e1', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i5', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e2', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []}]},
            # Again note that I changed the IDs
            {'document_id': 'BioInfer.d0.s1',
            'relations': [{'id': 'BioInfer.d0.s1.i0', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s1.e0', 'arg2_id': 'BioInfer.d0.s1.e1',
                'normalized': []}]}]

@pytest.fixture
def rel_anns_mult_doc_output():
    return {
            'BioInfer.d0.s0.i0': 'R1\tPPI Arg1:T1 Arg2:T2',
            'BioInfer.d0.s0.i1': 'R2\tPPI Arg1:T1 Arg2:T3',
            'BioInfer.d0.s0.i2': 'R3\tPPI Arg1:T1 Arg2:T4',
            'BioInfer.d0.s0.i3': 'R4\tPPI Arg1:T2 Arg2:T3',
            'BioInfer.d0.s0.i4': 'R5\tPPI Arg1:T2 Arg2:T4',
            'BioInfer.d0.s0.i5': 'R6\tPPI Arg1:T3 Arg2:T4',
            'BioInfer.d0.s1.i0': 'R7\tPPI Arg1:T5 Arg2:T6'
            }

def test_get_rel_anns_mult_doc(ent_anns_mult_doc_output,
        rel_anns_mult_doc_input, rel_anns_mult_doc_output):

    rel_anns = hbb.get_rel_anns(rel_anns_mult_doc_input,
            ent_anns_mult_doc_output)

    assert rel_anns == rel_anns_mult_doc_output


@pytest.fixture
def join_docs_one_doc_input():
    return [{'document_id': 'BioInfer.d0.s0', 'type': 'Sentence',
        'text': 'alpha-catenin inhibits beta-catenin signaling by preventing '
        'formation of a beta-catenin*T-cell factor*DNA complex.',
        'entities': [
            {'id': 'BioInfer.d0.s0.e0', 'offsets': [[88, 101]],
                'text': ['T-cell factor'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e1', 'offsets': [[0, 13]],
                'text': ['alpha-catenin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e2', 'offsets': [[23, 35]],
                'text': ['beta-catenin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e3', 'offsets': [[75, 87]],
                'text': ['beta-catenin'], 'type': 'Individual_protein',
                'normalized': []}],
        'relations': [
            {'id': 'BioInfer.d0.s0.i0', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e1',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i1', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e2',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i2', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i3', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e1', 'arg2_id': 'BioInfer.d0.s0.e2',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i4', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e1', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i5', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e2', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []}]}]

@pytest.fixture
def join_docs_one_doc_output(join_docs_one_doc_input):
    joined_dset = {}
    new_doc_key = 'BioInfer.d0'
    joined_dset[new_doc_key] = join_docs_one_doc_input
    return joined_dset

def test_join_docs_one_doc(join_docs_one_doc_input, join_docs_one_doc_output):

    joined_dset = hbb.join_docs(join_docs_one_doc_input)

    assert joined_dset == join_docs_one_doc_output


@pytest.fixture
def join_docs_mult_doc_input():
    return [{'document_id': 'BioInfer.d0.s0', 'type': 'Sentence',
        'text': 'alpha-catenin inhibits beta-catenin signaling by preventing '
        'formation of a beta-catenin*T-cell factor*DNA complex.',
        'entities': [
            {'id': 'BioInfer.d0.s0.e0', 'offsets': [[88, 101]],
                'text': ['T-cell factor'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e1', 'offsets': [[0, 13]],
                'text': ['alpha-catenin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e2', 'offsets': [[23, 35]],
                'text': ['beta-catenin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e3', 'offsets': [[75, 87]],
                'text': ['beta-catenin'], 'type': 'Individual_protein',
                'normalized': []}],
        'relations': [
            {'id': 'BioInfer.d0.s0.i0', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e1',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i1', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e2',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i2', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i3', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e1', 'arg2_id': 'BioInfer.d0.s0.e2',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i4', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e1', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i5', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e2', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []}]},
            # Changed IDs
            {'document_id': 'BioInfer.d0.s1', 'type': 'Sentence',
            'text': 'A binary complex of birch profilin and skeletal muscle '
            'actin could be isolated by gel chromatography.',
            'entities': [
                {'id': 'BioInfer.d0.s1.e0', 'offsets': [[26, 34]],
                    'text': ['profilin'], 'type': 'Individual_protein',
                    'normalized': []},
                {'id': 'BioInfer.d0.s1.e1', 'offsets': [[39, 60]],
                    'text': ['skeletal muscle actin'], 'type': 'Individual_protein',
                    'normalized': []}],
            'relations': [
                {'id': 'BioInfer.d0.s1.i0', 'type': 'PPI',
                    'arg1_id': 'BioInfer.d0.s1.e0', 'arg2_id': 'BioInfer.d0.s1.e1',
                    'normalized': []}]}]

@pytest.fixture
def join_docs_mult_doc_output(join_docs_mult_doc_input):
    joined_dset = {}
    new_doc_key = 'BioInfer.d0'
    joined_dset[new_doc_key] = join_docs_mult_doc_input
    return joined_dset

def test_join_docs_mult_doc(join_docs_mult_doc_input,
        join_docs_mult_doc_output):

    joined_dset = hbb.join_docs(join_docs_mult_doc_input)

    assert joined_dset == join_docs_mult_doc_output


@pytest.fixture
def join_docs_unique_input():
    return [{'document_id': 'BioInfer.d0.s0', 'type': 'Sentence',
        'text': 'alpha-catenin inhibits beta-catenin signaling by preventing '
        'formation of a beta-catenin*T-cell factor*DNA complex.',
        'entities': [
            {'id': 'BioInfer.d0.s0.e0', 'offsets': [[88, 101]],
                'text': ['T-cell factor'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e1', 'offsets': [[0, 13]],
                'text': ['alpha-catenin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e2', 'offsets': [[23, 35]],
                'text': ['beta-catenin'], 'type': 'Individual_protein',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.e3', 'offsets': [[75, 87]],
                'text': ['beta-catenin'], 'type': 'Individual_protein',
                'normalized': []}],
        'relations': [
            {'id': 'BioInfer.d0.s0.i0', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e1',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i1', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e2',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i2', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e0', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i3', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e1', 'arg2_id': 'BioInfer.d0.s0.e2',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i4', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e1', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []},
            {'id': 'BioInfer.d0.s0.i5', 'type': 'PPI',
                'arg1_id': 'BioInfer.d0.s0.e2', 'arg2_id': 'BioInfer.d0.s0.e3',
                'normalized': []}]},
            {'document_id': 'BioInfer.d1.s0', 'type': 'Sentence',
            'text': 'A binary complex of birch profilin and skeletal muscle '
            'actin could be isolated by gel chromatography.',
            'entities': [
                {'id': 'BioInfer.d1.s0.e0', 'offsets': [[26, 34]],
                    'text': ['profilin'], 'type': 'Individual_protein',
                    'normalized': []},
                {'id': 'BioInfer.d1.s0.e1', 'offsets': [[39, 60]],
                    'text': ['skeletal muscle actin'], 'type': 'Individual_protein',
                    'normalized': []}],
            'relations': [
                {'id': 'BioInfer.d1.s0.i0', 'type': 'PPI',
                    'arg1_id': 'BioInfer.d1.s0.e0', 'arg2_id': 'BioInfer.d1.s0.e1',
                    'normalized': []}]}]

@pytest.fixture
def join_docs_unique_output(join_docs_unique_input):
    joined_dset = {}
    joined_dset['BioInfer.d0'] = [join_docs_unique_input[0]]
    joined_dset['BioInfer.d1'] = [join_docs_unique_input[1]]
    return joined_dset

def test_join_docs_unique(join_docs_unique_input,
        join_docs_unique_output):

    joined_dset = hbb.join_docs(join_docs_unique_input)

    assert joined_dset == join_docs_unique_output

