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
def PMID24961571_abstract_trips():
    return {'PMID24961571_abstract': [
        ["HvOPR2", "interacts", "PCR amplification technique"],
        ["HvOPR2", "inhibits", "semi-quantitative RT-PCR"]]}

@pytest.fixture
def PMID18941053_abstract_trips():
    return {'PMID18941053_abstract':  [
        ["Seed germination", "produces", "phytohormones"],
        ["seed germination", "interacts", "germination"],
        ["ABA", "activates", "ABI5"], # Note that this isn't what was
        # predicted, but the performance is so bad that there are no other
        # matching relations so I changed this one
        ["GA synthesis", "interacts", "XERICO"],
        ["ABI5 activity", "inhibits", "phosphorylation"],
        ["sleepy1 mutant seeds", "inhibits", "ABI5 expression"],
        ["ABI5", "inhibits", "ABA and GA levels"]]}

@pytest.fixture
def interacts_order_same_trips():
    return {'doc3': [['Protein 1', 'interacts', 'Protein 2'],
        ['Protein 1', 'activates', 'Protein 2']]}

@pytest.fixture
def interacts_order_reversed_trips():
    return {'doc4': [['Protein 2', 'interacts', 'Protein 1'],
        ['Protein 1', 'activates', 'Protein 2']]}

@pytest.fixture
def interacts_and_activates_order_reversed_trips():
    return {'doc4': [['Protein 2', 'interacts', 'Protein 1'],
        ['Protein 1', 'activates', 'Protein 2']]}

@pytest.fixture
def duplicate_goldstd_trips():
    return {'doc4': [['Protein 2', 'interacts', 'Protein 1'],
        ['Protein 1', 'activates', 'Protein 2']]}

@pytest.fixture
def no_punc_gold_trips():
    return {'doc1':[("my name", "is", "sparty"),
            ("I", "work at", "MSU")]}

@pytest.fixture
def punc_gold_trips():
    return {'doc2':[("hello , my name", "is", "sparty"),
            ("I", "work in", "East Lansing , Michigan")]}

@pytest.fixture
def PMID24961571_abstract_gold_trips():
    return {'PMID24961571_abstract':  [["HvOPR2", "is-in", "barley"],
        ["OPR gene family", "is-in", "barley"], ["HvOPR2", "is-in", "Hordeum vulgare L."],
        ["OPR gene family", "is-in", "Hordeum vulgare L."],
        ["flavin oxidoreductase/NADH oxidase substrate-binding domain", "is-in", "HvOPR2"],
        ["HvOPR2", "produces", "OPR of subgroup I"],
        ["hydrogen peroxide", "activates", "HvOPR2 gene"],
        ["HvOPR2", "interacts", "barley defense/response to abiotic stresses and signaling molecules"]]}

@pytest.fixture
def PMID18941053_abstract_gold_trips():
    return {'PMID18941053_abstract': [
        ["phytohormones gibberellic acid ( GA ) and abscisic acid ( ABA )", "interacts", "Seed germination"],
        ["GA", "activates", "seed germination"],
        ["GA", "activates", "proteasome-mediated destruction of RGL2"],
        ["DELLA factor", "inhibits", "seed germination"],
        ["RGA-LIKE2", "inhibits", "seed germination"],
        ["ABA", "activates", "ABA-INSENSITIVE5"],
        ["ABA", "activates", "basic domain/leucine zipper transcription factor"],
        ["ABA", "activates", "ABI5"],
        ["GA synthesis", "interacts", "stabilized RGL2"],
        ["XERICO", "activates", "ABA synthesis"],
        ["GA synthesis", "activates", "endogenous ABA"],
        ["RING-H2 zinc finger factor", "activates", "ABA synthesis"],
        ["endogenous ABA synthesis", "activates", "ABI5 RNA and protein"],
        ["endogenous ABA synthesis", "activates", "RGL2"],
        ["ABI5 protein", "inhibits", "seed germination"],
        ["SnRK2 protein kinase", "activates", "endogenous ABI5 phosphorylation"],
        ["RGL2", "is-in", "sleepy1 mutant seeds"],
        ["ABI5", "interacts", "ABA"], ["ABI5", "interacts", "GA"]]}

@pytest.fixture
def interacts_order_same_gold_trips():
    return {'doc3': [['Protein 1', 'interacts', 'Protein 2'],
        ['Protein 1', 'activates', 'Protein 2'],
        ['Arabidopsis', 'interacts', 'B. nigra']]}

@pytest.fixture
def interacts_order_reversed_gold_trips():
    return {'doc4': [['Protein 1', 'interacts', 'Protein 2'],
        ['Protein 1', 'activates', 'Protein 2'],
        ['Arabidopsis', 'interacts', 'B. nigra']]}

@pytest.fixture
def interacts_and_activates_order_reversed_gold_trips():
    return {'doc4': [['Protein 1', 'interacts', 'Protein 2'],
        ['Protein 2', 'activates', 'Protein 1'],
        ['Arabidopsis', 'interacts', 'B. nigra']]}

@pytest.fixture
def duplicate_goldstd_gold_trips():
    return {'doc4': [['Protein 2', 'interacts', 'Protein 1'],
        ['Protein 2', 'activates', 'Protein 1'],
        ['Protein 1', 'interacts', 'Protein 2'],
        ['Protein 1', 'activates', 'Protein 2']]}

@pytest.fixture
def PMID18941053_abstract_gold_trips_undeduped_no_rel_labs():
    return [
        [["phytohormones", "gibberellic", "acid", "(", "GA", ")" "and",
            "abscisic", "acid", "(", "ABA", ")"], ["Seed", "germination"]],
        [["GA"], ["seed", "germination"]],
        [["GA"], ["proteasome-mediated", "destruction", "of", "RGL2"]],
        [["DELLA", "factor"], ["seed", "germination"]],
        [["RGA-LIKE2"], ["seed", "germination"]],
        [["ABA"], ["ABA-INSENSITIVE5"]],
        [["ABA"], ["basic", "domain/leucine", "zipper", "transcription",
            "factor"]],
        [["ABA"], ["ABI5"]],
        [["GA", "synthesis"], ["stabilized", "RGL2"]],
        [["XERICO"], ["ABA", "synthesis"]],
        [["GA", "synthesis"], ["endogenous", "ABA"]],
        [["RING-H2", "zinc", "finger", "factor"], ["ABA", "synthesis"]],
        [["endogenous", "ABA", "synthesis"], ["ABI5", "RNA", "and", "protein"]],
        [["endogenous", "ABA", "synthesis"], ["RGL2"]],
        [["ABI5", "protein"], ["seed", "germination"]],
        [["SnRK2", "protein", "kinase"], ["endogenous", "ABI5",
            "phosphorylation"]],
        [["RGL2"], ["sleepy1", "mutant", "seeds"]],
        [["ABI5"], ["ABA"]], [["ABI5"], ["GA"]]]

@pytest.fixture
def PMID18941053_abstract_gold_trips_deduped_no_rel_labs():
    return [
        [["phytohormones", "gibberellic", "acid", "(", "GA", ")" "and",
            "abscisic", "acid", "(", "ABA", ")"], ["Seed", "germination"]],
        [["GA"], ["seed", "germination"]],
        [["GA"], ["proteasome-mediated", "destruction", "of", "RGL2"]],
        [["DELLA", "factor"], ["seed", "germination"]],
        [["RGA-LIKE2"], ["seed", "germination"]],
        [["ABA"], ["ABA-INSENSITIVE5"]],
        [["ABA"], ["basic", "domain/leucine", "zipper", "transcription",
            "factor"]],
        [["ABA"], ["ABI5"]],
        [["GA", "synthesis"], ["stabilized", "RGL2"]],
        [["XERICO"], ["ABA", "synthesis"]],
        [["GA", "synthesis"], ["endogenous", "ABA"]],
        [["RING-H2", "zinc", "finger", "factor"], ["ABA", "synthesis"]],
        [["endogenous", "ABA", "synthesis"], ["ABI5", "RNA", "and", "protein"]],
        [["endogenous", "ABA", "synthesis"], ["RGL2"]],
        [["ABI5", "protein"], ["seed", "germination"]],
        [["SnRK2", "protein", "kinase"], ["endogenous", "ABI5",
            "phosphorylation"]],
        [["RGL2"], ["sleepy1", "mutant", "seeds"]],
        [["ABI5"], ["GA"]]]

@pytest.fixture
def duplicate_goldstd_gold_trips_undeduped_rel_labs():
    return [[['Protein', '2'], ['interacts'], ['Protein', '1']],
        [['Protein', '2'], ['activates'], ['Protein', '1']],
        [['Protein', '1'], ['interacts'], ['Protein', '2']],
        [['Protein', '1'], ['activates'], ['Protein', '2']]]

@pytest.fixture
def duplicate_goldstd_gold_trips_deduped_rel_labs():
    return [[['Protein', '2'], ['interacts'], ['Protein', '1']],
        [['Protein', '2'], ['activates'], ['Protein', '1']],
        [['Protein', '1'], ['activates'], ['Protein', '2']]]

@pytest.fixture
def duplicate_goldstd_gold_trips_undeduped_no_rel_labs():
    return [[['Protein', '2'], ['Protein', '1']],
        [['Protein', '2'], ['Protein', '1']],
        [['Protein', '1'], ['Protein', '2']],
        [['Protein', '1'], ['Protein', '2']]]

@pytest.fixture
def duplicate_goldstd_gold_trips_deduped_no_rel_labs():
    return [[['Protein', '2'], ['Protein', '1']]]

@pytest.fixture
def nlp():
    return spacy.load("en_core_sci_sm")

@pytest.fixture
def sym_labs():
    return ['interacts']

@pytest.fixture
def no_punc_f1_input_no_rel_lab():
    return (2, 2, 1)

@pytest.fixture
def punc_f1_input_no_rel_lab():
    return (3, 2, 2)

@pytest.fixture
def PMID24961571_abstract_no_rel_lab():
    return (2, 8, 0)

@pytest.fixture
def PMID18941053_abstract_no_rel_lab():
    return (7, 18, 1)

@pytest.fixture
def no_punc_f1_input_rel_lab():
    return (2, 2, 0)

@pytest.fixture
def punc_f1_input_rel_lab():
    return (3, 2, 1)

@pytest.fixture
def PMID24961571_abstract_rel_lab():
    return (2, 8, 0)

@pytest.fixture
def PMID18941053_abstract_rel_lab():
    return (7, 19, 1)

@pytest.fixture
def interacts_order_same_rel_lab():
    return (2, 3, 2)

@pytest.fixture
def interacts_order_reversed_rel_lab():
    return (2, 3, 2)

@pytest.fixture
def interacts_and_activates_order_reversed_rel_lab():
    return (2, 3, 1)

@pytest.fixture
def interacts_and_activates_order_reversed_no_rel_lab():
    return (1, 2, 1)

@pytest.fixture
def duplicate_goldstd_rel_lab():
    return (2, 3, 2)


def test_remove_trip_dups_PMID18941053_abstract_gold_trips_deduped_no_rel_labs(
        PMID18941053_abstract_gold_trips_undeduped_no_rel_labs,
        PMID18941053_abstract_gold_trips_deduped_no_rel_labs, sym_labs):

    unduped_trips = et.remove_trip_dups(
            PMID18941053_abstract_gold_trips_undeduped_no_rel_labs,
            False, sym_labs)

    assert unduped_trips == PMID18941053_abstract_gold_trips_deduped_no_rel_labs

def test_remove_trip_dups_duplicate_goldstd_gold_trips_deduped_rel_labs(
        duplicate_goldstd_gold_trips_undeduped_rel_labs,
        duplicate_goldstd_gold_trips_deduped_rel_labs, sym_labs):

    unduped_trips = et.remove_trip_dups(
            duplicate_goldstd_gold_trips_undeduped_rel_labs, True, sym_labs)

    assert unduped_trips == duplicate_goldstd_gold_trips_deduped_rel_labs

def test_remove_trip_dups_duplicate_goldstd_gold_trips_deduped_no_rel_labs(
        duplicate_goldstd_gold_trips_undeduped_no_rel_labs,
        duplicate_goldstd_gold_trips_deduped_no_rel_labs, sym_labs):

    unduped_trips = et.remove_trip_dups(
            duplicate_goldstd_gold_trips_undeduped_no_rel_labs, False, sym_labs)

    assert unduped_trips == duplicate_goldstd_gold_trips_deduped_no_rel_labs

def test_get_f1_input_no_punc(no_punc_trips, no_punc_gold_trips, nlp,
        no_punc_f1_input_no_rel_lab):

    f1_input = et.get_f1_input(no_punc_trips, no_punc_gold_trips, nlp, [])

    assert f1_input == no_punc_f1_input_no_rel_lab

def test_get_f1_input_punc_no_rel_lab(punc_trips, punc_gold_trips, nlp,
        punc_f1_input_no_rel_lab):

    f1_input = et.get_f1_input(punc_trips, punc_gold_trips, nlp, [])

    assert f1_input == punc_f1_input_no_rel_lab

def test_get_f1_input_PMID24961571_abstract_no_rel_lab(PMID24961571_abstract_trips,
        PMID24961571_abstract_gold_trips, nlp, PMID24961571_abstract_no_rel_lab):

    f1_input = et.get_f1_input(PMID24961571_abstract_trips,
            PMID24961571_abstract_gold_trips, nlp, [])

    assert f1_input == PMID24961571_abstract_no_rel_lab

def test_get_f1_input_PMID18941053_abstract_no_rel_lab(PMID18941053_abstract_trips,
        PMID18941053_abstract_gold_trips, nlp, PMID18941053_abstract_no_rel_lab):

    f1_input = et.get_f1_input(PMID18941053_abstract_trips,
            PMID18941053_abstract_gold_trips, nlp, [])

    assert f1_input == PMID18941053_abstract_no_rel_lab

def test_get_f1_input_no_punc_rel_lab(no_punc_trips, no_punc_gold_trips, nlp,
        no_punc_f1_input_rel_lab, sym_labs):

    f1_input = et.get_f1_input(no_punc_trips, no_punc_gold_trips, nlp,
            sym_labs, check_rel_labels=True)

    assert f1_input == no_punc_f1_input_rel_lab

def test_get_f1_input_punc_rel_lab(punc_trips, punc_gold_trips, nlp,
        punc_f1_input_rel_lab, sym_labs):

    f1_input = et.get_f1_input(punc_trips, punc_gold_trips, nlp,
            sym_labs, check_rel_labels=True)

    assert f1_input == punc_f1_input_rel_lab

def test_get_f1_input_PMID24961571_abstract_rel_lab(PMID24961571_abstract_trips,
        PMID24961571_abstract_gold_trips, nlp, PMID24961571_abstract_rel_lab,
        sym_labs):

    f1_input = et.get_f1_input(PMID24961571_abstract_trips,
            PMID24961571_abstract_gold_trips, nlp, sym_labs, check_rel_labels=True)

    assert f1_input == PMID24961571_abstract_rel_lab

def test_get_f1_input_PMID18941053_abstract_rel_lab(PMID18941053_abstract_trips,
        PMID18941053_abstract_gold_trips, nlp, PMID18941053_abstract_rel_lab,
        sym_labs):

    f1_input = et.get_f1_input(PMID18941053_abstract_trips,
            PMID18941053_abstract_gold_trips, nlp, sym_labs, check_rel_labels=True)

    assert f1_input == PMID18941053_abstract_rel_lab

def test_get_f1_input_interacts_order_same_rel_lab(interacts_order_same_trips,
        interacts_order_same_gold_trips, nlp, interacts_order_same_rel_lab,
        sym_labs):

    f1_input = et.get_f1_input(interacts_order_same_trips,
            interacts_order_same_gold_trips, nlp, sym_labs, check_rel_labels=True)

    assert f1_input == interacts_order_same_rel_lab


def test_get_f1_input_interacts_order_reversed_rel_lab(interacts_order_reversed_trips,
        interacts_order_reversed_gold_trips, nlp, interacts_order_reversed_rel_lab,
        sym_labs):

    f1_input = et.get_f1_input(interacts_order_reversed_trips,
            interacts_order_reversed_gold_trips, nlp, sym_labs, check_rel_labels=True)

    assert f1_input == interacts_order_reversed_rel_lab

def test_get_f1_input_interacts_and_activates_order_reversed_rel_lab(
        interacts_and_activates_order_reversed_trips,
        interacts_and_activates_order_reversed_gold_trips, nlp,
        interacts_and_activates_order_reversed_rel_lab,
        sym_labs):

    print('TEST HELLO\n\n\n')
    print(f'sym labs passed to test: {sym_labs}')
    f1_input = et.get_f1_input(interacts_and_activates_order_reversed_trips,
            interacts_and_activates_order_reversed_gold_trips, nlp, sym_labs,
            check_rel_labels=True)

    assert f1_input == interacts_and_activates_order_reversed_rel_lab

def test_get_f1_input_interacts_and_activates_order_reversed_no_rel_lab(
        interacts_and_activates_order_reversed_trips,
        interacts_and_activates_order_reversed_gold_trips, nlp,
        interacts_and_activates_order_reversed_no_rel_lab,
        sym_labs):

    f1_input = et.get_f1_input(interacts_and_activates_order_reversed_trips,
            interacts_and_activates_order_reversed_gold_trips, nlp,
            sym_labs, check_rel_labels=False)

    assert f1_input == interacts_and_activates_order_reversed_no_rel_lab

def test_get_f1_input_duplicate_goldstd_rel_lab(duplicate_goldstd_trips,
        duplicate_goldstd_gold_trips, nlp, duplicate_goldstd_rel_lab,
        sym_labs):

    f1_input = et.get_f1_input(duplicate_goldstd_trips,
            duplicate_goldstd_gold_trips, nlp, sym_labs, check_rel_labels=True)

    assert f1_input == duplicate_goldstd_rel_lab
