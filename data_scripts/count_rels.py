"""
Function to easily classify valid and invalid relations for a given dataset.

Author: Serena G. Lotreck
"""

def count_validity(preds, valid_rels):
    """
    Count valid and invalid relations, No relations, and incorrectly formatted
    relations from a given GPT output.
    
    parameters:
        preds, dict: keys are doc keys, values are predictions from GPT
        valid_rels, list of str: valid relation labels
    
    returns:
        dataset_counts, dict: keys are count types, values are counts
        dataset_proportions, dict: counts normalized by the number of predicted
            relations
    """
    # Count number of total predicted rels
    total_num = 0
    for doc, trips in preds.items():
        total_num += len(trips)
 
    # Count each category 
    valid = 0
    invalid = 0
    bad_form = 0
    no_rel = 0
    for doc, trips in preds.items():
        for trip in trips:
            if trip == 'No relation':
                no_rel += 1
            elif isinstance(trip, list):
                try:
                    assert len(trip) == 3
                    if trip[1] not in valid_rels:
                        invalid += 1
                    elif trip[1] in valid_rels:
                        valid += 1
                except AssertionError:
                    bad_form += 1
            else:
                bad_form += 1
            
    # Make output dictionaries
    dataset_counts = {
        'valid': valid,
        'invalid': invalid,
        'bad_form': bad_form,
        'no_rel': no_rel
    }
    dataset_proportions = {
        'valid_prop': valid/total_num,
        'invalid_prop': invalid/total_num,
        'bad_form': bad_form/total_num,
        'no_rel': no_rel/total_num
    }

    return dataset_counts, dataset_proportions
                