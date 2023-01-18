"""
Module to load a BERT model and get its embeddings.

Code to prepare inputs and get embeddings taken from
https://github.com/arushiprakash/MachineLearning/blob/main/BERT%20Word%20Embeddings.ipynb

Author: Serena G. Lotreck
"""
from transformers import AutoTokenizer, AutoModelForTokenClassification


def load_model(pretrained="alvaroalon2/biobert_genetic_ner"):
    """
    Load model and tokenizer.

    parameters:
        pretrained, str: name of the pretrained model to load.
            Documentation for the default option found at
            https://huggingface.co/alvaroalon2/biobert_genetic_ner

    returns:
        tokenizer, AutoTokenizer instance: tokenizer
        model, AutoModelForTokenClassification instance: model
    """
    # Note: I'm not sure that these classes work with all models,
    # may need to change to include other models later
    tokenizer = AutoTokenizer.from_pretrained(
            "alvaroalon2/biobert_genetic_ner")

    model = AutoModelForTokenClassification.from_pretrained(
            "alvaroalon2/biobert_genetic_ner")

    return tokenizer, model


def embed_labels(label_dict, tokenizer, model):
    """
    Get context-averaged embeddings for the relation labels.

    parameters:
        label_dict, dict: keys are labels, values are lists of sentences
            that use the label word literally to indicate a relation
        tokenizer: tokenizer
        model: model

    returns:
        label_embed_dict, dict: keys are labels, values are their embeddings
    """
    label_embed_dict = {}
    for label, sents in label_dict.items():
        total_embeds = []
        for sent in sents:
            embed = get_phrase_embedding(sent, label, tokenizer, model)
            total_embeds.append(embed)
        avg_embed = np.mean(np.asarray(total_embeds), axis=0)
        label_embed_dict[label] = avg_embed

    return label_embed_dict


def get_phrase_embedding(sent, phrase, tokenizer, model):
    """
    Gets the embedding for a given phrase.For OOV words, averages the
    vectors of the subwords to generate a representation. For multi-word
    n-grams, the same process is used.

    parameters:
        sentence, str: entire sentence in which the phrase is contained
        phrase, str: phrase to embed
        tokenizer: tokenizer
        model: model

    returns:
        embedding, vector: embedding for the phrase.
    """
    # Get the embeddings for all tokens in the 
    tokenized_sent, tokens_tensor, segments_tensors = bert_text_preparation(
            sent, tokenizer)
    list_token_embeddings = get_bert_embeddings(tokens_tensor,
            segments_tensors, model)

    # Tokenize the phrase in order to get the indices of the phrase
    # within the tokenized sentence
    tokenized_phrase = tokenizer.tokenize(phrase)

    # Get the indices of this sublist in the tokenized sentence
    start_idx, end_idx = find_sub_list(tokenized_phrase, tokenized_sent)

    # Get all embedings and average
    embed_mat = np.asarray(list_token_embeddings[start_idx:end_idx])
    embedding = np.mean(embed_mat, axis=0)

    return embedding


def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT

    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.

    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids

    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model

    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids

    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token

    """

    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist()
        for token_embed in token_embeddings]

    return list_token_embeddings


def find_sub_list(sl,l):
    """
    Get the start and end indices of a sublist in a list.

    Code from https://stackoverflow.com/a/17870684/13340814
    """
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1
