import torch
import random
from tokenizer import BertTokenizer
from typing import Any
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag

# Download NLTK resources if not already available
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load the tokenizer corresponding to 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_wordnet_pos(treebank_tag):
    """Convert Treebank POS tag to WordNet POS tag."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def do_replacement(word, pos, new_sentence):
    wordnet_pos = get_wordnet_pos(pos)
    if wordnet_pos == wordnet.ADJ:
        synonyms = wordnet.synsets(word, pos=wordnet_pos)
        if synonyms:
            all_lemmas = [lemma for synset in synonyms for lemma in synset.lemmas()]
            print("There exist {} synonyms!".format(len(all_lemmas)))
            if all_lemmas:
                synonym = random.choice(all_lemmas).name()  # Choose a random synonym
                print("I replaced {} with {}!".format(word, synonym))
                new_sentence.append(synonym)
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)
    else:
        new_sentence.append(word)

def replace_synonyms(ids: torch.Tensor, tokenizer: BertTokenizer, p: float) -> torch.Tensor:
    """
    A function that replaces certain token IDs with their synonyms based on a given probability.
    
    Args:
        ids (torch.Tensor): A tensor of token IDs.
        tokenizer (BertTokenizer): The tokenizer to use for converting token IDs to words and back.
        p (float): The probability of replacing a token ID with a synonym.
        
    Returns:
        torch.Tensor: The tensor of token IDs after potentially replacing some with synonyms.
    """
    # Print the shape of the tensor
    print(f"Shape of ids tensor: {ids.shape}")

    # Randomly decide whether to perform replacements
    r = random.random()
    if r >= p:
        return ids

    print("HIT!")

    # Initialize a list to hold the new sentences
    new_sentences = []

    # Iterate over each sentence in the batch
    for sentence in ids:
        # Convert token IDs to words
        words = tokenizer.convert_ids_to_tokens(sentence.tolist())
        # Get POS tags for the words
        pos_tags = pos_tag(words)
        
        new_sentence = []
        
        # Determine iteration direction --> 50-50 traverse from front or back to find the first adjective to replace
        forward_direction = random.random() < 0.5

        # Iterate through pos_tags in the chosen direction
        if forward_direction:
            for word, pos in pos_tags:
                do_replacement(word, pos, new_sentence)
        else:
            # Backward direction of the sentence
            for word, pos in reversed(pos_tags):
                do_replacement(word, pos, new_sentence)
            new_sentence.reverse()  # Reverse to maintain original order
        
        # Convert the new sentence back to token IDs
        new_ids = tokenizer.convert_tokens_to_ids(new_sentence)
        new_sentences.append(new_ids)
    
    # Convert the list of new sentences back to a tensor
    new_ids_tensor = torch.tensor(new_sentences)

    # Print the shape of the tensor
    print(f"Shape of new_ids_tensor: {new_ids_tensor.shape}")
    return new_ids_tensor