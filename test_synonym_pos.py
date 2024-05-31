import pytest
from nltk.corpus import wordnet
from synonym_pos import get_wordnet_pos, synonym_replacement

# Tests
def test_get_wordnet_pos():
    assert get_wordnet_pos('JJ') == wordnet.ADJ
    assert get_wordnet_pos('VB') == wordnet.VERB
    assert get_wordnet_pos('NN') == wordnet.NOUN
    assert get_wordnet_pos('RB') == wordnet.ADV
    assert get_wordnet_pos('UNK') is None

def test_synonym_replacement_no_replacement():
    sentence = "The quick brown fox jumps over the lazy dog"
    replaced_sentence = synonym_replacement(sentence, replacement_probability=0)
    assert replaced_sentence == sentence

def test_synonym_replacement_partial_replacement():
    sentence = "The quick brown fox jumps over the lazy dog"
    replaced_sentence = synonym_replacement(sentence, replacement_probability=0.5)
    assert replaced_sentence != sentence  # Likely to be different
    assert len(replaced_sentence.split()) == len(sentence.split())  # Same number of words

def test_synonym_replacement_full_replacement():
    sentence = "The quick brown fox jumps over the lazy dog"
    replaced_sentence = synonym_replacement(sentence, replacement_probability=1)
    assert replaced_sentence != sentence  # Should be different
    assert len(replaced_sentence.split()) == len(sentence.split())  # Same number of words


if __name__ == '__main__':
    print("Hello")
    pytest.main()
    print("Goodbye")
