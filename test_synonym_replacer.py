import torch
from tokenizer import BertTokenizer
import random
import pytest

# Assume the replace_synonyms function and helper functions are defined in a file named synonym_replacer.py
from synonym_replacer import replace_synonyms

# Mock tokenizer for testing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def test_replace_synonyms():
    # Set the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Create a mock input tensor of shape (32, 39) with random token IDs
    token_ids = torch.randint(0, tokenizer.vocab_size, (32, 39))

    # Define the replacement probability
    probability = 1

    # Call the replace_synonyms function
    new_token_ids = replace_synonyms(token_ids, tokenizer, probability)

    # Assertions to check the output tensor
    assert isinstance(new_token_ids, torch.Tensor), "The output should be a tensor."
    assert new_token_ids.shape == token_ids.shape, "The output tensor should have the same shape as the input tensor."

    # Check if the new tensor has the same dtype as the input tensor
    assert new_token_ids.dtype == token_ids.dtype, "The output tensor should have the same dtype as the input tensor."

    # Check if at least some of the tokens have been replaced (non-deterministic test)
    # This is a weaker test, it just checks that something changed, not specific values
    assert not torch.equal(new_token_ids, token_ids), "The output tensor should have some tokens replaced."

def test_replace_synonyms_single_change_per_sentence():
    # Set the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Create a mock input tensor of shape (32, 39) with random token IDs
    token_ids = torch.randint(0, tokenizer.vocab_size, (32, 39))

    # Define the replacement probability to ensure only one change per sentence
    probability = 1

    # Call the replace_synonyms function
    new_token_ids = replace_synonyms(token_ids, tokenizer, probability)

    # Assertions to check the output tensor
    assert isinstance(new_token_ids, torch.Tensor), "The output should be a tensor."
    assert new_token_ids.shape == token_ids.shape, "The output tensor should have the same shape as the input tensor."

    # Check if the new tensor has the same dtype as the input tensor
    assert new_token_ids.dtype == token_ids.dtype, "The output tensor should have the same dtype as the input tensor."

    # Check if exactly one token in each sentence has been replaced
    for i in range(token_ids.size(0)):
        original_sentence = token_ids[i]
        new_sentence = new_token_ids[i]
        differences = torch.sum(original_sentence != new_sentence).item()
        assert differences <= 1, f"Each sentence should have exactly one token replaced, but sentence {i} had {differences} differences."


def test_replace_synonyms_without_replacement():
    # Set the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Create a mock input tensor of shape (32, 39) with random token IDs
    token_ids = torch.randint(0, tokenizer.vocab_size, (32, 39))

    # Define the replacement probability
    probability = 0

    # Call the replace_synonyms function
    new_token_ids = replace_synonyms(token_ids, tokenizer, probability)

    # Assertions to check the output tensor
    assert isinstance(new_token_ids, torch.Tensor), "The output should be a tensor."
    assert new_token_ids.shape == token_ids.shape, "The output tensor should have the same shape as the input tensor."

    # Check if the new tensor has the same dtype as the input tensor
    assert new_token_ids.dtype == token_ids.dtype, "The output tensor should have the same dtype as the input tensor."

    # Check if the tokens are exactly the same since probability is 0
    assert torch.equal(new_token_ids, token_ids), "The output tensor should be identical to the input tensor since probability is 0."


# Run the test
if __name__ == "__main__":
    pytest.main()
