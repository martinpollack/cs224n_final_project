import pandas as pd
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import random

# Download required nltk data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')



# Function to get the wordnet POS tag from the nltk POS tag
def get_wordnet_pos(treebank_tag):
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

# Function to replace words with their synonyms based on POS tagging with a given probability
def synonym_replacement(sentence, replacement_probability=0.1):
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    new_sentence = []

    for word, pos in pos_tags:
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN  # Default to NOUN if no POS tag is found
        r = random.random()
        if r < replacement_probability:
            synonyms = wordnet.synsets(word, pos=wordnet_pos)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()  # Choose the first synonym
                new_sentence.append(synonym)
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)

    return ' '.join(new_sentence)

if __name__ == '__main__':
    # Load the dataset
    input_path = 'data/ids-cfimdb-train.csv'
    df = pd.read_csv(input_path, delimiter='\t', on_bad_lines='skip')
    print(df.head())

    # Apply the synonym replacement to the text column with a specified probability
    # Assuming the text column is named 'sentence'
    replacement_probability = 0.1
    df['sentence'] = df['sentence'].apply(lambda x: synonym_replacement(x, replacement_probability))

    # Save the modified DataFrame to a new CSV file
    output_path = 'data/ids_cfimdb_train_synonym.csv'
    with open(output_path, 'w') as f:
        f.write("\tid\tsentence\tsentiment\n")
        df.to_csv(f, index=False, header=False, sep='\t')

    print(f"Synonym replacement complete. The modified dataset is saved as {output_path}.")
