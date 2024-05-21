import pandas as pd
import nltk
from nltk.corpus import wordnet
import random

# Download required nltk data
nltk.download('wordnet')

# Load the dataset
input_path = 'data/ids-cfimdb-train.csv'
df = pd.read_csv(input_path, delimiter='\t', on_bad_lines='skip')
print(df.head())


# Function to replace words with their synonyms with a given probability
def synonym_replacement(sentence, replacement_probability=0.1):
    words = sentence.split()
    new_sentence = []

    for word in words:
        if random.random() < replacement_probability:
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = synonyms[0].lemmas()[0].name()  # Choose the first synonym
                new_sentence.append(synonym)
            else:
                new_sentence.append(word)
        else:
            new_sentence.append(word)

    return ' '.join(new_sentence)


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
