import pandas as pd

# Load the dev and prediction files
dev_file = 'data/quora-dev.csv'
pred_file = 'analysis/analysis_data/para_dev_output_alt_sts_loss_and_synonym_baseline.csv'

dev_df = pd.read_csv(dev_file, sep='\t')
pred_df = pd.read_csv(pred_file, sep=',', names=['id', 'Predicted_Is_Paraphrase'])

# Merge the dataframes on the id and sentence identifier
merged_df = dev_df.merge(pred_df, left_on=['id', 'sentence1'], right_on=['id', 'id'])

# Compare the predictions with actual values and find mismatches
mismatches = merged_df[merged_df['is_duplicate'] != merged_df['Predicted_Is_Paraphrase']]

# Print the mismatched predictions along with sentences
for index, row in mismatches.iterrows():
    print(f"ID: {row['id']}")
    print(f"Sentence 1: {row['sentence1']}")
    print(f"Sentence 2: {row['sentence2']}")
    print(f"Actual: {row['is_duplicate']}")
    print(f"Predicted: {row['Predicted_Is_Paraphrase']}\n")

