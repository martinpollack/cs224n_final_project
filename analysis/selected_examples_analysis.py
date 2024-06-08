import pandas as pd
import numpy as np

def main():
    # File paths
    dev_file = './data/quora-dev.csv'
    pred_file = './analysis/analysis_data/para_dev_output_alt_sts_loss_and_synonym_baseline.csv'

    # Load the dev and prediction files
    try:
        dev_df = pd.read_csv(dev_file, sep='\t')
        print(f"Loaded {len(dev_df)} rows from {dev_file}")
    except Exception as e:
        print(f"Failed to load {dev_file}: {e}")
        return

    try:
        pred_df = pd.read_csv(pred_file, sep=',', names=['id', 'Predicted_Is_Paraphrase'])
        print(f"Loaded {len(pred_df)} rows from {pred_file}")
    except Exception as e:
        print(f"Failed to load {pred_file}: {e}")
        return

    # Ensure id columns are of the same type and strip any leading/trailing spaces
    dev_df['id'] = dev_df['id'].astype(str).str.strip()
    pred_df['id'] = pred_df['id'].astype(str).str.strip()

    # Merge the dataframes on the id
    merged_df = dev_df.merge(pred_df, on='id', how='inner')
    print(f"Merged dataframe has {len(merged_df)} rows")

    # Compare the predictions with actual values and find mismatches
    mismatches = merged_df[merged_df['is_duplicate'] != merged_df['Predicted_Is_Paraphrase']]
    print(f"Found {len(mismatches)} mismatched rows")

    # Save the mismatched predictions to a CSV file
    # mismatches.to_csv('./analysis/mismatched_predictions.csv', index=False)
    # print("Mismatched predictions saved to './analysis/mismatched_predictions.csv'")

    # Randomly sample mismatched predictions
    sample_size = min(100, len(mismatches))  # Adjust the sample size as needed
    random_sample = mismatches.sample(n=sample_size, random_state=333)
    random_sample.to_csv('./analysis/analysis_data/random_sample_mismatched_predictions-politics.csv', index=False)
    print("Random sample of mismatched predictions saved to './analysis/analysis_data/random_sample_mismatched_predictions-politics.csv'")

    # Print the random sample for reference
    for index, row in random_sample.iterrows():
        print(f"ID: {row['id']}")
        print(f"Sentence 1: {row['sentence1']}")
        print(f"Sentence 2: {row['sentence2']}")
        print(f"Actual: {row['is_duplicate']}")
        print(f"Predicted: {row['Predicted_Is_Paraphrase']}\n")

if __name__ == "__main__":
    main()
