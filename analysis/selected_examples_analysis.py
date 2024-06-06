import pandas as pd

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

    # Check for any leading/trailing spaces in sentence1 column (if used in merge)
    dev_df['sentence1'] = dev_df['sentence1'].astype(str).str.strip()

    # Merge the dataframes on the id
    merged_df = dev_df.merge(pred_df, on='id', how='inner')
    print(f"Merged dataframe has {len(merged_df)} rows")

    # Compare the predictions with actual values and find mismatches
    mismatches = merged_df[merged_df['is_duplicate'] != merged_df['Predicted_Is_Paraphrase']]
    print(f"Found {len(mismatches)} mismatched rows")

    # Print the mismatched predictions along with sentences
    for index, row in mismatches.iterrows():
        print(f"ID: {row['id']}")
        print(f"Sentence 1: {row['sentence1']}")
        print(f"Sentence 2: {row['sentence2']}")
        print(f"Actual: {row['is_duplicate']}")
        print(f"Predicted: {row['Predicted_Is_Paraphrase']}\n")

if __name__ == "__main__":
    main()
