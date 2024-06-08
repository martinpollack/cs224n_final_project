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

    # Find the correctly predicted rows
    correct_predictions = merged_df[merged_df['is_duplicate'] == merged_df['Predicted_Is_Paraphrase']]
    print(f"Found {len(correct_predictions)} correctly predicted rows")

    # Save the correctly predicted rows to a CSV file
    correct_predictions.to_csv('./analysis/analysis_data/quora_dev_correct_predictions_all.csv', index=False)
    print("Correctly predicted rows saved to './analysis/analysis_data/quora_dev_correct_predictions_all.csv'")

if __name__ == "__main__":
    main()
