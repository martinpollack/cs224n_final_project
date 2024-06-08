import pandas as pd

# Define the path to your CSV file
csv_file_path = './data/quora-train.csv'

try:
    # Try to load the CSV file with tab delimiter
    df = pd.read_csv(csv_file_path, delimiter='\t')

    # Sample 100 rows randomly
    sampled_df = df.sample(n=100, random_state=333)  # Setting random_state for reproducibility

    # Save the sampled rows to a new CSV file
    sampled_df.to_csv('./analysis/analysis_data/quora-train-sampled-politics.csv', index=False, sep='\t')

    print("Random sample of 100 rows saved to quora-train-sampled-politics.csv")
except Exception as e:
    print(f"Error: {e}")
