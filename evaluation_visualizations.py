import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_classification_report(csv_path, title, output_path):
    # Load the classification report from CSV
    report_df = pd.read_csv(csv_path, index_col=0)

    # Plot the precision, recall, and F1-score for each class
    metrics = ['precision', 'recall', 'f1-score']
    report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')  # Drop unnecessary rows

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    for i, metric in enumerate(metrics):
        report_df[metric].plot(kind='bar', ax=ax[i], color='skyblue')
        ax[i].set_title(f'{metric.capitalize()} per Class')
        ax[i].set_xlabel('Classes')
        ax[i].set_ylabel(metric.capitalize())
        ax[i].set_ylim(0, 1)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)  # Save the figure
    plt.close()
    print(f"Saved {title} visualization to '{output_path}'")


def visualize_sts_correlation(file_path, output_path):
    # Load STS correlation from the text file
    with open(file_path, 'r') as f:
        sts_corr = float(f.read().strip().split()[-1])

    plt.figure(figsize=(6, 4))
    plt.bar(['STS Correlation'], [sts_corr], color='skyblue')
    plt.ylim(0, 1)
    plt.title('STS Correlation')
    plt.ylabel('Correlation')
    plt.savefig(output_path)  # Save the figure
    plt.close()
    print(f"Saved STS correlation visualization to '{output_path}'")


def visualize_sts_results(csv_path, output_path_prefix):
    # Load the STS results from CSV
    sts_results_df = pd.read_csv(csv_path)

    # Plot predicted vs. true scores
    plt.figure(figsize=(10, 6))
    plt.scatter(sts_results_df['true_score'], sts_results_df['pred_score'], alpha=0.5)
    plt.title('STS Predicted vs True Scores')
    plt.xlabel('True Scores')
    plt.ylabel('Predicted Scores')
    plt.savefig(f'{output_path_prefix}_pred_vs_true.png')  # Save the figure
    plt.close()
    print(f"Saved STS predicted vs true scores visualization to '{output_path_prefix}_pred_vs_true.png'")

    # Plot distribution of prediction errors
    sts_results_df['error'] = sts_results_df['true_score'] - sts_results_df['pred_score']
    plt.figure(figsize=(10, 6))
    sns.histplot(sts_results_df['error'], bins=30, kde=True)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_path_prefix}_error_distribution.png')  # Save the figure
    plt.close()
    print(f"Saved STS prediction error distribution visualization to '{output_path_prefix}_error_distribution.png'")


# Visualize Sentiment Classification Report
visualize_classification_report('sentiment_classification_report.csv', 'Sentiment Classification Report',
                                'sentiment_classification_report.png')

# Visualize Paraphrase Classification Report
visualize_classification_report('paraphrase_classification_report.csv', 'Paraphrase Classification Report',
                                'paraphrase_classification_report.png')

# Visualize STS Correlation Report
visualize_sts_correlation('sts_correlation_report.txt', 'sts_correlation_report.png')

# Visualize STS Results
visualize_sts_results('sts_results.csv', 'sts_results')
