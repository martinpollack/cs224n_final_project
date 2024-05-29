import torch
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity
from datasets import load_nli_data, NLIClassificationDataset
from multitask_classifier import seed_everything
from bert import BertModel
import argparse
from tqdm import tqdm
import pandas as pd


def main(args):
    # Load and sample data
    nli_data = load_nli_data("data/nli_for_simcse.csv")

    nli_dataset = NLIClassificationDataset(nli_data, args)
    nli_dataloader = DataLoader(nli_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=nli_dataset.collate_fn, num_workers=args.num_workers)

    # Load BERT model
    bert = BertModel.from_pretrained('bert-base-uncased')
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    bert.to(device)

    cos = CosineSimilarity(dim=1)
    premise_entailment_dists = []
    premise_contradiction_dists = []

    for batch in tqdm(nli_dataloader, desc="Processing batches"):
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_ids_3, b_mask_3 = (batch['token_ids_1'],
                                                                   batch['attention_mask_1'],
                                                                   batch['token_ids_2'],
                                                                   batch['attention_mask_2'],
                                                                   batch['token_ids_3'],
                                                                   batch['attention_mask_3'])

        b_ids_1, b_mask_1 = b_ids_1.to(device), b_mask_1.to(device)
        b_ids_2, b_mask_2 = b_ids_2.to(device), b_mask_2.to(device)
        b_ids_3, b_mask_3 = b_ids_3.to(device), b_mask_3.to(device)

        with torch.no_grad():  # Disable gradient calculation
            b_bert_embeddings_1 = bert(b_ids_1, b_mask_1)['pooler_output']
            b_bert_embeddings_2 = bert(b_ids_2, b_mask_2)['pooler_output']
            b_bert_embeddings_3 = bert(b_ids_3, b_mask_3)['pooler_output']

        premise_entailment_dists.append(1 - cos(b_bert_embeddings_1, b_bert_embeddings_2))
        premise_contradiction_dists.append(1 - cos(b_bert_embeddings_1, b_bert_embeddings_3))

        # b_bert_embeddings = torch.cat((b_bert_embeddings_1.unsqueeze(-1), b_bert_embeddings_2.unsqueeze(-1), b_bert_embeddings_3.unsqueeze(-1)),
        #                               dim=-1)

        # bert_embeddings.append(b_bert_embeddings)

    # Concatenate all collected distances and move to device
    premise_entailment_dists = torch.cat(premise_entailment_dists, dim=0).to(device)
    premise_contradiction_dists = torch.cat(premise_contradiction_dists, dim=0).to(device)

    # bert_embeddings = torch.cat(bert_embeddings, dim=0)

    # premise = bert_embeddings[:, :, 0]

    # entailment = bert_embeddings[:, :, 1]

    # contradiction = bert_embeddings[:, :, 2]

    # cos = CosineSimilarity(dim=1)

    # premise_entailment_dist = 1 - cos(premise, entailment)

    # premise_contradiction_dist = 1 - cos(premise, contradiction)

    # Difficulty classification
    difficulty_classification = torch.zeros(premise_entailment_dists.shape[0], device=device)
    m = args.distance_margin

    difficulty_classification = torch.where((premise_entailment_dists < premise_contradiction_dists) &
                                            (premise_contradiction_dists <= premise_entailment_dists + m),
                                            torch.full(premise_entailment_dists.shape, 1, device=device),
                                            difficulty_classification)

    difficulty_classification = torch.where(premise_contradiction_dists <= premise_entailment_dists,
                                            torch.full(premise_entailment_dists.shape, 2, device=device),
                                            difficulty_classification)

    print(difficulty_classification)
    torch.save(difficulty_classification, 'triplet_difficulty_classification.pt')


# def main():

#     snli_data, multinli_data = load_nli_data(snli_filename='data/snli_1.0/snli_1.0_train.txt', multinli_filename='data/multinli_1.0/multinli_1.0_train.txt')

#     snli_data = group_data(snli_data)

#     multinli_data = group_data(multinli_data)

#     full_nli_data = []
#     full_nli_data.extend(snli_data)
#     full_nli_data.extend(multinli_data)

#     print(len(full_nli_data))

# def group_data(data):
#     df = pd.DataFrame(data, columns=('sentence1', 'sentence2', 'gold_label'))
#     print(df.shape)
#     output = df.loc[df['gold_label'].isin(['entailment', 'contradiction']), :]\
#         .sort_values(['sentence1', 'gold_label'], ascending=False)\
#         .groupby('sentence1')\
#         .agg({'sentence2': tuple})\
#         .reset_index()

#     print(output.shape)

#     output = output.loc[output['sentence2'].apply(len) == 2, :]\
#         .apply(create_tuple, axis=1)\
#         .values\
#         .tolist()

#     return output

# def create_tuple(row):
#         return (row['sentence1'], row['sentence2'][0], row['sentence2'][1])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--distance_margin", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
