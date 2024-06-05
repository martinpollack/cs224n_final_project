import torch
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity
from datasets import load_nli_data, NLIClassificationDataset
from multitask_classifier import seed_everything
from bert import BertModel
import argparse
from tqdm import tqdm
import pandas as pd

def calculate_difficulty_labels(args):
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

        with torch.no_grad():
            b_bert_embeddings_1 = bert(b_ids_1, b_mask_1)['pooler_output']
            b_bert_embeddings_2 = bert(b_ids_2, b_mask_2)['pooler_output']
            b_bert_embeddings_3 = bert(b_ids_3, b_mask_3)['pooler_output']

        premise_entailment_dists.append(1 - cos(b_bert_embeddings_1, b_bert_embeddings_2))
        premise_contradiction_dists.append(1 - cos(b_bert_embeddings_1, b_bert_embeddings_3))

    premise_entailment_dists = torch.cat(premise_entailment_dists, dim=0).to(device)
    premise_contradiction_dists = torch.cat(premise_contradiction_dists, dim=0).to(device)

    torch.save(premise_entailment_dists, "premise_entailment_dists.pt")
    torch.save(premise_contradiction_dists, "premise_contradiction_dists.pt")

    assign_difficulty_labels(args)

def assign_difficulty_labels(args):
    premise_entailment_dists = torch.load("premise_entailment_dists.pt")
    premise_contradiction_dists = torch.load("premise_contradiction_dists.pt")

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    premise_entailment_dists.to(device)
    premise_contradiction_dists.to(device)

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
    torch.save(difficulty_classification, f"triplet_difficulty_classification_{args.distance_margin}.pt")

    print(torch.unique(difficulty_classification, return_counts=True))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--distance_margin", type=float, default=0.2)
    parser.add_argument("--full_run", action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    calculate_difficulty_labels(args)
