import random
import numpy as np
import argparse
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from bert import BertModel
from optimizer import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from datasets import (
    NLIClassificationDataset,
    load_nli_data
)
from curriculum_contrastive_learning import ContrastiveLearningLoss
from multitask_classifier import seed_everything

TQDM_DISABLE = False
BERT_HIDDEN_SIZE = 768


class BertEmbeddingTraining(nn.Module):
    """
    BERT embedding model for extracting [CLS] token embeddings.
    """
    def __init__(self):
        super(BertEmbeddingTraining, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask)
        pooler_output = bert_output['pooler_output']
        return pooler_output


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def pacing_function(t, T, k, lambda_):
    return int((t / T) ** lambda_ * k)


def main(args):
    """
    Main function to run the training loop.
    """
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    nli_data = load_nli_data(args.nli_filename)
    print("NLI Data loaded")
    dataset = NLIClassificationDataset(nli_data, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn,
                            num_workers=args.num_workers)
    print("NLI Dataloader created")

    # Setup config to be saved later
    config = {'tau': args.tau,
              'hidden_size': BERT_HIDDEN_SIZE,
              'data_dir': '.'}
    config = SimpleNamespace(**config)

    model = BertEmbeddingTraining().to(device)
    print("Model initialized")
    criterion = ContrastiveLearningLoss(tau=args.tau)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    if args.curriculum_training:
        difficulty_labels = torch.load(f"triplet_difficulty_classification_{args.distance_margin}.pt").to(device)
        print(f"Loaded difficulty labels from triplet_difficulty_classification_{args.distance_margin}.pt")
        sorted_indices = torch.argsort(difficulty_labels)
        k = len(difficulty_labels)
        T = args.epochs

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        if args.curriculum_training:
            g_t = pacing_function(epoch + 1, T, k, 1)
            selected_indices = sorted_indices[:g_t]
            subset = Subset(dataset, selected_indices)
            dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=True,
                                    collate_fn=dataset.collate_fn, num_workers=args.num_workers)

        for batch in tqdm(dataloader):
            token_ids_1 = batch['token_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            token_ids_2 = batch['token_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            token_ids_3 = batch['token_ids_3'].to(device)
            attention_mask_3 = batch['attention_mask_3'].to(device)

            optimizer.zero_grad()

            with autocast():
                premise_emb = model(token_ids_1, attention_mask_1)
                entailment_emb = model(token_ids_2, attention_mask_2)
                contradiction_emb = model(token_ids_3, attention_mask_3)

                loss = criterion(premise_emb, entailment_emb, contradiction_emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    # Save the trained model
    save_model(model, optimizer, args, config, args.output_model_path)
    print(f"Model saved to {args.output_model_path}")


def get_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    # DONE
    parser.add_argument("--nli_filename", type=str, default="data/nli_for_simcse.csv")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--tau', type=float, default=0.1, help='Temperature parameter for contrastive loss')
    parser.add_argument("--curriculum_training", action='store_true')
    parser.add_argument("--distance_margin", type=float, default=0.2)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    if args.curriculum_training:
        args.output_model_path = f'embeddings-{args.epochs}-{args.lr}-{args.tau}-contrastive_curriculum.pt'
    else:
        args.output_model_path = f'embeddings-{args.epochs}-{args.lr}-{args.tau}-contrastive-baseline.pt'  # Save path.
    seed_everything(args.seed)
    main(args)
