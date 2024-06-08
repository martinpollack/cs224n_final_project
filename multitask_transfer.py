'''
Multitask BERT class, training code, evaluation code, transfer learning.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask
from contrastive_training import BertEmbeddingTraining
from synonym_replacer import replace_synonyms

TQDM_DISABLE = False


def load_model(filepath, device='cuda'):
    """
    Load the saved model, optimizer state, training arguments, and random states.

    Args:
        filepath (str): Path to the saved model file.
        device (str): Device to load the model on ('cpu' or 'cuda').

    Returns:
        model (BertEmbeddingModel): The loaded model.
        optimizer (torch.optim.Optimizer): The loaded optimizer.
        args (argparse.Namespace): The training arguments.
        config (dict): The model configuration.
    """
    checkpoint = torch.load(filepath, map_location=device)

    # Recreate the model
    model = BertEmbeddingTraining()
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # Recreate the optimizer
    optimizer = AdamW(model.parameters(), lr=checkpoint['args'].lr)
    optimizer.load_state_dict(checkpoint['optim'])

    # Load other items
    args = checkpoint['args']
    config = checkpoint['model_config']

    # Restore random states
    # random.setstate(checkpoint['system_rng'])
    # np.random.set_state(checkpoint['numpy_rng'])
    # torch.random.set_rng_state(checkpoint['torch_rng'])

    return model, optimizer, args, config

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, preds, target):
        return 1 - self.cosine_similarity(preds, target)


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERTTransfer(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config, pretrained_model):
        super(MultitaskBERTTransfer, self).__init__()
        self.bert = pretrained_model
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        # Sentiment classification layers
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, N_SENTIMENT_CLASSES)
        )

        # Paraphrase detection layers
        self.paraphrase_classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Semantic Textual Similarity (STS) layers
        if args.alt_sts_loss:
            self.cosine_head = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, 128),
                nn.ReLU()
            )
            self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
            self.relu = nn.ReLU()
        else:
            self.sts_classifier = nn.Sequential(
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        # Directly extract pooler_output as in classifier.py
        pooler_output = self.bert(input_ids, attention_mask)
        return pooler_output

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        # use same implementation as classifier.py
        x = self.forward(input_ids, attention_mask)
        x = self.sentiment_classifier(x)
        return x

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        # pooler_output_1 = self.forward(input_ids_1, attention_mask_1)
        # pooler_output_2 = self.forward(input_ids_2, attention_mask_2)
        # x = torch.cat((pooler_output_1, pooler_output_2), dim=1)
        input_ids_cat = torch.cat((input_ids_1, input_ids_2), dim=1)
        attention_mask_cat = torch.cat((attention_mask_1, attention_mask_2), dim=1)
        pooler_output = self.forward(input_ids_cat, attention_mask_cat)
        x = self.paraphrase_classifier(pooler_output)
        return x


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        pooler_output_1 = self.forward(input_ids_1, attention_mask_1)
        pooler_output_2 = self.forward(input_ids_2, attention_mask_2)

        if args.alt_sts_loss:
            x1 = self.cosine_head(pooler_output_1)
            x2 = self.cosine_head(pooler_output_2)
            x = self.cosine_similarity(x1, x2)
            # x = self.relu(x)
            # x = x * 5.0
            x = torch.sigmoid(x) * 5.0
        else:
            input_ids_cat = torch.cat((input_ids_1, input_ids_2), dim=1)
            attention_mask_cat = torch.cat((attention_mask_1, attention_mask_2), dim=1)
            x = self.forward(input_ids_cat, attention_mask_cat)
            x = self.sts_classifier(x) * 5.0
        return x


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


def train_multitask(args):
    '''Train MultitaskBERT.'''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(args.sst_train, args.para_train,
                                                                                      args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev, args.para_dev,
                                                                                args.sts_dev, split='train')

    # Load all data
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    # SST dataloader
    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn, num_workers=args.num_workers)

    # Paraphrase dataloader
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn, num_workers=args.num_workers)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)

    # STS dataloader
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn, num_workers=args.num_workers)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    pretrain_checkpoint, _, _, _ = load_model(args.transfer_path)

    model = MultitaskBERTTransfer(config, pretrain_checkpoint)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()

    sst_best_dev_acc = 0
    para_best_dev_acc = 0
    sts_best_dev_corr = -2

    # Define loss functions
    classification_loss_fn = nn.CrossEntropyLoss()
    paraphrase_loss_fn = nn.BCEWithLogitsLoss()
    similarity_loss_fn = nn.MSELoss()

    # Loss weighting
    sst_weight = args.sst_weight
    para_weight = args.para_weight
    sts_weight = args.sts_weight


    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        sst_train_loss = 0
        para_train_loss = 0
        sts_train_loss = 0
        num_batches = 0

        # Iterate through longest dataloader
        max_batches = max(len(sst_train_dataloader), len(para_train_dataloader), len(sts_train_dataloader))
        sst_iter = iter(sst_train_dataloader)
        para_iter = iter(para_train_dataloader)
        sts_iter = iter(sts_train_dataloader)

        progress_bar = tqdm(range(max_batches), desc=f'train-{epoch}', disable=TQDM_DISABLE)

        for _ in progress_bar:
            optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)
            sst_loss = torch.tensor(0.0, device=device)
            para_loss = torch.tensor(0.0, device=device)
            sts_loss = torch.tensor(0.0, device=device)

            # SST Batch
            try:
                batch = next(sst_iter)
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                           batch['attention_mask'], batch['labels'])
                
                if args.sgd_synonym_replacement:
                    b_ids = replace_synonyms(ids=b_ids, tokenizer=sst_train_data.tokenizer, p=args.sgd_synonym_replacement_p)

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)
                with autocast():
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss_classification = classification_loss_fn(logits, b_labels.view(-1)) / args.batch_size
                    sst_loss = sst_weight * loss_classification
                    loss += sst_loss
            except StopIteration:
                if args.round_robin:
                    sst_iter = iter(sst_train_dataloader)
                else:
                    pass

            # Paraphrase batch
            try:
                batch = next(para_iter)
                (b_ids1, b_mask1, b_ids2, b_mask2,
                 b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                              batch['token_ids_2'], batch['attention_mask_2'],
                              batch['labels'])

                if args.sgd_synonym_replacement:
                    b_ids1 = replace_synonyms(ids=b_ids1, tokenizer=para_train_data.tokenizer, p=args.sgd_synonym_replacement_p)
                    b_ids2 = replace_synonyms(ids=b_ids2, tokenizer=para_train_data.tokenizer, p=args.sgd_synonym_replacement_p)

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels = b_labels.to(device)

                with autocast():
                    logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                    loss_paraphrase = paraphrase_loss_fn(logits.squeeze(), b_labels.float()) / args.batch_size
                    para_loss = para_weight * loss_paraphrase
                    loss += para_loss
            except StopIteration:
                if args.round_robin:
                    para_iter = iter(para_train_dataloader)
                else:
                    pass

            # STS batch
            try:
                batch = next(sts_iter)
                (b_ids1, b_mask1, b_ids2, b_mask2,
                 b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                              batch['token_ids_2'], batch['attention_mask_2'],
                              batch['labels'])

                if args.sgd_synonym_replacement:
                    b_ids1 = replace_synonyms(ids=b_ids1, tokenizer=sts_train_data.tokenizer, p=args.sgd_synonym_replacement_p)
                    b_ids2 = replace_synonyms(ids=b_ids2, tokenizer=sts_train_data.tokenizer, p=args.sgd_synonym_replacement_p)

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)
                b_labels = b_labels.to(device)

                with autocast():
                    logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                    y_1s = torch.ones_like(b_labels).to(device)
                    loss_similarity = similarity_loss_fn(logits.squeeze(), b_labels.float()) / args.batch_size
                    sts_loss = sts_weight * loss_similarity
                    loss += sts_loss
            except StopIteration:
                if args.round_robin:
                    sts_iter = iter(sts_train_dataloader)
                else:
                    pass

            if loss.item() > 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_loss += loss.item()
            sst_train_loss += sst_loss.item()
            para_train_loss += para_loss.item()
            sts_train_loss += sts_loss.item()
            num_batches += 1

            progress_bar.set_postfix({
                "Total Loss": f"{loss.item():.4f}",
                "SST Loss": f"{sst_loss.item():.4f}",
                "Para Loss": f"{para_loss.item():.4f}",
                "STS Loss": f"{sts_loss.item():.4f}"
            })

        train_loss /= max(num_batches, 1)
        sst_train_loss /= max(num_batches, 1)
        para_train_loss /= max(num_batches, 1)
        sts_train_loss /= max(num_batches, 1)

        # Print average loss for the epoch
        tqdm.write(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {train_loss:.4f} (SST: {sst_train_loss:.4f}, Para: {para_train_loss:.4f}, STS: {sts_train_loss:.4f})")

    save_model(model, optimizer, args, config, args.filepath)


        # Evaluate on all tasks
        # if epoch == args.epochs-1:
        #     train_sentiment_accuracy, train_sst_y_pred, train_sst_sent_ids, \
        #     train_paraphrase_accuracy, train_para_y_pred, train_para_sent_ids, \
        #     train_sts_corr, train_sts_y_pred, train_sts_sent_ids = model_eval_multitask(sst_train_dataloader,
        #                                                                                 para_train_dataloader,
        #                                                                                 sts_train_dataloader, model, device)
        #
        #     dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
        #     dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
        #     dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
        #                                                                           para_dev_dataloader,
        #                                                                           sts_dev_dataloader, model, device)

        # if dev_sentiment_accuracy > sst_best_dev_acc:
        #     sst_best_dev_acc = dev_sentiment_accuracy
        #     save_model(model, optimizer, args, config, args.filepath)
        #
        # if dev_paraphrase_accuracy > para_best_dev_acc:
        #     para_best_dev_acc = dev_paraphrase_accuracy
        #     save_model(model, optimizer, args, config, args.filepath)
        #
        # if dev_sts_corr > sts_best_dev_corr:
        #     sts_best_dev_corr = dev_sts_corr
        #     save_model(model, optimizer, args, config, args.filepath)

        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, sst train acc :: {train_sentiment_accuracy :.3f}, "
        #   f"sst dev acc :: {dev_sentiment_accuracy :.3f}, para train acc :: {train_paraphrase_accuracy :.3f}, "
        #   f" para dev acc :: {dev_paraphrase_accuracy :.3f}, sts train corr :: {train_sts_corr :.3f}, "
        #   f" sts dev corr :: {dev_sts_corr :.3f}")


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        pretrain_checkpoint, _, _, _ = load_model(args.transfer_path)
        model = MultitaskBERTTransfer(config, pretrain_checkpoint)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels, para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test, args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels, para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                                  para_dev_dataloader,
                                                                                  sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
            model_eval_test_multitask(sst_test_dataloader,
                                      para_test_dataloader,
                                      sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    # Round-robin training
    parser.add_argument("--round_robin", action='store_true')

    # Cosine similarity
    parser.add_argument("--alt_sts_loss", action='store_true')

    # Task weights
    parser.add_argument("--sst_weight", type=float, default=1.0, help="Weight for the sentiment classification task")
    parser.add_argument("--para_weight", type=float, default=1.0, help="Weight for the paraphrase detection task")
    parser.add_argument("--sts_weight", type=float, default=1.0, help="Weight for the STS task")

    # SGD synonym replacement https://journals.agh.edu.pl/csci/article/view/3023/2181
    parser.add_argument("--sgd_synonym_replacement", action='store_true')
    parser.add_argument("--sgd_synonym_replacement_p", type=float, default=0.25, help="Probability of replacement for a given SGD minibatch")

    # Pretrained contrastive learning weights path
    parser.add_argument("--transfer_path", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt'  # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
