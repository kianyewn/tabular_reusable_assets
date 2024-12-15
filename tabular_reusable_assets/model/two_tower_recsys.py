import itertools
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

from tabular_reusable_assets.datasets import DummyRecsysDataset


class NegativeSampler:
    """https://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
    The way this selection is implemented in the C code is interesting.
    They have a large array with 100M elements (which they refer to as the unigram table).
    They fill this table with the index of each word in the vocabulary multiple times,
    and the number of times a word's index appears in the table is given by P(wi)* table_size.
    Then, to actually select a negative sample, you just generate a random integer between 0 and 100M,
    and use the word at that index in the table.
    Since the higher probability words occur more times in the table,
    you're more likely to pick those.
    """

    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.table = self.create_unigram_table()

    def create_unigram_table(self):
        word_freq = pd.DataFrame(
            Counter(self.dataset["item_id"]).items(), columns=["item_id", "count"]
        )
        word_freq["freq"] = word_freq["count"] / word_freq["count"].sum()
        word_freq["freq_scaled"] = np.power(word_freq["freq"], 3 / 4)
        word_freq["prob"] = word_freq["freq_scaled"] / word_freq["freq_scaled"].sum()
        word_freq["prob"] = word_freq["prob"]
        freq_table = dict(word_freq[["item_id", "prob"]].values)

        unigram_table_size = 5000
        unigram_table = list(
            itertools.chain(
                *[
                    [int(w)] * round(p * unigram_table_size)
                    for w, p in freq_table.items()
                ]
            )
        )
        return unigram_table

    def sample(self, num_samples):
        """Generate item_ids for negative samples"""
        candidate_samples = random.sample(self.table, k=num_samples)
        return candidate_samples


class UnigramTable:
    def __init__(self, dataset: pd.DataFrame, table_size=100000, power=0.75):
        self.dataset = dataset
        self.table_size = table_size
        self.power = power
        self.unigram_table = self.create_unigram_table()

    def create_unigram_table(self):
        vocab = Counter(self.dataset["item_id"])
        self.table = np.zeros(self.table_size, dtype=np.int32)
        total_freq = sum([freq**self.power for freq in vocab.values()])

        # fill the table with item_ids
        i = 0
        for item_id, freq in vocab.items():
            prob = (freq**self.power) / total_freq
            count = int(prob * self.table_size)
            self.table[i : i + count] = item_id
            i += count

            if i >= self.table_size:
                break

        # shuffle table for randomness
        np.random.shuffle(self.table)

    def sample(self, num_samples):
        indices = random.sample(population=self.table, k=num_samples)
        return self.unigram_table[indices]


class UserInteractions:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.user_to_pos_interactions = self.get_positive_interactions_dict()

    def get_positive_interactions_dict(self):
        user_id_pos_dict = (
            self.dataset.groupby(["user_id"])["item_id"].unique().apply(set).to_dict()
        )
        return user_id_pos_dict


class RetrievalDataset(torch.utils.data.Dataset):
    """
    Dataset for retrieval task
    - Output : user_id, pos_item, neg_items
    """

    def __init__(
        self,
        dataset: torch.Tensor,
        negative_sampler: NegativeSampler,
        neg_samples: int,
        user_historical_interactions: UserInteractions,
    ):
        self.dataset = dataset
        self.user_ids = dataset.iloc[:, 0]
        self.item_ids = dataset.iloc[:, 1]
        self.target = dataset.iloc[:, 2]
        self.negative_sampler = negative_sampler
        self.neg_samples = neg_samples
        self.user_historical_interactions = user_historical_interactions

    def __len__(self):
        return len(self.dataset)

    def generate_negative_samples(self, user_id, positive_item):
        user_pos_interactions = (
            self.user_historical_interactions.user_to_pos_interactions[user_id]
        )

        candidate_samples = []
        while len(candidate_samples) < self.neg_samples:
            negative_samples = self.negative_sampler.sample(self.neg_samples)
            for neg_sample in negative_samples:
                if (
                    len(candidate_samples) < self.neg_samples
                    and neg_sample not in user_pos_interactions
                ):
                    candidate_samples.append(neg_sample)
        return torch.tensor(candidate_samples)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]  # (B,)
        pos_item_id = self.item_ids[idx]  # (B,)
        negative_samples = self.generate_negative_samples(
            user_id=user_id, positive_item=pos_item_id
        )  # (B, neg_samples)
        # posiitve sample is always at index 0
        labels = torch.tensor([0], dtype=torch.long)
        return user_id, pos_item_id, negative_samples, labels


class TwoTowerRetrievalModel(nn.Module):
    def __init__(self, n_users, n_items, embd_dim, hidden_mults=[2, 3, 4]):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users + 1, embd_dim)
        self.item_embedding = nn.Embedding(n_items + 1, embd_dim)
        self.hidden_mults = hidden_mults
        hidden_dims = [embd_dim] + [embd_dim * h for h in hidden_mults] + [embd_dim]

        self.user_tower = nn.ModuleList()
        self.item_tower = nn.ModuleList()
        for idx, (in_ft, out_ft) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.user_tower.append(nn.Linear(in_ft, out_ft))
            self.item_tower.append(nn.Linear(in_ft, out_ft))
            if idx != len(hidden_dims) - 2:
                self.user_tower.append(nn.ReLU())
                self.item_tower.append(nn.ReLU())
        self.user_tower = nn.Sequential(*self.user_tower)
        self.item_tower = nn.Sequential(*self.item_tower)

    def forward(self, user, pos_item, neg_items, labels=None):
        user_emb = self.user_embedding(user)  # (B, embd_dim)
        pos_item_emb = self.item_embedding(pos_item)  # (B, embd_dim)
        neg_items_emb = self.item_embedding(neg_items)  # (B, neg_samples, embd_dim)

        # apply non linearity
        user_emb = self.user_tower(user_emb)  # (B, embd_dim)
        pos_item_emb = self.item_tower(pos_item_emb)  # (B, embd_dim)
        neg_items_emb = self.item_tower(neg_items_emb)  # (B, neg_samples, embd_dim)

        # compute similarity
        pos_similarities = torch.sum(
            user_emb * pos_item_emb, dim=-1
        )  # (B, emb_dim), (B, emd_bim) -> (B,)
        neg_similarities = torch.matmul(
            user_emb.unsqueeze(1), neg_items_emb.transpose(-1, -2)
        ).squeeze()  # (B, 1, embd_dim) , (B, embd_dim, neg_samples) -> (B, neg_samples)

        neg_similarities = torch.sum(
            user_emb.unsqueeze(1) * neg_items_emb, dim=-1
        )  # (B, 1, embd_dim) , (B, neg_samples, embd_dim) -> (B, neg_samples)

        # print(pos_similarities.shape, neg_similarities.shape)
        similarities = torch.cat(
            [pos_similarities.unsqueeze(-1), neg_similarities], dim=-1
        )  # (B, 1 + neg_samples)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(similarities, labels.reshape(-1))
        return similarities, loss

    def get_user_embedding(self, user_id):
        return self.user_tower(self.user_embedding(user_id))

    def get_item_embedding(self, item_id):
        return self.item_tower(self.item_embedding(item_id))

    def get_item_embeddings(self):
        return self.item_tower(self.item_embedding.weight)


def train_one_epoch_retrieval_model(model, data_loader, optimizer, device, epoch_num):
    model.train()
    losses = []
    for user, pos_item, neg_items, labels in data_loader:
        user, pos_item, neg_items, labels = (
            user.to(device),
            pos_item.to(device),
            neg_items.to(device),
            labels.to(device),
        )
        similarities, loss = model(user, pos_item, neg_items, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def train_retrieval_model(model, data_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        losses = train_one_epoch_retrieval_model(
            model, data_loader, optimizer, device, epoch
        )
        if (epoch == 0) or ((epoch + 1) % 2 == 0):
            avg_loss = sum(losses) / len(losses)
            print(f"Epoch {epoch+1} Loss: {avg_loss}")


class RetrievalIndex:
    def __init__(self, stored_embeddings: torch.Tensor):
        self.stored_embeddings = stored_embeddings
        # self.index = faiss.IndexHNSWFlat(stored_embeddings.shape[1])
        # self.index.add(stored_embeddings.numpy())

    @torch.no_grad()
    def nearest_neighbors(self, query_embedding, k=10):
        # distances, indices = self.index.search(query_embedding.numpy(), k)
        query_embedding = torch.tensor(query_embedding)
        distances = torch.square(
            query_embedding.unsqueeze(1) - self.stored_embeddings.unsqueeze(0)
        ).sum(
            dim=-1
        )  # (n_users, 1, embd_dim), (1, n_items, embd_dim) -> (n_users, n_items)

        top_k_values, top_k_indices = torch.topk(distances, k, dim=-1, largest=False)
        return top_k_values, top_k_indices

    def search(self, query_embedding, k=10):
        distances, indices = self.nearest_neighbors(query_embedding, k)
        return indices

    # def search(self, query_embedding, k=10):
    # distances, indices = self.index.search(query_embedding.numpy(), k)
    # return indices


class ReRankingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: torch.Tensor,
        user_historical_interactions: UserInteractions,
        retrieval_index: RetrievalIndex,
        k=10,
        cache=dict(),
    ):
        self.dataset = dataset
        self.users = dataset.iloc[:, 0]
        self.items = dataset.iloc[:, 1]
        self.index = retrieval_index
        self.k = k
        self.cache = cache
        self.user_historical_interactions = user_historical_interactions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        user_id = self.users[idx]  # (B,)
        pos_item_id = self.items[idx]  # (B,)

        # get candidate items
        if user_id not in self.cache:
            candidate_items = self.index.search(user_id.reshape(-1), k=self.k).squeeze()
            self.cache[user_id] = candidate_items  # (B, self.k)
        else:
            candidate_items = self.cache[user_id]  # (B, self.k)

        # add positive item to the end of the list
        if pos_item_id not in candidate_items.tolist():
            # print('pos item not in candidate items')
            candidate_items[-1] = pos_item_id

        # create labels
        user_pos_interactions = (
            self.user_historical_interactions.user_to_pos_interactions[user_id]
        )
        labels = torch.tensor(
            [1 if cand.item() in user_pos_interactions else 0 for cand in candidate_items],
            dtype=torch.float,
        )
        return user_id, candidate_items, labels


class RerankerModel(nn.Module):
    def __init__(self, n_users, n_items, embd_dim, hidden_mults=[2, 3, 4]):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users + 1, embd_dim)
        self.item_embedding = nn.Embedding(n_items + 1, embd_dim)
        self.hidden_mults = hidden_mults
        hidden_dims = [embd_dim * 2] + [embd_dim * h for h in hidden_mults] + [1]

        self.deep = nn.ModuleList()

        for idx, (in_ft, out_ft) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            self.deep.append(nn.Linear(in_ft, out_ft))
            if idx != len(hidden_dims) - 2:
                self.deep.append(nn.ReLU())

        self.deep = nn.Sequential(*self.deep)

        self.deep_layer = nn.Sequential(
            nn.Linear(embd_dim * 2, embd_dim),
            nn.ReLU(),
            nn.Linear(embd_dim, 1),
            nn.Sigmoid(),
        )
        self.wide_layer = nn.Linear(embd_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, candidate_items, labels):
        user_emb = self.user_embedding(user_id)  # (B,emb_dim)
        candidate_items_emb = self.item_embedding(candidate_items)  # (B, k, embd_dim)
        # print(user_emb.shape, candidate_items_emb.shape)
        x = torch.cat(
            [
                user_emb.unsqueeze(1).repeat(1, candidate_items_emb.shape[1], 1),
                candidate_items_emb,
            ],
            dim=-1,
        )  # (B, 1, embd_dim), (B, k, embd_dim) -> (B, k, 2 * embd_dim)
        deep_output = self.deep(x)  # (B, k, 1)
        wide_output = self.wide_layer(x)  # (B, k, 1)
        output = deep_output + wide_output  # (B, k, 1)
        output = self.sigmoid(output).squeeze()
        loss = None
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy(output, labels)
        return output, loss


def train_one_epoch_reranker_model(model, data_loader, optimizer, device, epoch):
    model.train()
    losses = []
    for user_id, candidate_items, labels in data_loader:
        user_id, candidate_items, labels = (
            user_id.to(device),
            candidate_items.to(device),
            labels.to(device),
        )
        output, loss = model(user_id, candidate_items, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def train_reranker_model(model, data_loader, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        losses = train_one_epoch_reranker_model(
            model, data_loader, optimizer, device, epoch
        )
        if (epoch == 0) or ((epoch + 1) % 2 == 0):
            avg_loss = sum(losses) / len(losses)
            print(f"Epoch {epoch+1} Loss: {avg_loss}")


if __name__ == "__main__":
    n_users, n_items, dataset, target = DummyRecsysDataset.generate_dataset()
    negative_sampler = NegativeSampler(dataset)
    user_historical_interactions = UserInteractions(dataset=dataset)

    retrieval_dataset = RetrievalDataset(
        dataset,
        negative_sampler=negative_sampler,
        neg_samples=5,
        user_historical_interactions=user_historical_interactions,
    )

    retrieval_dataloader = torch.utils.data.DataLoader(
        dataset=retrieval_dataset, batch_size=50, shuffle=True
    )
    sample_retrieval_dataloader = next(iter(retrieval_dataloader))

    print(
        sample_retrieval_dataloader[0].shape,
        sample_retrieval_dataloader[1].shape,
        sample_retrieval_dataloader[2].shape,
        sample_retrieval_dataloader[3].shape,
    )

    # Retrievalmodel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retrieval_model = TwoTowerRetrievalModel(
        n_users, n_items, embd_dim=32, hidden_mults=[2, 3, 4]
    ).to(device)

    optimizer = AdamW(retrieval_model.parameters(), lr=0.001)
    train_retrieval_model(
        retrieval_model, retrieval_dataloader, optimizer, device, num_epochs=1
    )

    # Re ranking
    retrieval_index = RetrievalIndex(
        stored_embeddings=retrieval_model.get_item_embeddings()
    )
    reranking_dataset = ReRankingDataset(
        dataset=dataset,
        user_historical_interactions=user_historical_interactions,
        retrieval_index=retrieval_index,
        k=10,
    )
    reranking_dataloader = torch.utils.data.DataLoader(
        dataset=reranking_dataset, batch_size=50, shuffle=True
    )
    sample_reranking_dataloader = next(iter(reranking_dataloader))
    print(
        sample_reranking_dataloader[0].shape,
        sample_reranking_dataloader[1].shape,
        sample_reranking_dataloader[2].shape,
        sample_reranking_dataloader[2],
    )

    reranker = RerankerModel(n_users, n_items, embd_dim=32, hidden_mults=[2, 3, 4]).to(
        device
    )
    optimizer = AdamW(reranker.parameters(), lr=0.001)
    train_reranker_model(
        reranker, reranking_dataloader, optimizer, device, num_epochs=3
    )
