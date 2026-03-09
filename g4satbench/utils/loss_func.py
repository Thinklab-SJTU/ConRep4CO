import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F


# define cross entropy loss
def ce_loss(pred, target):
    target = F.one_hot(target, num_classes=2).float()
    return F.cross_entropy(pred, target)


# define binary cross entropy loss
def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target)


# define the loss for contrastive learning
def contrastive_loss(batched_sat_emb, batched_graph_emb, temperature=0.1):
    batch_size, emb_dim = batched_sat_emb.shape
    # divide each embedding by the 2-norm of the embedding
    normed_sat_emb = F.normalize(batched_sat_emb, p=2, dim=-1)
    normed_graph_emb = F.normalize(batched_graph_emb, p=2, dim=-1)

    # batched_sat_emb shape: [batch_size, emb_dim]
    # batched_graph_emb shape: [batch_size, emb_dim]
    # compute the similarities between all combinations of embeddings: sims shape: [batch_size * batch_size, 1]
    # repeat the embeddings to compute all combinations of embeddings
    normed_sat_emb = normed_sat_emb.unsqueeze(1).repeat(1, batched_graph_emb.shape[0], 1).view(-1, 1, emb_dim)
    normed_graph_emb = normed_graph_emb.unsqueeze(0).repeat(batched_sat_emb.shape[0], 1, 1).view(-1, emb_dim, 1)

    # normed_sat_emb shape: [batch_size * batch_size, 1, emb_dim]
    # normed_graph_emb shape: [batch_size * batch_size, emb_dim, 1]
    # calculate the cosine similarity between the embeddings by matrix multiplication
    sims = torch.bmm(normed_sat_emb, normed_graph_emb).squeeze(-1).squeeze(-1)
    sims /= temperature
    sims = torch.exp(sims)
    sims = sims.view(batch_size, batch_size)

    # compute the loss
    return_loss = 0
    for idx in range(batch_size):
        pos_sim = sims[idx, idx]
        neg_sim = torch.sum(sims[idx]) - pos_sim
        sym_neg_sim = torch.sum(sims[:, idx]) - pos_sim
        return_loss += -torch.log(pos_sim / neg_sim)
        return_loss += -torch.log(pos_sim / sym_neg_sim)

    return return_loss / (batch_size * 2)


    # use softmax and cross entropy loss to compute the loss
    # sims = torch.bmm(normed_sat_emb, normed_graph_emb).squeeze(-1).squeeze(-1)
    # sims = sims.view(batch_size, batch_size)
    # labels = torch.arange(batch_size).to(sims.device)
    # loss = F.cross_entropy(sims, labels)
    # sym_loss = F.cross_entropy(sims.T, labels)

    # return (loss + sym_loss) / 2
