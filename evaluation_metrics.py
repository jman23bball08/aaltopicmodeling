import torch
import pyro
import pyro.distributions as dist
import torch.nn.functional as F
import math
import pandas as pd


def get_overall_topic_distribution(model, data):
    all_topic_distributions = []
    for docs in data:
        docs = docs.unsqueeze(0)
        logtheta_loc, logtheta_scale = model.encoder(docs)
        topic_distributions = F.softmax(logtheta_loc, -1)
        all_topic_distributions.append(topic_distributions)
    all_topic_distributions = torch.cat(all_topic_distributions, dim=0)
    overall_topic_distribution = all_topic_distributions.mean(dim=0)
    return overall_topic_distribution

def encode_documents(model, data):
    all_topic_distributions = []
    for docs in data:
        docs = docs.unsqueeze(0)
        logtheta_loc, logtheta_scale = model.encoder(docs)
        topic_distributions = F.softmax(logtheta_loc, -1)
        all_topic_distributions.append(topic_distributions)
    all_topic_distributions = torch.cat(all_topic_distributions, dim=0)

    # Map the topic probabilities to actual words in the vocabulary
    # vocab = pd.DataFrame(columns=['word', 'index'])
    # topic_words = []
    # for topic_probs in all_topic_distributions:
    #     topic_indices = torch.argsort(topic_probs, descending=True)
    #     topic_words.append([vocab[word_idx.item()] for word_idx in topic_indices])

    return all_topic_distributions #, topic_words


def compute_perplexity(model, data, num_samples=10):
    model.eval()
    total_cross_ent = 0.0
    total_words = 0
    for docs in data:
        # Skip if the document is empty
        if docs.sum() == 0:
            continue

        docs = docs.unsqueeze(0)

        # a) Predict the parameters of the topic distribution of the document
        logtheta_loc, logtheta_scale = model.encoder(docs)

        # b) Sample multiple theta distributions
        logtheta = dist.Normal(logtheta_loc, logtheta_scale).sample((num_samples,))
        theta = F.softmax(logtheta, -1)

        # # c) Decode the document from the theta distributions in part b
        # count_param = model.decoder(theta)

        # Reshape theta to be 1D
        theta_1d = theta.view(num_samples, -1)
        count_param_1d = model.decoder(theta_1d)
        # Reshape count_param back to its original shape
        count_param = count_param_1d.view(num_samples, 1, -1)

        # d) Softmax the decoded document and average over the resulting samples
        doc_probs = F.softmax(count_param, dim=-1).mean(dim=0)

        # e) Calculate the Cross Entropy loss between the original document BOW representation and the predicted reconstructions
        total_count = int(docs.sum(-1).max())
        # print("docs:", docs)
        # print("docs.sum(-1):", docs.sum(-1))
        # print("docs.sum(-1).max():", docs.sum(-1).max())
        # print("total_count:", total_count)

        #cross_entropy = dist.Multinomial(total_count, doc_probs).sample()
        cross_entropy = -torch.sum(docs * torch.log(doc_probs + 1e-10))
        total_cross_ent += cross_entropy.sum().item()
        total_words += docs.sum().item()

    # Compute perplexity
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(total_cross_ent, total_words)
    perplexity = math.exp(-total_cross_ent / total_words)
    return perplexity