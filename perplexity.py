import torch
import pyro
import pyro.distributions as dist
import torch.nn.functional as F


def compute_perplexity(model, docs, num_samples=10):
    pyro.module("decoder", model.decoder)
    with pyro.plate("documents", docs.shape[0]):

        # a) Predict the parameters of the topic distribution of the document
        logtheta_loc = docs.new_zeros((docs.shape[0], model.num_topics))
        logtheta_scale = docs.new_ones((docs.shape[0], model.num_topics))

        # b) Sample multiple theta distributions
        logtheta = dist.Normal(logtheta_loc, logtheta_scale).sample((num_samples,))
        theta = F.softmax(logtheta, -1)

        # c) Decode the document from the theta distributions in part b
        count_param = model.decoder(theta)

        # d) Softmax the decoded document and average over the resulting samples
        doc_probs = F.softmax(count_param, dim=-1).mean(dim=0)

        # e) Calculate the Cross Entropy loss between the original document BOW representation and the predicted reconstructions
        total_count = int(docs.sum(-1).max())
        pyro.sample(
            'obs',
            dist.Multinomial(total_count, doc_probs),
            obs=docs
        )