import os
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import pyro.distributions as dist
import topic_metrics 

news = fetch_20newsgroups(subset='all')
vectorizer = CountVectorizer(max_df=0.5, min_df=20, stop_words='english')
docs = torch.from_numpy(vectorizer.fit_transform(news['data']).toarray())

vocab = pd.DataFrame(columns=['word', 'index'])
vocab['word'] = vectorizer.get_feature_names_out()
vocab['index'] = vocab.index




assert pyro.__version__.startswith('1.8.5')
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

class Encoder(nn.Module):
    # Base class for the encoder net, used in the guide
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)  # to avoid component collapse
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fcmu = nn.Linear(hidden, num_topics)
        self.fclv = nn.Linear(hidden, num_topics)
        # NB: here we set `affine=False` to reduce the number of learning parameters
        # See https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        # for the effect of this flag in BatchNorm1d
        self.bnmu = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse
        self.bnlv = nn.BatchNorm1d(num_topics, affine=False)  # to avoid component collapse

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        # μ and Σ are the outputs
        logtheta_loc = self.bnmu(self.fcmu(h))
        logtheta_logvar = self.bnlv(self.fclv(h))
        logtheta_scale = (0.5 * logtheta_logvar).exp()  # Enforces positivity
        return logtheta_loc, logtheta_scale


class Decoder(nn.Module):
    # Base class for the decoder net, used in the model
    def __init__(self, vocab_size, num_topics, dropout):
        super().__init__()
        self.beta = nn.Linear(num_topics, vocab_size, bias=False)
        self.bn = nn.BatchNorm1d(vocab_size, affine=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        inputs = self.drop(inputs)
        # the output is σ(βθ)
        return F.softmax(self.bn(self.beta(inputs)), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.encoder = Encoder(vocab_size, num_topics, hidden, dropout)
        self.decoder = Decoder(vocab_size, num_topics, dropout)

    def model(self, docs):
        pyro.module("decoder", self.decoder)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution
            logtheta_loc = docs.new_zeros((docs.shape[0], self.num_topics))
            logtheta_scale = docs.new_ones((docs.shape[0], self.num_topics))
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))
            theta = F.softmax(logtheta, -1)

            # conditional distribution of 𝑤𝑛 is defined as
            # 𝑤𝑛|𝛽,𝜃 ~ Categorical(𝜎(𝛽𝜃))
            count_param = self.decoder(theta)
            # Currently, PyTorch Multinomial requires `total_count` to be homogeneous.
            # Because the numbers of words across documents can vary,
            # we will use the maximum count accross documents here.
            # This does not affect the result because Multinomial.log_prob does
            # not require `total_count` to evaluate the log probability.
            total_count = int(docs.sum(-1).max())
            pyro.sample(
                'obs',
                dist.Multinomial(total_count, count_param),
                obs=docs
            )

    def guide(self, docs):
        pyro.module("encoder", self.encoder)
        with pyro.plate("documents", docs.shape[0]):
            # Dirichlet prior 𝑝(𝜃|𝛼) is replaced by a logistic-normal distribution,
            # where μ and Σ are the encoder network outputs
            logtheta_loc, logtheta_scale = self.encoder(docs)
            logtheta = pyro.sample(
                "logtheta", dist.Normal(logtheta_loc, logtheta_scale).to_event(1))

    def beta(self):
        # beta matrix elements are the weights of the FC layer on the decoder
        return self.decoder.beta.weight.cpu().detach().T
    

if __name__ == '__main__':
    # # setting global variables
    # seed = 0
    # torch.manual_seed(seed)
    # pyro.set_rng_seed(seed)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # num_topics = 20 if not smoke_test else 3
    # docs = docs.float().to(device)
    # batch_size = 32
    # learning_rate = 1e-3
    # num_epochs = 50 if not smoke_test else 1

    # # training
    # pyro.clear_param_store()

    # prodLDA = ProdLDA(
    #     vocab_size=docs.shape[1],
    #     num_topics=num_topics,
    #     hidden=100 if not smoke_test else 10,
    #     dropout=0.2
    # )
    # prodLDA.to(device)

    # optimizer = pyro.optim.Adam({"lr": learning_rate})
    # svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    # num_batches = int(math.ceil(docs.shape[0] / batch_size)) if not smoke_test else 1

    # bar = trange(num_epochs)
    # for epoch in bar:
    #     running_loss = 0.0
    #     for i in range(num_batches):
    #         batch_docs = docs[i * batch_size:(i + 1) * batch_size, :]
    #         loss = svi.step(batch_docs)
    #         running_loss += loss / batch_docs.size(0)

    #     bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))


    import dill

    # # Save the trained model
    # with open("prodlda_model.pkl", "wb") as f:
    #     dill.dump(prodLDA, f)

    with open("prodlda_model.pkl", "rb") as f:
        prodLDA = dill.load(f)
        
  
    beta_matrix = prodLDA.beta().detach().numpy()
    num_terms = 20
    topic_terms = []

    for i in range(beta_matrix.shape[0]):
        # Get the indices of the top num_terms terms for this topic
        top_term_indices = beta_matrix[i].argsort()[-num_terms:]

        # Append the indices to the list
        topic_terms.append(list(top_term_indices))

    from scipy import sparse

    numpy_array = docs.numpy()
    docs_sparse = sparse.csr_matrix(numpy_array)

    coherence, diversity, quality, all_cohs = topic_metrics.topic_eval(ref_bows=docs_sparse, all_topic_ixs=topic_terms)
    print(coherence, diversity, quality, all_cohs)

    # plot word cloud

    # def plot_word_cloud(b, ax, v, n):
    #     sorted_, indices = torch.sort(b, descending=True)
    #     df = pd.DataFrame(indices[:100].numpy(), columns=['index'])
    #     words = pd.merge(df, vocab[['index', 'word']],
    #                      how='left', on='index')['word'].values.tolist()
    #     sizes = (sorted_[:100] * 1000).int().numpy().tolist()
    #     freqs = {words[i]: sizes[i] for i in range(len(words))}
    #     wc = WordCloud(background_color="white", width=800, height=500)
    #     wc = wc.generate_from_frequencies(freqs)
    #     ax.set_title('Topic %d' % (n + 1))
    #     ax.imshow(wc, interpolation='bilinear')
    #     ax.axis("off")

    # if not smoke_test:
    #     import matplotlib.pyplot as plt
    #     from wordcloud import WordCloud
    #     beta = prodLDA.beta()
    #     fig, axs = plt.subplots(7, 3, figsize=(14, 24))
    #     for n in range(beta.shape[0]):
    #         i, j = divmod(n, 3)
    #         plot_word_cloud(beta[n], axs[i, j], vocab, n)
    #     axs[-1, -1].axis('off');

    #     plt.show()