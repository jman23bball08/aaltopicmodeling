from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
import emoji
import torch
import pandas as pd
import pyro
import math
import os
from pyro.infer import SVI, TraceMeanField_ELBO
from tqdm import trange
from sklearn.feature_extraction.text import CountVectorizer
from prodlda import ProdLDA



posts = []
with open("emotion_exp/emotion_explanations_train.txt", "r") as f:
    for line in f:
        posts.append(line.strip())


def process(text):
    #tokenize text
    tokens = word_tokenize(text)

    punc = string.punctuation
    stop_words = stopwords.words("english")
    all_emojis = list(emoji.unicode_codes._EMOJI_UNICODE.values())

    #remove stopwords, punctuation, numbers, and turn to lowercase
    tokens = [token.lower() for token in tokens if token not in stop_words and token not in punc and token not in all_emojis and token.isalpha()]


    return tokens

#apply to each post
processed_posts = [process(post) for post in posts]
#print(processed_posts)

vectorizer = CountVectorizer(max_df=0.5, min_df=20, stop_words='english')
docs = torch.from_numpy(vectorizer.fit_transform([" ".join(post) for post in processed_posts]).toarray())

vocab = pd.DataFrame(columns=['word', 'index'])
vocab['word'] = vectorizer.get_feature_names_out()
vocab['index'] = vocab.index

print('Dictionary size: %d' % len(vocab))
print('Corpus size: {}'.format(docs.shape))

#print(vocab)

assert pyro.__version__.startswith('1.8.5')
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ


    


# setting global variables
seed = 0
torch.manual_seed(seed)
pyro.set_rng_seed(seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for incr in range(1,21):
#change these params to adjust model
    num_topics = incr if not smoke_test else 3

    docs = docs.float().to(device)
    batch_size = 30 #8801 mod 32 = 1 causes issues
    learning_rate = 1e-3
    num_epochs = 50 if not smoke_test else 1

    # training
    pyro.clear_param_store()

    prodLDA = ProdLDA(
        vocab_size=docs.shape[1],
        num_topics=num_topics,
        hidden=100 if not smoke_test else 10,
        dropout=0.2
    )
    prodLDA.to(device)

    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=TraceMeanField_ELBO())
    num_batches = int(math.ceil(docs.shape[0] / batch_size)) if not smoke_test else 1

    bar = trange(num_epochs)
    for epoch in bar:
        running_loss = 0.0
        for i in range(num_batches):
            batch_docs = docs[i * batch_size:(i + 1) * batch_size, :]
            loss = svi.step(batch_docs)
            running_loss += loss / batch_docs.size(0)

        bar.set_postfix(epoch_loss='{:.2e}'.format(running_loss))


    import dill

    save_model = "emotion_exp/models/emotion_exp_prodlda_model_" + str(incr)
    print(save_model)
    # Save the trained model
    with open(save_model, "wb") as f:
        dill.dump(prodLDA, f)