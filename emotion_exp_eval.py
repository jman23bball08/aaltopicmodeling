import dill
import torch
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
import os
import pyro
import topic_metrics
import emoji
import evaluation_metrics

assert pyro.__version__.startswith('1.8.5')
# Enable smoke test - run the notebook cells on CI.
smoke_test = 'CI' in os.environ

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

    #remove stopwords, punctuation, and turn to lowercase
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
#print(vocab)

# max_qual = -1
# max_ind = 1

# for incr in range(21,41):
#     load_model = "emotion_exp/models/emotion_exp_prodlda_model_" + str(incr)
#     with open(load_model, "rb") as f:
#         prodLDA = dill.load(f)


#     beta_matrix = prodLDA.beta().detach().numpy()
#     num_terms = 5
#     topic_terms = []

#     for i in range(beta_matrix.shape[0]):
#         # Get the indices of the top num_terms terms for this topic
#         top_term_indices = beta_matrix[i].argsort()[-num_terms:]

#         # Append the indices to the list
#         topic_terms.append(list(top_term_indices))

#     from scipy import sparse

#     numpy_array = docs.numpy()
#     docs_sparse = sparse.csr_matrix(numpy_array)

#     diversity, coherence, quality, all_cohs = topic_metrics.topic_eval(ref_bows=docs_sparse, all_topic_ixs=topic_terms)
#     #print(coherence, diversity, quality, all_cohs)
#     text_results = "Results for {}: \n".format(incr)
#     text_results += "coherence: {} \ndiversity: {} \nquality: {} \n\n".format(coherence,diversity,quality)
#     if quality > max_qual:
#         max_qual = coherence
#         max_ind = incr
#     print(incr)
#     print("coherence: ", coherence)
#     print("diversity: ", diversity)
#     print("quality: ", quality)
#     print("\n")

#     topic_word_distributions = prodLDA.beta().detach().numpy()

#     num_topics = topic_word_distributions.shape[0]
#     num_top_words = 5  # Number of top words to retrieve for each topic

#     top_words_per_topic = []
#     for topic in range(num_topics):
#         word_probs = topic_word_distributions[topic]
#         top_word_indices = word_probs.argsort()[::-1][:num_top_words]
#         top_words = [vocab.loc[vocab['index'] == word_index, 'word'].values[0] for word_index in top_word_indices]
#         top_words_per_topic.append(top_words)

#     #print(top_words_per_topic)
#     text_results += str(top_words_per_topic)

#     file_name = "emotion_exp/evals/emotion_results_{}.txt".format(incr)
#     text_file = open(file_name, "w")
#     text_file.write(text_results)
#     text_file.close()

# print(max_ind, max_qual)

load_model = "emotion_exp/models/emotion_exp_prodlda_model_23"
with open(load_model, "rb") as f:
    prodLDA = dill.load(f)

# plot word cloud

# def plot_word_cloud(b, ax, v, n):
#     sorted_, indices = torch.sort(b, descending=True)
#     df = pd.DataFrame(indices[:100].numpy(), columns=['index'])
#     words = pd.merge(df, vocab[['index', 'word']],
#                         how='left', on='index')['word'].values.tolist()
#     #sizes = (sorted_[:100] * 1000).int().numpy().tolist()
#     sizes = (abs(sorted_[:100]) * 1000).int().numpy().tolist()
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

# print(max_ind, max_coh)

docs = docs.float()
print(docs.shape)
ppx = evaluation_metrics.compute_perplexity(prodLDA, docs)
print(ppx)

overall_topic_distribution = evaluation_metrics.get_overall_topic_distribution(prodLDA, docs)
print("Overall Topic Distribution:")
print(overall_topic_distribution)

all_topic_distributions = evaluation_metrics.encode_documents(prodLDA, docs)
print("Encoded Topic Distributions for Each Document:")
print(all_topic_distributions)