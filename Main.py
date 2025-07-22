# ✅ Step 1: Install necessary libraries
!pip install -q numpy scikit-learn matplotlib nltk wordcloud pyLDAvis seaborn

# ✅ Step 2: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pyLDAvis
from wordcloud import WordCloud
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans

# ✅ Step 3: Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

# ✅ Step 4: Load dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
data = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})

# ✅ Step 5: Text Preprocessing
def preprocess(text):
    import re
    text = re.sub(r'\W+', ' ', text.lower())
    return ' '.join([word for word in text.split() if word not in stop_words and len(word) > 2])

data['clean_text'] = data['text'].apply(preprocess)

# ✅ Step 6: TF-IDF Vectorization
tfidf = TfidfVectorizer(max_df=0.9, min_df=10, stop_words='english')
X_tfidf = tfidf.fit_transform(data['clean_text'])

# ✅ Step 7: KMeans Clustering
k = 20
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_tfidf)
data['cluster'] = labels

# ✅ Step 8: PCA Visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(10,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette='tab20', legend='full', s=10)
plt.title("K-Means Clusters (PCA projection)")
plt.show()

# ✅ Step 9: WordCloud for each cluster
for cluster_num in range(k):
    cluster_text = " ".join(data[data.cluster == cluster_num]['clean_text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cluster_text)
    plt.figure(figsize=(10,4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for Cluster {cluster_num}")
    plt.show()

# ✅ Step 10: LDA Topic Modeling
count_vectorizer = CountVectorizer(max_df=0.9, min_df=10, stop_words='english')
X_counts = count_vectorizer.fit_transform(data['clean_text'])

lda = LatentDirichletAllocation(n_components=10, max_iter=10, learning_method='online', random_state=42)
lda.fit(X_counts)

# ✅ Step 11: Show top words for each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        print()

display_topics(lda, count_vectorizer.get_feature_names_out(), 10)

