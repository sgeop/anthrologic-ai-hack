# from bertopic import BERTopic
# from bertopic.vectorizers import ClassTfidfTransformer
# from sklearn.feature_extraction.text import CountVectorizer
# from sentence_transformers import SentenceTransformer
# from hdbscan import HDBSCAN
#
#
# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
# hdbscan_model = HDBSCAN(
#     min_cluster_size=15,
#     metric="euclidean",
#     cluster_selection_method="eom",
#     prediction_data=True,
# )
# vectorizer_model = CountVectorizer(
#     ngram_range=(1, 3), stop_words="english", max_features=5000
# )
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
#
# topic_model = BERTopic(
#     embedding_model=sentence_model,
#     hdbscan_model=hdbscan_model,
#     vectorizer_model=vectorizer_model,
# )
