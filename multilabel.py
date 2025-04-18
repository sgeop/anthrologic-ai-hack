from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

# import pandas as pd
#
# mlb = MultiLabelBinarizer()
#
#
# def encode_labels(
#     tickets_df: pd.DataFrame, text_col="text", labels_col="tags"
# ) -> pd.DataFrame:
#     labels = mlb.fit_transform(tickets_df[labels_col])
#     df = pd.concat([tickets_df[[text_col]], pd.DataFrame(labels)], axis=1)
#     df.columns = [text_col] + list(mlb.classes_)
#     return df


# vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_rHHnge=(1, 2))
