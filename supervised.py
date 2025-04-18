import polars as pl
from typing import List

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
import datasets

import loader

THRESHOLD_NORM = 0.01

tickets = loader.get_tickets()
labels = loader.indexed_labels(tickets, threshold_norm=THRESHOLD_NORM)
text_labels = labels.select("label_text").to_series().to_list()
data = loader.to_dataset(tickets, threshold_norm=THRESHOLD_NORM).train_test_split(
    test_size=0.02
)

# docs = data["train"]["text"]
# y = data["train"]["label"]
#
# topics, probs = topic_model.fit_transform(docs, y=y)
#
# mappings = topic_model.topic_mapper_.get_mappings()
# mappings = {value: text_labels[key] for key, value in mappings.items()}
#
# topic_df = topic_model.get_topic_info()
# topic_df["class"] = topic_df.Topic.map(mappings)


empty_dimensionality_model = BaseDimensionalityReduction()
clf = LogisticRegression()
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)


class LabelInference:
    def __init__(self, train_data: datasets.Dataset, text_labels: List[str]):
        self.train_data = train_data
        self.text_labels = text_labels

        # initialize topic model
        self.topic_model = BERTopic(
            umap_model=empty_dimensionality_model,
            hdbscan_model=clf,
            ctfidf_model=ctfidf_model,
        )

        self.topics, self.probs = self.topic_model.fit_transform(
            train_data["text"], y=train_data["label"]
        )

        mappings = self.topic_model.topic_mapper_.get_mappings()
        mappings = {value: text_labels[key] for key, value in mappings.items()}
        self.mappings = mappings

        self.topic_df = self.topic_model.get_topic_info()
        self.topic_df["class"] = self.topic_df.Topic.map(mappings)

    def predict(self, text: str) -> str:
        topic, _ = self.topic_model.transform(text)
        label = topic[0]
        label_text = self.mappings[label]
        return label_text

    def evaluate(self, test_data: datasets.Dataset) -> datasets.Dataset:
        def process(doc):
            predicted = self.predict(doc["text"])
            doc["predicted"] = predicted
            doc["match"] = predicted in doc["tags"]
            return doc

        return test_data.map(process)
