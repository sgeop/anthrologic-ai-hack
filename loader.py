import polars as pl
import datasets as ds
import skrub
import nltk

nltk.download("words")

english_words = set(nltk.corpus.words.words())


def filter_words_udf(series):
    results = [
        "_".join([word for word in elem.split("_") if word in english_words])
        for elem in series
    ]
    return pl.Series(results)


def clean_text(alias: str):
    clean = (
        pl.element()  # .str.replace_all(r"[^. a-zA-Z0-9]", "")
        .str.replace_all(r"[\n\t\s\-]+", " ")
        .str.strip_chars(" ")
    )
    q = (
        pl.col("body_text")
        .str.split("\n")
        .list.eval(clean)
        .list.join(" ")
        .str.replace_all(r"[\s]+", " ")
        .alias(alias)
    )
    return q


def clean_text_2(alias: str):
    return (
        pl.col("body_text")
        .str.replace_all(r"[\n\t\s\r]", " ")
        .str.replace_all(r"http[s]?://[^\s]+", "")
        .str.replace_all(r"[^. a-zA-Z0-9]|http[s]?[^s]+", " ")
        .str.replace_all(r"[\s]+", " ")
        .alias(alias)
    )


def clean_tags(alias: str):
    strip = (
        pl.element()
        .str.strip_chars(" -,")
        .str.to_lowercase()
        .str.replace_all(r"[\s\-]+", "_")
    )
    clean_tag = pl.element().str.split("/").list.eval(strip).list.join("_")
    q = pl.col("tags").str.split(",").list.eval(clean_tag).alias(alias)
    return q


def get_tickets():
    tags_column = "tags"
    tickets = pl.read_csv("data/tickets.csv").with_columns(
        pl.col("tags").alias("original_tags")
    )
    with_tags = tickets.with_columns(
        clean_text(alias="orig_text"),
        clean_text_2(alias="text"),
        clean_tags(alias=tags_column),
    )

    return dedupe_tags(with_tags)
    # return with_tags


def to_dataset(tickets: pl.DataFrame, threshold_norm: float) -> ds.Dataset:
    labels = indexed_labels(tickets, threshold_norm=threshold_norm)
    joined = (
        (
            tickets.explode("tags")
            .join(labels, left_on="tags", right_on="label_text", how="inner")
            .sort("count", descending=False)
        )
        .group_by("ticket_id")
        .agg(
            pl.first("text"),
            pl.col("tags").alias("tags"),
            pl.first("tags").alias("label_text"),
            pl.first("label"),  # .cast(int, strict=False),
            pl.first("count").alias("label_count"),
        )
    )

    return ds.Dataset.from_polars(joined)


def dedupe_tags(tickets: pl.DataFrame) -> pl.DataFrame:
    tags = tickets.explode("tags")

    # tags_unique = tags.select(pl.col("tags").unique()).filter(
    #     pl.col("tags").str.contains("^[a-z]")
    #     & (~pl.col("tags").str.contains(r"[\d]{4}_[\d]{2}_[\d]{2}"))
    # )
    tags_filtered = tags.select(
        pl.col("tags").map_batches(filter_words_udf),
    ).filter(pl.col("tags").str.contains("[^a-z]"))

    tags_unique = tags_filtered.select(
        pl.col("tags").unique(maintain_order=True),
        pl.col("tags").unique_counts().alias("count"),
    ).sort("count", descending=True)

    normalized = skrub.deduplicate(tags_unique.select("tags").to_series())

    tags_mapping = tags_unique.with_columns(pl.Series("normalized_tags", normalized))

    return (
        tags.join(tags_mapping, on="tags", how="inner")
        .group_by("ticket_id", "body_text", "text")
        .agg(pl.col("original_tags"), pl.col("tags"), pl.col("normalized_tags"))
    )


def indexed_labels(
    tickets: pl.DataFrame, tag_col="tags", threshold_norm: float = 0.01
) -> pl.DataFrame:
    q = (
        tickets.explode(tag_col)
        .select(
            pl.col(tag_col).unique(maintain_order=True).alias("label_text"),
            pl.col(tag_col).unique_counts().alias("count"),
        )
        .sort("count", descending=True)
        .with_row_index("label")
    )
    percent = (pl.col("count") / q.height).round(2).alias("pct")
    return q.with_columns(percent).filter((pl.col("pct") / 100) >= threshold_norm)


def most_common_tag(tickets: pl.DataFrame) -> pl.DataFrame:
    tags = tickets.explode("normalized_tags")
    n_uniuque = tags.select(
        pl.col("normalized_tags").unique(maintain_order=True),
        pl.col("normalized_tags").unique_counts().alias("count"),
    )

    return (
        tags.join(n_uniuque, on="normalized_tags", how="inner")
        .group_by(
            "ticket_id",
            "body_text",
        )
        .agg(
            pl.first("tags"),
            pl.col("normalized_tags"),
            pl.first("normalized_tags").alias("tag"),
        )
    )
