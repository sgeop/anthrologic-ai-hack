import polars as pl
import skrub


def clean_tags(alias: str):
    normalize = (
        pl.element().str.strip_chars(" ").str.to_lowercase().str.replace_all(r"\s", "-")
    )
    clean_tag = pl.element().str.split("/").list.eval(normalize).list.join("/")
    q = pl.col("tags").str.split(",").list.eval(clean_tag).alias(alias)
    return q


def get_tickets(tags_column="tags"):
    tickets = pl.read_csv("data/tickets.csv")
    return tickets.with_columns(clean_tags(alias=tags_column))


def dedupe_tags(tickets: pl.DataFrame) -> pl.DataFrame:
    tags = tickets.explode("tags")

    tags_unique = tags.select(pl.col("tags").unique()).filter(
        pl.col("tags").str.contains("^[a-z]")
    )

    normalized = skrub.deduplicate(tags_unique.to_series())

    tags_mapping = tags_unique.with_columns(pl.Series("normalized_tags", normalized))

    return (
        tags.join(tags_mapping, on="tags", how="inner")
        .group_by("ticket_id", "body_text")
        .agg(pl.col("tags"), pl.col("normalized_tags"))
    )
