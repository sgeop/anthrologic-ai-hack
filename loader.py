import polars as pl


def clean_tags(alias: str):
    normalize = pl.element().str.strip_chars(" ").str.to_lowercase().str.replace(r"\s", "-")
    clean_tag = pl.element().str.split("/").list.eval(normalize).list.join("/")
    q = pl.col("tags").str.split(",").list.eval(clean_tag).alias(alias)
    return q


def get_tickets(tags_column="tags"):
    tickets = pl.read_csv("data/tickets.csv")
    return tickets.with_columns(clean_tags(alias=tags_column))

