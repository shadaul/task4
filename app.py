from datetime import datetime
import re
from itertools import combinations
from collections import defaultdict

import pandas as pd
import streamlit as st
import yaml


st.set_page_config(page_title="Store Analytics", layout="wide")


def parse_price(value):
    """Return a USD amount, including prices with an explicit cents symbol."""
    from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
    text = str(value).strip().upper()
    euro = "EUR" in text or "€" in text
    text = text.replace("USD", "").replace("EUR", "")
    text = text.replace("$", "").replace("€", ".").replace("¢", ".")
    text = text.strip(". ").replace(",", ".")
    if text.count(".") > 1:
        return float("nan")
    try:
        amount = Decimal(text)
        if not amount.is_finite() or amount < 0:
            return float("nan")
        if euro:
            amount *= Decimal("1.2")
        return float(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except InvalidOperation:
        return float("nan")


def parse_date(value):
    """Parse the date token explicitly; never fill missing parts with today."""
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime)):
        return value
    text = str(value).strip()
    patterns = [
        (r"(?<!\d)\d{4}-\d{1,2}-\d{1,2}(?!\d)", ("%Y-%m-%d",)),
        (r"\b\d{1,2}-[A-Za-z]+-\d{4}\b", ("%d-%B-%Y", "%d-%b-%Y")),
        (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", ("%m/%d/%Y", "%m/%d/%y")),
        (r"\b\d{1,2}\.\d{1,2}\.\d{4}\b", ("%d.%m.%Y",)),
    ]
    for pattern, formats in patterns:
        match = re.search(pattern, text)
        if match:
            for fmt in formats:
                try:
                    return datetime.strptime(match.group(), fmt)
                except ValueError:
                    pass
            return pd.NaT
    for fmt in ("%a %b %d %H:%M:%S %Y", "%a %b %d %H:%M:%S %Z %Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return pd.NaT


def get_author_set(author):
    if author is None:
        return ()

    if isinstance(author, float) and pd.isna(author):
        return ()

    if isinstance(author, (list, tuple, set)):
        names = author
    else:
        names = str(author).split(",")

    return tuple(sorted(
        {str(name).strip() for name in names if str(name).strip()},
        key=str.lower
    ))


def count_unique_users(users):
    users = users.reset_index(drop=True).copy()
    fields = ["name", "address", "phone", "email"]
    indexes = defaultdict(list)

    for i, row in users.iterrows():
        for field in fields:
            value = str(row[field]).strip().lower() if pd.notna(row[field]) else ""

            if field == "phone":
                value = re.sub(r"\D", "", value)
            elif field in ("name", "address"):
                value = re.sub(r"[^a-z0-9]", "", value)

            if value:
                indexes[(field, value)].append(i)

    parent = list(range(len(users)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    matches = defaultdict(int)

    for rows in indexes.values():
        for a, b in combinations(rows, 2):
            matches[(a, b)] += 1

    for (a, b), count in matches.items():
        if count >= 3:
            union(a, b)

    users["person_key"] = [find(i) for i in range(len(users))]
    user_map = dict(zip(users["id"].astype(str), users["person_key"]))

    return users, user_map, users["person_key"].nunique()


@st.cache_data
def load_store(store_id):
    from pathlib import Path
    folder = Path(__file__).resolve().parent / f"STORE{store_id}"

    users = pd.read_csv(f"{folder}/users.csv")
    users = users.drop_duplicates(subset="id")

    with open(f"{folder}/books.yaml", encoding="utf-8") as file:
        books = yaml.safe_load(file)

    authors = {}

    for book in books:
        book_id = book.get(":id", book.get("id"))
        author = book.get(":author", book.get("author"))
        authors[str(book_id)] = get_author_set(author)

    users, user_map, unique_users = count_unique_users(users)

    orders = pd.read_parquet(f"{folder}/orders.parquet")
    orders = orders.drop_duplicates(subset="id").copy()
    orders["unit_price"] = orders["unit_price"].apply(parse_price)
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["paid_price"] = (
        orders["unit_price"] * orders["quantity"]
    ).round(2)

    orders["timestamp"] = orders["timestamp"].apply(parse_date)
    orders["date"] = orders["timestamp"].apply(
        lambda x: x.date() if pd.notna(x) else pd.NaT
    )

    invalid_rows = int(orders[["date", "quantity", "paid_price"]].isna().any(axis=1).sum())
    orders = orders.dropna(subset=["date", "quantity", "paid_price"])
    orders = orders[(orders["quantity"] > 0) & (orders["quantity"] % 1 == 0)]

    orders["user_key"] = (
        orders["user_id"].astype(str).map(user_map).astype("string")
    )

    orders["user_key"] = orders["user_key"].fillna(
        "unknown_" + orders["user_id"].astype(str)
    )

    orders["author_set"] = orders["book_id"].astype(str).map(authors)
    orders["author_set"] = orders["author_set"].apply(
        lambda x: x if isinstance(x, tuple) else ()
    )
    orders["author_label"] = orders["author_set"].apply(
        lambda x: ", ".join(x) if x else "Unknown"
    )

    book_rows = [{str(k).lstrip(":"): v for k, v in book.items()} for book in books]
    book_table = pd.DataFrame(book_rows).rename(columns={"author": "authors"})
    book_table["authors"] = book_table["id"].astype(str).map(authors).map(lambda x: ", ".join(x))
    result = summarize(orders, users, authors)
    result.update(orders=orders, customers=users, books=book_table, invalid_rows=invalid_rows)
    return result


def summarize(orders, users, authors):
    daily_revenue = orders.groupby("date")["paid_price"].sum().round(2)
    top_days = daily_revenue.sort_values(ascending=False).head(5)

    sold_books = orders.loc[orders["author_set"].map(bool).astype(bool)]
    popular = sold_books.groupby("author_label")["quantity"].sum()

    if not popular.empty:
        popular = popular[popular == popular.max()]

    spending = orders.groupby("user_key")["paid_price"].sum()

    if spending.empty:
        buyer_key = None
        buyer_ids = []
        buyer_names = []
        buyer_spending = 0
    else:
        buyer_key = spending.idxmax()
        buyer = users[
            users["person_key"].astype(str) == str(buyer_key)
        ]
        buyer_ids = buyer["id"].tolist()
        buyer_names = buyer["name"].dropna().unique().tolist()
        buyer_spending = spending[buyer_key]

    return {
        "invalid_rows": 0,
        "users": users["person_key"].nunique(),
        "author_sets": len({value for value in authors.values() if value}),
        "daily_revenue": daily_revenue,
        "top_days": top_days,
        "popular_authors": popular,
        "buyer_ids": buyer_ids,
        "buyer_names": buyer_names,
        "buyer_spending": buyer_spending,
    }


from ai_chat import render_dashboard

render_dashboard(load_store, summarize)
