import pandas as pd
import yaml
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Multi-Store Analytics Dashboard", layout="wide")

def load_store_data(folder_id):
    folder_path = f'STORE{folder_id}'
    
    user_df = pd.read_csv(f'{folder_path}/users.csv')
    if 'id' in user_df.columns:
        user_df = user_df.rename(columns={'id': 'user_id'})
    elif ':id' in user_df.columns:
        user_df = user_df.rename(columns={':id': 'user_id'})
    user_df['store'] = folder_id
    
    orders_df = pd.read_parquet(f'{folder_path}/orders.parquet')
    orders_df['store'] = folder_id
    
    with open(f'{folder_path}/books.yaml', 'r', encoding='utf-8') as f:
        books_data = yaml.safe_load(f)
        
    books_df = pd.DataFrame(books_data)
    id_col = ':id' if ':id' in books_df.columns else 'id'
    books_df['book_id'] = books_df[id_col]
    
    return user_df, orders_df, books_df

@st.cache_data
def process_all_data():
    all_users, all_orders, all_books = [], [], []
    
    for i in range(1, 6):
        u_df, o_df, b_df = load_store_data(i)
        all_users.append(u_df)
        all_orders.append(o_df)
        all_books.append(b_df)

    final_orders = pd.concat(all_orders, ignore_index=True)
    final_users = pd.concat(all_users, ignore_index=True)
    final_books = pd.concat(all_books, ignore_index=True)

    final_orders['unit_price'] = final_orders['unit_price'].astype(str)
    is_euro = final_orders['unit_price'].str.contains('€', na=False)
    final_orders['unit_price'] = final_orders['unit_price'].str.replace(r'[^0-9.]', '', regex=True).astype(float)
    final_orders.loc[is_euro, 'unit_price'] *= 1.2
    final_orders['paid_price'] = final_orders['quantity'] * final_orders['unit_price']

    # Корректный парсинг дат и очистка аномальных выбросов (отсекаем мусор и даты из будущего)
    final_orders['timestamp'] = pd.to_datetime(final_orders['timestamp'], errors='coerce', utc=True)
    final_orders = final_orders.dropna(subset=['timestamp'])
    
    # Фильтруем аномалии: оставляем только даты до конца 2025 / начала 2026, убирая ложные пики
    final_orders = final_orders[final_orders['timestamp'] <= '2026-01-01']
    
    final_orders['date'] = final_orders['timestamp'].dt.strftime('%Y-%m-%d')

    daily_revenue = final_orders.groupby('date')['paid_price'].sum().reset_index()
    top_5_days = daily_revenue.sort_values(by='paid_price', ascending=False).head(5)

    author_col = ':author' if ':author' in final_books.columns else 'author'
    unique_author_sets = set()
    book_to_author_tuple = {}

    for _, row in final_books.iterrows():
        b_id = row.get('book_id')
        authors = row.get(author_col, [])
        if isinstance(authors, str):
            authors = [authors]
        elif not isinstance(authors, list):
            authors = [str(authors)]
        sorted_authors = tuple(sorted([str(a) for a in authors]))
        unique_author_sets.add(sorted_authors)
        book_to_author_tuple[b_id] = sorted_authors

    final_orders['author_set'] = final_orders['book_id'].map(book_to_author_tuple)
    author_sales = final_orders.groupby('author_set')['quantity'].sum()
    most_popular_author_tuple = author_sales.idxmax()
    most_popular_author = ", ".join(most_popular_author_tuple) if isinstance(most_popular_author_tuple, tuple) else str(most_popular_author_tuple)

    g = nx.Graph()
    user_cols = [c for c in final_users.columns if c not in ['user_id', 'store']]

    for _, row in final_users.iterrows():
        u_id = row['user_id']
        g.add_node(u_id)
        for col in user_cols:
            val = row[col]
            if pd.notna(val):
                g.add_edge(u_id, f"{col}:{val}")

    unique_users_count = 0
    user_to_cluster = {}
    for cluster in nx.connected_components(g):
        u_ids = [n for n in cluster if not any(str(n).startswith(f"{c}:") for c in user_cols)]
        if u_ids:
            unique_users_count += 1
            cluster_tuple = tuple(sorted(u_ids))
            for u_id in u_ids:
                user_to_cluster[u_id] = cluster_tuple

    final_users['cluster_id'] = final_users['user_id'].map(lambda x: user_to_cluster.get(x, (x,)))
    order_user_map = final_users.set_index('user_id')['cluster_id'].to_dict()
    final_orders['cluster_id'] = final_orders['user_id'].map(order_user_map)

    cluster_spending = final_orders.groupby('cluster_id')['paid_price'].sum()
    top_cluster = cluster_spending.idxmax()
    top_customer_ids = list(top_cluster) if isinstance(top_cluster, tuple) else [top_cluster]

    return daily_revenue, top_5_days, unique_users_count, len(unique_author_sets), most_popular_author, top_customer_ids

daily_revenue, top_5_days, unique_users_count, unique_authors_count, most_popular_author, top_customer_ids = process_all_data()

st.title("Multi-Store Analytics Dashboard")

tab1, tab2, tab3 = st.tabs(["Sales and Revenue", "Users and Authors", "Chat Interface"])

with tab1:
    st.header("Daily Revenue Performance")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Top 5 Days by Revenue")
        top_5_display = top_5_days.copy()
        top_5_display.columns = ["Date", "Revenue"]
        top_5_display["Revenue"] = top_5_display["Revenue"].map("${:,.2f}".format)
        st.dataframe(top_5_display, use_container_width=True, hide_index=True)
        
    with col2:
        st.subheader("Revenue Timeline")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pd.to_datetime(daily_revenue['date']), daily_revenue['paid_price'], color='#1f77b4', linewidth=1.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue ($)")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.header("Core Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Unique Real Users", unique_users_count)
    m2.metric("Unique Author Sets", unique_authors_count)
    m3.metric("Most Popular Author", most_popular_author)
    
    st.divider()
    st.subheader("Top Customer ID Aliases")
    st.json({"top_customer_user_ids": top_customer_ids})

with tab3:
    st.header("Analytics Assistant")
    st.write("Interactive query interface for data verification.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "System initialized. Available queries you can enter:\n- 'revenue' or 'top days'\n- 'users'\n- 'author'\n- 'customer'"
        }]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input("Enter command or query..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)
        
        query = user_input.lower()
        
        if "revenue" in query or "day" in query or "выруч" in query:
            reply = f"Top 5 days by revenue:\n\n{top_5_days.to_string(index=False)}"
        elif "user" in query or "пользовател" in query:
            reply = f"Total unique real users count: {unique_users_count}"
        elif "author" in query or "автор" in query:
            reply = f"Most popular author: {most_popular_author}. Total unique author sets: {unique_authors_count}"
        elif "customer" in query or "buyer" in query or "покупател" in query:
            reply = f"Top customer user IDs: {top_customer_ids}"
        else:
            reply = f"Processed query: '{user_input}'. Pipeline operational. To view specific data, try asking about: revenue, users, author, or customer."
            
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)