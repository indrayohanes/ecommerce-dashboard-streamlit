import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style='darkgrid')

@st.cache_data
def load_and_prepare_data():
    data_path = 'data/'

    customers_df = pd.read_csv(os.path.join(data_path, 'customers_dataset.csv'))
    orders_df = pd.read_csv(os.path.join(data_path, 'orders_dataset.csv'))
    order_items_df = pd.read_csv(os.path.join(data_path, 'order_items_dataset.csv'))
    order_payments_df = pd.read_csv(os.path.join(data_path, 'order_payments_dataset.csv'))
    products_df = pd.read_csv(os.path.join(data_path, 'products_dataset.csv'))
    product_translation_df = pd.read_csv(os.path.join(data_path, 'product_category_name_translation.csv'))

    df = pd.merge(orders_df, order_items_df, on='order_id')
    df = pd.merge(df, products_df, on='product_id')
    df = pd.merge(df, order_payments_df, on='order_id')
    df = pd.merge(df, customers_df, on='customer_id')
    df = pd.merge(df, product_translation_df, on='product_category_name')

    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    return df

main_df = load_and_prepare_data()

def create_rfm_df(df):
    snapshot_date = df['order_purchase_timestamp'].max() + pd.DateOffset(days=1)
    rfm_df = df.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (snapshot_date - x.max()).days,
        'order_id': 'nunique',
        'payment_value': 'sum'
    }).reset_index()
    rfm_df.rename(columns={'order_purchase_timestamp': 'Recency', 'order_id': 'Frequency', 'payment_value': 'Monetary'}, inplace=True)

    rfm_df['R_score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    
    rfm_df['RFM_Score'] = rfm_df['R_score'].astype(str) + rfm_df['F_score'].astype(str) + rfm_df['M_score'].astype(str)
    return rfm_df


st.title('📈 Dashboard Analisis E-Commerce')


st.sidebar.header("Filter")
min_date = main_df["order_purchase_timestamp"].min()
max_date = main_df["order_purchase_timestamp"].max()

start_date, end_date = st.sidebar.date_input(
    label='Pilih Rentang Waktu',
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date]
)


st.sidebar.markdown("---") 
st.sidebar.header("Created by:")
st.sidebar.markdown("**Nama:** Indra Yohanes")
st.sidebar.markdown("**Email:** indrayohanes3@gmail.com")


filtered_df = main_df[(main_df['order_purchase_timestamp'] >= pd.to_datetime(start_date)) & 
                      (main_df['order_purchase_timestamp'] <= pd.to_datetime(end_date))]

tab1, tab2, tab3 = st.tabs(["Analisis Demografi", "Performa Produk", "Analisis Pelanggan (RFM)"])
with tab1:
    st.header("Demografi Pelanggan")

    customer_by_state = filtered_df.groupby('customer_state')['customer_id'].nunique().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    customer_by_state.head(10).plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Top 10 Negara Bagian Berdasarkan Jumlah Pelanggan', fontsize=16)
    ax.set_xlabel('Negara Bagian (State)')
    ax.set_ylabel('Jumlah Pelanggan Unik')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

with tab2:
    st.header("Performa Produk")
    
    category_performance = filtered_df.groupby('product_category_name_english')['payment_value'].sum().sort_values(ascending=False)

    st.subheader("Kategori Produk Terbaik")
    fig, ax = plt.subplots(figsize=(12, 6))
    category_performance.head(10).plot(kind='barh', ax=ax, color='mediumseagreen').invert_yaxis()
    ax.set_title('Top 10 Kategori Produk dengan Pendapatan Tertinggi', fontsize=16)
    ax.set_xlabel('Total Pendapatan (R$)')
    ax.set_ylabel(None)
    st.pyplot(fig)

    st.subheader("Kategori Produk Terburuk")
    fig, ax = plt.subplots(figsize=(12, 6))
    category_performance.tail(10).plot(kind='barh', ax=ax, color='salmon').invert_yaxis()
    ax.set_title('Top 10 Kategori Produk dengan Pendapatan Terendah', fontsize=16)
    ax.set_xlabel('Total Pendapatan (R$)')
    ax.set_ylabel(None)
    st.pyplot(fig)

with tab3:
    st.header("Segmentasi Pelanggan Terbaik (RFM)")
    
    rfm_df = create_rfm_df(filtered_df)
    
    st.subheader("Pelanggan Terbaik (Skor RFM = 555)")
    best_customers_df = rfm_df[rfm_df['RFM_Score'] == '555'].sort_values('Monetary', ascending=False)

    st.dataframe(best_customers_df.head(10))

st.caption('Copyright (c) Indra Yohanes 2025')