
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import joypy

st.set_page_config(page_title='Bike Sales Dashboard', layout='wide')

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['Revenue'] = df['Price'] * df['Quantity']
    return df

# === Load data (adjust path if necessary) ===
DATA_PATH = 'clean_bike_sales.csv'  # or full path if different
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Data file not found at {DATA_PATH}. Upload it or update DATA_PATH.")
    st.stop()

# === Sidebar filters ===
st.sidebar.header('Filters')
min_date = df['Date'].min()
max_date = df['Date'].max()
start, end = st.sidebar.date_input('Date range', value=(min_date, max_date))

models = ['All'] + sorted(df['Bike_Model'].unique().tolist())
selected_model = st.sidebar.selectbox('Bike Model', models)

stores = ['All'] + sorted(df['Store_Location'].unique().tolist())
selected_store = st.sidebar.selectbox('Store Location', stores)

payments = ['All'] + sorted(df['Payment_Method'].unique().tolist())
selected_payment = st.sidebar.selectbox('Payment Method', payments)

age_min, age_max = int(df['Customer_Age'].min()), int(df['Customer_Age'].max())
age_range = st.sidebar.slider('Customer age', age_min, age_max, (age_min, age_max))

genders = ['All'] + sorted(df['Customer_Gender'].dropna().unique().tolist())
selected_gender = st.sidebar.selectbox('Customer gender', genders)

# apply filters
mask = (df['Date'] >= pd.to_datetime(start)) & (df['Date'] <= pd.to_datetime(end))
if selected_model != 'All':
    mask &= (df['Bike_Model'] == selected_model)
if selected_store != 'All':
    mask &= (df['Store_Location'] == selected_store)
if selected_payment != 'All':
    mask &= (df['Payment_Method'] == selected_payment)
mask &= df['Customer_Age'].between(age_range[0], age_range[1])
if selected_gender != 'All':
    mask &= (df['Customer_Gender'] == selected_gender)

filtered = df[mask].copy()

# === Top-level KPIs ===
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total revenue', f"${filtered['Revenue'].sum():,.0f}")
col2.metric('Total units sold', int(filtered['Quantity'].sum()))
col3.metric('Average price', f"${filtered['Price'].mean():.2f}")
col4.metric('Unique customers', int(filtered['Customer_ID'].nunique()))

# === Tabs for sections ===
tab1, tab2, tab3, tab4 = st.tabs(['Overview', 'Product Analysis', 'Customer Analysis', 'Advanced Visuals'])

with tab1:
    st.header('Sales over time')
    ts = filtered.set_index('Date').resample('M')['Revenue'].sum().reset_index()
    fig = px.line(ts, x='Date', y='Revenue', title='Monthly revenue')
    st.plotly_chart(fig, use_container_width=True)

    st.header('Revenue by Store and Model')
    fig2 = px.sunburst(filtered, path=['Store_Location', 'Bike_Model'], values='Revenue', title='Store → Bike Model → Revenue')
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header('Top bike models')
    top_models = filtered.groupby('Bike_Model').agg({'Revenue': 'sum', 'Quantity': 'sum'}).reset_index().sort_values('Revenue', ascending=False)
    st.dataframe(top_models.head(10))

    st.subheader('Price distribution by model')
    fig_violin = px.violin(filtered, x='Bike_Model', y='Price', box=True, points='all', title='Price distribution per model')
    st.plotly_chart(fig_violin, use_container_width=True)

with tab3:
    st.header('Customer age vs purchase behavior')
    fig_scatter = px.scatter(filtered, x='Customer_Age', y='Revenue', size='Quantity', color='Bike_Model', hover_data=['Customer_ID'], title='Age vs Revenue (bubble = quantity)')
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader('Parallel coordinates (numerical features)')
    # choose numeric columns for parallel coords
    num_cols = ['Price', 'Quantity', 'Customer_Age', 'Revenue']
    try:
        fig_pc = px.parallel_coordinates(filtered.sample(min(1000, len(filtered))), dimensions=num_cols, color='Price')
        st.plotly_chart(fig_pc, use_container_width=True)
    except Exception as e:
        st.warning('Parallel coordinates failed: ' + str(e))

with tab4:
    st.header('Sankey: Store -> Payment -> Bike Model')
    # Build sankey
    try:
        src = filtered['Store_Location'].astype(str)
        mid = filtered['Payment_Method'].astype(str)
        dst = filtered['Bike_Model'].astype(str)

        labels = list(pd.concat([src, mid, dst]).unique())
        label_index = {l: i for i, l in enumerate(labels)}

        # Source -> middle
        s = src.map(label_index)
        t = mid.map(label_index)
        v = filtered.groupby([src, mid]).size().values
        # We'll build edges by grouping explicitly
        link_source = []
        link_target = []
        link_value = []
        g1 = filtered.groupby(['Store_Location', 'Payment_Method']).size().reset_index(name='count')
        for _, r in g1.iterrows():
            link_source.append(label_index[r['Store_Location']])
            link_target.append(label_index[r['Payment_Method']])
            link_value.append(r['count'])
        g2 = filtered.groupby(['Payment_Method', 'Bike_Model']).size().reset_index(name='count')
        for _, r in g2.iterrows():
            link_source.append(label_index[r['Payment_Method']])
            link_target.append(label_index[r['Bike_Model']])
            link_value.append(r['count'])

        sankey = go.Figure(data=[go.Sankey(node={'label': labels}, link={'source': link_source, 'target': link_target, 'value': link_value})])
        sankey.update_layout(title_text='Sankey: Store → Payment → Bike Model', font_size=10)
        st.plotly_chart(sankey, use_container_width=True)
    except Exception as e:
        st.warning('Could not build Sankey: ' + str(e))

    st.subheader('3D scatter: Price, Quantity, Customer_Age')
    try:
        fig3d = px.scatter_3d(filtered.sample(min(2000, len(filtered))), x='Price', y='Quantity', z='Customer_Age', color='Bike_Model', size='Revenue', title='3D scatter')
        st.plotly_chart(fig3d, use_container_width=True)
    except Exception as e:
        st.warning('3D scatter failed: ' + str(e))

    st.subheader('Ridgeline (price distribution by model)')
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        # prepare data for joyful ridgeline: group price arrays by model
        subset = filtered[['Bike_Model', 'Price']].dropna()
        # sample to keep plot readable
        sample = subset.groupby('Bike_Model').apply(lambda x: x.sample(min(500, len(x)), random_state=1)).reset_index(drop=True)
        joypy.joyplot(sample, by='Bike_Model', column='Price', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.warning('Ridgeline failed (requires joypy & matplotlib): ' + str(e))

# Footer
st.markdown('---')
st.caption('Dashboard generated with Streamlit + Plotly')
