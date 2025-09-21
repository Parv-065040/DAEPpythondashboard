"""
Bike Sales Dashboard - Streamlit App
Author: Your Team
Description: Interactive dashboard for analyzing bike sales data
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="🚴‍♂️ Bike Sales Analytics",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS Styling
# ----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Data Loading Functions
# ----------------------------
@st.cache_data
def load_data():
    """Load and preprocess the bike sales data"""
    try:
        # Try multiple possible file paths
        possible_paths = [
            "clean_bike_sales.csv",
            "data/clean_bike_sales.csv",
            "./clean_bike_sales.csv",
            "/content/drive/MyDrive/Colab Notebooks/clean_bike_sales.csv"
        ]
        
        df = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                st.success(f"✅ Data loaded successfully from: {path}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                st.error(f"Error reading {path}: {str(e)}")
                continue
        
        if df is None:
            st.error("❌ CSV file not found in any of the expected locations.")
            st.info("Please ensure 'clean_bike_sales.csv' is in the same directory as app.py")
            return pd.DataFrame()
        
        # Data preprocessing
        required_columns = ["Store_Location", "Bike_Model", "Customer_Gender", "Customer_Age", "Revenue", "Payment_Method"]
        
        # Check if all required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns in dataset: {missing_cols}")
            st.info("Available columns: " + ", ".join(df.columns.tolist()))
            return pd.DataFrame()
        
        # Convert to appropriate data types
        categorical_cols = ["Store_Location", "Bike_Model", "Customer_Gender", "Payment_Method"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
        
        # Handle missing values
        df = df.dropna(subset=required_columns)
        
        # Add derived columns
        df['Revenue_Category'] = pd.cut(df['Revenue'], 
                                       bins=[0, 1000, 5000, 10000, float('inf')], 
                                       labels=['Low', 'Medium', 'High', 'Premium'])
        
        df['Age_Group'] = pd.cut(df['Customer_Age'], 
                                bins=[0, 25, 35, 45, 55, 100], 
                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        return df
        
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def filter_data(df, locations, models, genders, age_range, revenue_range):
    """Filter data based on user selections"""
    if df.empty:
        return df
        
    filtered_df = df[
        (df["Store_Location"].isin(locations)) &
        (df["Bike_Model"].isin(models)) &
        (df["Customer_Gender"].isin(genders)) &
        (df["Customer_Age"].between(age_range[0], age_range[1])) &
        (df["Revenue"].between(revenue_range[0], revenue_range[1]))
    ].copy()
    
    return filtered_df

# ----------------------------
# Main App
# ----------------------------
def main():
    # Header
    st.markdown('<h1 class="main-header">🚴‍♂️ Bike Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df.empty:
        st.stop()
    
    # Sidebar Filters
    st.sidebar.header("🔍 Filter Dashboard")
    st.sidebar.markdown("---")
    
    # Location filter
    locations = st.sidebar.multiselect(
        "📍 Store Locations",
        options=sorted(df["Store_Location"].cat.categories),
        default=sorted(df["Store_Location"].cat.categories)[:5] if len(df["Store_Location"].cat.categories) > 5 else sorted(df["Store_Location"].cat.categories),
        help="Select one or more store locations"
    )
    
    # Model filter
    models = st.sidebar.multiselect(
        "🚲 Bike Models",
        options=sorted(df["Bike_Model"].cat.categories),
        default=sorted(df["Bike_Model"].cat.categories)[:5] if len(df["Bike_Model"].cat.categories) > 5 else sorted(df["Bike_Model"].cat.categories),
        help="Select bike models to analyze"
    )
    
    # Gender filter
    genders = st.sidebar.multiselect(
        "👥 Customer Gender",
        options=sorted(df["Customer_Gender"].cat.categories),
        default=sorted(df["Customer_Gender"].cat.categories),
        help="Filter by customer gender"
    )
    
    # Age range filter
    min_age, max_age = int(df["Customer_Age"].min()), int(df["Customer_Age"].max())
    age_range = st.sidebar.slider(
        "🎂 Customer Age Range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        help="Select age range for customers"
    )
    
    # Revenue range filter
    min_rev, max_rev = float(df["Revenue"].min()), float(df["Revenue"].max())
    revenue_range = st.sidebar.slider(
        "💰 Revenue Range ($)",
        min_value=min_rev,
        max_value=max_rev,
        value=(min_rev, max_rev),
        format="$%.0f",
        help="Filter by revenue range"
    )
    
    # Apply filters
    filtered_df = filter_data(df, locations, models, genders, age_range, revenue_range)
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches your current filters. Please adjust your selections.")
        st.stop()
    
    # Display data info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.info(f"""
    **Original Dataset**: {len(df):,} records
    **Filtered Dataset**: {len(filtered_df):,} records
    **Reduction**: {((len(df) - len(filtered_df)) / len(df) * 100):.1f}%
    """)
    
    # Main Dashboard
    # Key Metrics Row
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_df['Revenue'].sum()
        st.metric(
            "Total Revenue",
            f"${total_revenue:,.0f}",
            delta=f"${total_revenue/len(filtered_df):.0f} avg per sale"
        )
    
    with col2:
        total_orders = len(filtered_df)
        avg_orders_per_location = total_orders / len(locations) if locations else 0
        st.metric(
            "Total Orders",
            f"{total_orders:,}",
            delta=f"{avg_orders_per_location:.0f} per location"
        )
    
    with col3:
        avg_revenue = filtered_df['Revenue'].mean()
        median_revenue = filtered_df['Revenue'].median()
        st.metric(
            "Average Revenue",
            f"${avg_revenue:.2f}",
            delta=f"${median_revenue:.0f} median"
        )
    
    with col4:
        unique_customers = filtered_df['Customer_Age'].nunique()
        customer_retention = (total_orders / unique_customers) if unique_customers > 0 else 0
        st.metric(
            "Customer Segments",
            f"{unique_customers:,}",
            delta=f"{customer_retention:.1f} orders per segment"
        )
    
    st.markdown("---")
    
    # Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏪 Revenue by Store Location")
        location_revenue = filtered_df.groupby("Store_Location")["Revenue"].sum().sort_values(ascending=True)
        
        fig_location = px.bar(
            x=location_revenue.values,
            y=location_revenue.index,
            orientation='h',
            title="Revenue Performance by Location",
            color=location_revenue.values,
            color_continuous_scale="Blues"
        )
        fig_location.update_layout(
            xaxis_title="Revenue ($)",
            yaxis_title="Store Location",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_location, use_container_width=True)
    
    with col2:
        st.markdown("### 🚲 Top Performing Bike Models")
        model_revenue = filtered_df.groupby("Bike_Model")["Revenue"].sum().nlargest(10)
        
        fig_model = px.pie(
            values=model_revenue.values,
            names=model_revenue.index,
            title="Revenue Share by Top 10 Bike Models"
        )
        fig_model.update_traces(textposition='inside', textinfo='percent+label')
        fig_model.update_layout(height=400)
        st.plotly_chart(fig_model, use_container_width=True)
    
    # Full width charts
    st.markdown("### 💳 Revenue Analysis by Payment Method and Bike Model")
    payment_model_df = filtered_df.groupby(["Payment_Method", "Bike_Model"])["Revenue"].sum().reset_index()
    top_models_for_payment = payment_model_df.groupby("Bike_Model")["Revenue"].sum().nlargest(8).index
    payment_model_df = payment_model_df[payment_model_df["Bike_Model"].isin(top_models_for_payment)]
    
    fig_payment = px.bar(
        payment_model_df,
        x="Payment_Method",
        y="Revenue",
        color="Bike_Model",
        title="Revenue by Payment Method and Bike Model (Top 8 Models)",
        text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_payment.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig_payment, use_container_width=True)
    
    # Age analysis
    st.markdown("### 👥 Customer Demographics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_gender_revenue = filtered_df.groupby(["Age_Group", "Customer_Gender"])["Revenue"].sum().reset_index()
        
        fig_age = px.bar(
            age_gender_revenue,
            x="Age_Group",
            y="Revenue",
            color="Customer_Gender",
            title="Revenue by Age Group and Gender",
            barmode="group"
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        revenue_dist = filtered_df["Revenue_Category"].value_counts()
        
        fig_rev_cat = px.bar(
            x=revenue_dist.index,
            y=revenue_dist.values,
            title="Distribution of Revenue Categories",
            color=revenue_dist.values,
            color_continuous_scale="Viridis"
        )
        fig_rev_cat.update_layout(
            xaxis_title="Revenue Category",
            yaxis_title="Number of Orders",
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_rev_cat, use_container_width=True)
    
    # Heatmap
    st.markdown("### 🔥 Revenue Heatmap: Store vs Payment Method")
    heatmap_data = filtered_df.groupby(["Store_Location", "Payment_Method"])["Revenue"].sum().unstack(fill_value=0)
    
    fig_heatmap = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="Viridis",
        aspect="auto",
        title="Revenue Intensity by Store and Payment Method"
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Data Table
    with st.expander("📋 View Filtered Data"):
        st.markdown("### Detailed Data View")
        
        # Add some statistics about the filtered data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows Displayed", len(filtered_df))
        with col2:
            st.metric("Columns", len(filtered_df.columns))
        with col3:
            st.metric("Data Size", f"{filtered_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Display the dataframe
        st.dataframe(
            filtered_df.head(1000),  # Limit to first 1000 rows for performance
            use_container_width=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"filtered_bike_sales_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚴‍♂️ <strong>Bike Sales Analytics Dashboard</strong> | Built with Streamlit & Plotly</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
