import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🚴‍♂️ Bike Sales Dashboard",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    """Load bike sales data from CSV file"""
    try:
        # List of possible file paths
        possible_paths = [
            "clean_bike_sales.csv",
            "data/clean_bike_sales.csv", 
            "./clean_bike_sales.csv",
            "bike_sales.csv",
            "sales_data.csv"
        ]
        
        df = None
        loaded_path = None
        
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                loaded_path = path
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                st.error(f"Error reading {path}: {str(e)}")
                continue
        
        if df is not None:
            st.success(f"✅ Data loaded successfully from: {loaded_path}")
            
            # Check required columns
            required_cols = ["Store_Location", "Bike_Model", "Customer_Gender", "Customer_Age", "Revenue", "Payment_Method"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.info(f"Available columns: {list(df.columns)}")
                return pd.DataFrame()
            
            # Clean the data
            df = df.dropna(subset=required_cols)
            
            # Add derived columns for analysis
            df['Revenue_Category'] = pd.cut(df['Revenue'], 
                                          bins=[0, 1000, 5000, 10000, float('inf')], 
                                          labels=['Low ($0-1K)', 'Medium ($1K-5K)', 'High ($5K-10K)', 'Premium ($10K+)'])
            
            df['Age_Group'] = pd.cut(df['Customer_Age'], 
                                   bins=[0, 25, 35, 45, 55, 100], 
                                   labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            
            return df
        else:
            st.warning("⚠️ CSV file not found. Creating sample data for demonstration...")
            return create_sample_data()
            
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        st.info("Creating sample data for demonstration...")
        return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration purposes"""
    np.random.seed(42)
    n_records = 1000
    
    stores = ['Downtown Store', 'Mall Location', 'Suburban Shop', 'City Center', 'Outlet Store']
    models = ['Mountain Pro', 'City Cruiser', 'Racing Elite', 'Comfort Ride', 'Electric Boost', 'Kids Special']
    genders = ['Male', 'Female']
    payments = ['Credit Card', 'Cash', 'Debit Card', 'Online Payment']
    
    data = {
        'Store_Location': np.random.choice(stores, n_records),
        'Bike_Model': np.random.choice(models, n_records),
        'Customer_Gender': np.random.choice(genders, n_records),
        'Customer_Age': np.random.randint(18, 70, n_records),
        'Revenue': np.random.randint(500, 15000, n_records),
        'Payment_Method': np.random.choice(payments, n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Add derived columns
    df['Revenue_Category'] = pd.cut(df['Revenue'], 
                                  bins=[0, 1000, 5000, 10000, float('inf')], 
                                  labels=['Low ($0-1K)', 'Medium ($1K-5K)', 'High ($5K-10K)', 'Premium ($10K+)'])
    
    df['Age_Group'] = pd.cut(df['Customer_Age'], 
                           bins=[0, 25, 35, 45, 55, 100], 
                           labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    
    return df

@st.cache_data
def filter_data(df, locations, models, genders, age_range, revenue_range):
    """Filter dataframe based on user selections"""
    if df.empty:
        return df
    
    filtered = df[
        (df['Store_Location'].isin(locations)) &
        (df['Bike_Model'].isin(models)) &
        (df['Customer_Gender'].isin(genders)) &
        (df['Customer_Age'] >= age_range[0]) &
        (df['Customer_Age'] <= age_range[1]) &
        (df['Revenue'] >= revenue_range[0]) &
        (df['Revenue'] <= revenue_range[1])
    ]
    
    return filtered

def main():
    # Header
    st.markdown('<h1 class="main-header">🚴‍♂️ Bike Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df.empty:
        st.error("❌ No data available. Please check your CSV file.")
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🔍 Dashboard Filters")
    st.sidebar.markdown("---")
    
    # Store location filter
    locations = st.sidebar.multiselect(
        "📍 Store Locations",
        options=sorted(df['Store_Location'].unique()),
        default=sorted(df['Store_Location'].unique()),
        help="Select store locations to include in analysis"
    )
    
    # Bike model filter  
    models = st.sidebar.multiselect(
        "🚲 Bike Models",
        options=sorted(df['Bike_Model'].unique()),
        default=sorted(df['Bike_Model'].unique()),
        help="Select bike models to analyze"
    )
    
    # Gender filter
    genders = st.sidebar.multiselect(
        "👥 Customer Gender",
        options=sorted(df['Customer_Gender'].unique()),
        default=sorted(df['Customer_Gender'].unique()),
        help="Filter by customer gender"
    )
    
    # Age range filter
    age_min, age_max = int(df['Customer_Age'].min()), int(df['Customer_Age'].max())
    age_range = st.sidebar.slider(
        "🎂 Age Range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
        help="Select customer age range"
    )
    
    # Revenue range filter
    rev_min, rev_max = float(df['Revenue'].min()), float(df['Revenue'].max())
    revenue_range = st.sidebar.slider(
        "💰 Revenue Range ($)",
        min_value=rev_min,
        max_value=rev_max,
        value=(rev_min, rev_max),
        format="$%.0f",
        help="Filter by revenue range"
    )
    
    # Apply filters
    filtered_df = filter_data(df, locations, models, genders, age_range, revenue_range)
    
    if filtered_df.empty:
        st.warning("⚠️ No data matches your current filters. Please adjust your selections.")
        st.stop()
    
    # Display filter summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Filter Summary")
    st.sidebar.info(f"""
    **Total Records**: {len(df):,}  
    **Filtered Records**: {len(filtered_df):,}  
    **Percentage Shown**: {(len(filtered_df)/len(df)*100):.1f}%
    """)
    
    # Main content area
    # Key Performance Indicators
    st.markdown("### 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_df['Revenue'].sum()
        avg_revenue = filtered_df['Revenue'].mean()
        st.metric(
            "Total Revenue",
            f"${total_revenue:,.0f}",
            delta=f"${avg_revenue:.0f} average"
        )
    
    with col2:
        total_orders = len(filtered_df)
        orders_per_store = total_orders / len(locations) if locations else 0
        st.metric(
            "Total Orders",
            f"{total_orders:,}",
            delta=f"{orders_per_store:.0f} per store"
        )
    
    with col3:
        avg_order_value = filtered_df['Revenue'].mean()
        median_order_value = filtered_df['Revenue'].median()
        st.metric(
            "Avg Order Value",
            f"${avg_order_value:.2f}",
            delta=f"${median_order_value:.0f} median"
        )
    
    with col4:
        unique_ages = filtered_df['Customer_Age'].nunique()
        age_spread = filtered_df['Customer_Age'].max() - filtered_df['Customer_Age'].min()
        st.metric(
            "Customer Diversity",
            f"{unique_ages} age points",
            delta=f"{age_spread} year span"
        )
    
    st.markdown("---")
    
    # Revenue Analysis Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏪 Revenue by Store Location")
        store_revenue = filtered_df.groupby('Store_Location')['Revenue'].sum().sort_values(ascending=False)
        st.bar_chart(store_revenue)
        
        # Top performers table
        st.markdown("**Top Performing Stores:**")
        for i, (store, revenue) in enumerate(store_revenue.head(3).items(), 1):
            percentage = (revenue / store_revenue.sum()) * 100
            st.write(f"{i}. {store}: ${revenue:,.0f} ({percentage:.1f}%)")
    
    with col2:
        st.markdown("### 🚲 Revenue by Bike Model")
        model_revenue = filtered_df.groupby('Bike_Model')['Revenue'].sum().sort_values(ascending=False)
        st.bar_chart(model_revenue)
        
        # Top models table
        st.markdown("**Top Selling Models:**")
        for i, (model, revenue) in enumerate(model_revenue.head(3).items(), 1):
            percentage = (revenue / model_revenue.sum()) * 100
            st.write(f"{i}. {model}: ${revenue:,.0f} ({percentage:.1f}%)")
    
    # Customer Analysis
    st.markdown("### 👥 Customer Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Revenue by Gender")
        gender_revenue = filtered_df.groupby('Customer_Gender')['Revenue'].sum()
        st.bar_chart(gender_revenue)
        
        # Gender breakdown
        st.markdown("**Gender Distribution:**")
        for gender, revenue in gender_revenue.items():
            orders = len(filtered_df[filtered_df['Customer_Gender'] == gender])
            avg_spend = revenue / orders if orders > 0 else 0
            st.write(f"• {gender}: {orders} orders, ${avg_spend:.0f} avg")
    
    with col2:
        st.markdown("#### Revenue by Age Group")
        age_revenue = filtered_df.groupby('Age_Group')['Revenue'].sum()
        st.bar_chart(age_revenue)
        
        # Age group insights
        st.markdown("**Age Group Performance:**")
        for age_group, revenue in age_revenue.items():
            orders = len(filtered_df[filtered_df['Age_Group'] == age_group])
            st.write(f"• {age_group}: {orders} customers, ${revenue:,.0f}")
    
    # Payment and Revenue Categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💳 Revenue by Payment Method")
        payment_revenue = filtered_df.groupby('Payment_Method')['Revenue'].sum().sort_values(ascending=False)
        st.bar_chart(payment_revenue)
    
    with col2:
        st.markdown("### 💰 Revenue Categories")
        category_counts = filtered_df['Revenue_Category'].value_counts()
        st.bar_chart(category_counts)
    
    # Detailed Analysis Tables
    st.markdown("### 📋 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Store Performance", "Model Analysis", "Customer Insights"])
    
    with tab1:
        st.markdown("#### Store Performance Summary")
        store_analysis = filtered_df.groupby('Store_Location').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Customer_Age': 'mean'
        }).round(2)
        store_analysis.columns = ['Total Revenue', 'Avg Revenue', 'Orders', 'Avg Customer Age']
        store_analysis['Revenue per Order'] = (store_analysis['Total Revenue'] / store_analysis['Orders']).round(2)
        st.dataframe(store_analysis.sort_values('Total Revenue', ascending=False), use_container_width=True)
    
    with tab2:
        st.markdown("#### Bike Model Performance")
        model_analysis = filtered_df.groupby('Bike_Model').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Customer_Age': 'mean'
        }).round(2)
        model_analysis.columns = ['Total Revenue', 'Avg Revenue', 'Orders', 'Avg Customer Age']
        st.dataframe(model_analysis.sort_values('Total Revenue', ascending=False), use_container_width=True)
    
    with tab3:
        st.markdown("#### Customer Demographics")
        demo_analysis = filtered_df.groupby(['Age_Group', 'Customer_Gender']).agg({
            'Revenue': ['sum', 'mean', 'count']
        }).round(2)
        demo_analysis.columns = ['Total Revenue', 'Avg Revenue', 'Orders']
        st.dataframe(demo_analysis, use_container_width=True)
    
    # Raw Data Explorer
    with st.expander("🔍 Raw Data Explorer"):
        st.markdown("### Explore Your Data")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(filtered_df))
        with col2:
            st.metric("Total Columns", len(filtered_df.columns))
        with col3:
            st.metric("Memory Usage", f"{filtered_df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Sample of data
        st.markdown("#### Sample Data (First 100 rows)")
        st.dataframe(filtered_df.head(100), use_container_width=True)
        
        # Download options
        col1, col2 = st.columns(2)
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data (CSV)",
                data=csv,
                file_name=f"bike_sales_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            summary_stats = filtered_df.describe(include='all')
            summary_csv = summary_stats.to_csv()
            st.download_button(
                label="📊 Download Summary Stats",
                data=summary_csv,
                file_name=f"bike_sales_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚴‍♂️ <strong>Bike Sales Analytics Dashboard</strong> | Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
