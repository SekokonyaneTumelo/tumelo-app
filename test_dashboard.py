import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, date, timedelta
import os
import io
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from streamlit_autorefresh import st_autorefresh
import time

# Configuration
API_URL = "http://localhost:5000/api/data"
UPLOAD_URL = "http://localhost:5000/api/upload"
CSV_FILE = "web_server_logs.csv"
API_KEY = "ai-solutions-key-2025"
st.set_page_config(
    layout="wide",
    page_title="AI-Solutions Global Sales Dashboard",
    page_icon="https://www.freepik.com/free-vector/global-technology-concept-with-globe-circuit_12152352.htm"
)

# Auto-refresh every 60 seconds
st_autorefresh(interval=120 * 1000, key="datarefresh")

# Custom CSS with vibrant colors
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;700&display=swap');
    body {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #F5F7FA 0%, #FF6F61 100%);
        color: #333;
        overflow-x: hidden;
    }
    .header {
        background: linear-gradient(90deg, #26A69A, #FF6F61);
        color: white;
        padding: 15px 30px;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .header h1 {
        font-size: 2.2em;
        font-weight: 600;
        margin: 0;
        display: inline;
        text-transform: uppercase;
    }
    .header img {
        width: 50px;
        height: 50px;
        vertical-align: middle;
        margin-right: 15px;
        border-radius: 50%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #26A69A;
        box-shadow: 0 6px 12px rgba(38, 166, 154, 0.2);
        text-align: center;
        margin: 15px 0;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: scale(1.05);
    }
    .chart-container {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #FF6F61;
        box-shadow: 0 6px 12px rgba(255, 111, 97, 0.2);
        margin-bottom: 25px;
    }
    .stButton > button {
        background: linear-gradient(45deg, #26A69A, #FF6F61);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 25px;
        font-weight: 500;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #1D7D74, #E65B4D);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
    }
    .stDownloadButton > button {
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 25px;
        font-weight: 500;
        transition: all 0.3s;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(45deg, #27ae60, #219653);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
    }
    .footer {
        text-align: center;
        font-size: 0.9em;
        color: #7f8c8d;
        padding: 15px;
        background: rgba(255, 255, 255, 0.8);
        border-top: 2px solid #FF6F61;
        border-radius: 10px;
        margin-top: 25px;
    }
    .help-tooltip {
        font-size: 0.9em;
        color: #26A69A;
        cursor: pointer;
        margin-left: 10px;
    }
    @media (max-width: 768px) {
        .header h1 {
            font-size: 1.6em;
        }
        .metric-card {
            padding: 15px;
        }
        .chart-container {
            padding: 15px;
        }
        .stButton > button, .stDownloadButton > button {
            padding: 10px 20px;
        }
    }
    </style>
""", unsafe_allow_html=True)
# Fetch data from API or CSV with retry logic
@st.cache_data
def load_data():
    if os.path.exists(CSV_FILE):
        try:
            os.remove(CSV_FILE)
        except Exception as e:
            st.warning(f"Could not clear CSV: {e}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            headers = {"X-API-Key": API_KEY}
            with st.spinner("Loading global sales data..."):
                response = requests.get(API_URL, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['transactions'])
                break
            elif response.status_code == 401:
                st.error("Invalid API key.")
                return pd.DataFrame()
            else:
                st.warning(f"API error: {response.status_code}. Retrying...")
                time.sleep(2)
        except requests.RequestException as e:
            st.warning(f"API unavailable: {e}. Retrying...")
            time.sleep(2)
    else:
        st.warning("API unavailable after retries. Generating sample data.")
        
    
    # Load from CSV as fallback if API fails
    if df.empty and os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        for col in ['revenue', 'cost', 'profit_loss', 'session_duration']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['sc-status'] = pd.to_numeric(df['sc-status'], errors='coerce', downcast='integer')
        df['demo_request_flag'] = pd.to_numeric(df['demo_request_flag'], errors='coerce', downcast='integer')
        df['is_anomaly'] = pd.to_numeric(df['is_anomaly'], errors='coerce', downcast='integer')
        df['converted'] = pd.to_numeric(df['converted'], errors='coerce', downcast='integer')
        df = df.dropna(subset=['timestamp', 'revenue', 'sc-status'])
    return df

# Perform anomaly detection using IsolationForest
def detect_anomalies(df):
    features = ['revenue', 'session_duration', 'sc-status']
    X = df[features].fillna(0)
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    df['is_anomaly'] = iso_forest.fit_predict(X)
    df['is_anomaly'] = df['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)  # Convert to 0 (normal) or 1 (anomaly)
    return df

# Perform user behavior clustering using K-means
def cluster_user_behavior(df):
    features = ['revenue', 'session_duration', 'converted']
    X = df[features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['behavior_cluster'] = kmeans.fit_predict(X_scaled)
    df['behavior_cluster'] = df['behavior_cluster'].map({0: 'Low Engagement', 1: 'Moderate Engagement', 2: 'High Engagement'})
    return df

# Generate PDF report
def generate_pdf_report(df):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFillColor(colors.darkblue)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 750, "AI-Solutions Global Sales Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y = 700
    c.drawString(100, y, f"Total Revenue: ${df['revenue'].sum():,.2f}")
    y -= 20
    c.drawString(100, y, f"Total Profit: ${df['profit_loss'].sum():,.2f}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Main dashboard
def main():
    # Header
    st.markdown('<div class="header"><img src="https://plus.unsplash.com/premium_photo-1682124651258-410b25fa9dc0?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTN8fGFydGlmaWNpYWwlMjBpbnRlbGxpZ2VuY2V8ZW58MHx8MHx8fDA%3D" alt="Logo"> <h1>AI-Solutions Global Sales</h1></div>', unsafe_allow_html=True)


    # Help Section
    with st.expander("Help & Instructions"):
        st.markdown("""
        - **Filters**: Use the sidebar to filter data by date, region, sales person, or request type. Click 'Reset Filters' to clear.
        - **Charts**: Interact with charts to explore trends and insights.
        - **Export**: Download reports in CSV or PDF format.
        - **Errors**: If data fails to load, try refreshing or uploading a CSV file.
        """)

    # Refresh Button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")
    with col2:
        if st.button("Refresh", key="manual_refresh"):
            st.rerun()

    df = load_data()
    if df.empty:
        st.error("No data available. Please check API or upload a CSV.")
        return

    # Apply anomaly detection and clustering
    df = detect_anomalies(df)
    df = cluster_user_behavior(df)

    # Sidebar with Filters and Upload
    st.sidebar.header("Filters")
    st.sidebar.markdown('<span class="help-tooltip" title="Select a date range to filter data">ℹ</span>', unsafe_allow_html=True)
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(datetime(2025, 1, 1), datetime(2025, 5, 18)),
        min_value=datetime(2025, 1, 1),
        max_value=datetime(2025, 5, 18),
        help="Select a date range for analysis."
    )
    regions = st.sidebar.multiselect("Region", options=sorted(df['region'].unique()), default=[])
    sales_people = st.sidebar.multiselect("Sales Person", options=sorted(df['sales_team_member'].unique()), default=[])
    request_types = st.sidebar.multiselect("Request Type", options=["All", "Requested", "Not Requested"], default=["All"])
    if st.sidebar.button("Reset Filters"):
        st.rerun()

    # Upload CSV Logs in Sidebar
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV Logs", type="csv", key="upload")
    if uploaded_file:
        with st.spinner("Processing upload..."):
            files = {'file': (uploaded_file.name, uploaded_file, 'text/csv')}
            headers = {"X-API-Key": API_KEY}
            response = requests.post(UPLOAD_URL, files=files, headers=headers)
            if response.status_code == 200:
                st.sidebar.success(response.json().get("message", "Upload successful!"))
            else:
                st.sidebar.error(response.json().get("error", "Upload failed."))

    # Apply filters
    filtered_df = df.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) &
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    if regions:
        filtered_df = filtered_df[filtered_df['region'].isin(regions)]
    if sales_people:
        filtered_df = filtered_df[filtered_df['sales_team_member'].isin(sales_people)]
    if request_types:
        if "All" not in request_types:
            if "Requested" in request_types and "Not Requested" in request_types:
                pass  # No filtering if both are selected
            elif "Requested" in request_types:
                filtered_df = filtered_df[filtered_df['demo_request_flag'] == 1]
            elif "Not Requested" in request_types:
                filtered_df = filtered_df[filtered_df['demo_request_flag'] == 0]

    # Tabs
    tabs = st.tabs(["Overview", "Financial Insights", "Customer & Temporal", "Technical Insights", "Data & AI"])

    with tabs[0]:  # Overview
        st.subheader("Global Sales Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">Total Revenue: ${:.2f}</div>'.format(filtered_df['revenue'].sum()), unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">Total Profit: ${:.2f}</div>'.format(filtered_df['profit_loss'].sum()), unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">Conversion Rate: {:.2f}%</div>'.format((filtered_df['converted'].mean() * 100)), unsafe_allow_html=True)
        with col4:
            top_sales_member = filtered_df.groupby('sales_team_member')['revenue'].sum().idxmax() if not filtered_df.empty else "N/A"
            top_sales_revenue = filtered_df.groupby('sales_team_member')['revenue'].sum().max() if not filtered_df.empty else 0
            st.markdown(f'<div class="metric-card">Top Sales Member: {top_sales_member}<br>(${top_sales_revenue:,.2f})</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # First row: Revenue by Region and Top 5 Countries
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue by Region")
            fig1 = px.bar(filtered_df.groupby('region')['revenue'].sum().reset_index(), x='region', y='revenue', title="Revenue by Region")
            fig1.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.subheader("Top 5 Countries")
            top_countries = filtered_df.groupby('country')['revenue'].sum().nlargest(5).reset_index()
            fig2 = px.pie(top_countries, names='country', values='revenue', title="Top 5 Countries")
            fig2.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig2, use_container_width=True)

        # Second row: Sales by Customer Type and Profit by Region
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sales by Customer Type")
            customer_type_sales = filtered_df.groupby('customer_type')['revenue'].sum().reset_index()
            fig12 = px.pie(customer_type_sales, names='customer_type', values='revenue', title="Sales by Customer Type")
            fig12.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig12, use_container_width=True)
        with col2:
            st.subheader("Profit by Region")
            fig13 = px.bar(filtered_df.groupby('region')['profit_loss'].sum().reset_index(), x='region', y='profit_loss', title="Profit by Region")
            fig13.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig13, use_container_width=True)

        # Third row: Conversion Rate by Region
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Conversion Rate by Region")
            conversion_by_region = filtered_df.groupby('region')['converted'].mean().reset_index()
            conversion_by_region['converted'] = conversion_by_region['converted'] * 100  # Convert to percentage
            fig14 = px.bar(conversion_by_region, x='region', y='converted', title="Conversion Rate by Region (%)")
            fig14.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig14, use_container_width=True)
        with col2:
            st.write("")  # Empty column for spacing
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:  # Financial Insights
        st.subheader("Financial Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">Avg Revenue: ${:.2f}</div>'.format(filtered_df['revenue'].mean()), unsafe_allow_html=True)
            fig3 = px.line(filtered_df.groupby(filtered_df['timestamp'].dt.date)['revenue'].sum().reset_index(), x='timestamp', y='revenue', title="Revenue Trend")
            fig3.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            st.markdown('<div class="metric-card">Profit Margin: {:.2f}%</div>'.format((filtered_df['profit_loss'].sum() / filtered_df['revenue'].sum() * 100)), unsafe_allow_html=True)
            fig4 = px.bar(filtered_df.groupby('sales_team_member')['profit_loss'].sum().reset_index(), x='sales_team_member', y='profit_loss', title="Profit by Team")
            fig4.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig4, use_container_width=True)

        # New Chart: Revenue Heatmap by Region and Time
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Revenue Heatmap by Region and Time")
            heatmap_data = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'region'])['revenue'].sum().reset_index()
            fig18 = px.density_heatmap(heatmap_data, x='timestamp', y='region', z='revenue', title="Revenue Heatmap by Region and Time")
            fig18.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig18, use_container_width=True)
        with col2:
            st.write("")  # Empty column for spacing

    with tabs[2]:  # Customer & Temporal
        st.subheader("Customer & Time Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">New Customers: {}</div>'.format(filtered_df[filtered_df['customer_type'] == 'new'].shape[0]), unsafe_allow_html=True)
            segment_counts = filtered_df['customer_segment'].value_counts().reset_index()
            segment_counts.columns = ['customer_segment', 'count']
            fig5 = px.pie(segment_counts, names='customer_segment', values='count', title="Customer Segments")
            fig5.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig5, use_container_width=True)
        with col2:
            st.markdown('<div class="metric-card">Avg Session: {:.2f}s</div>'.format(filtered_df['session_duration'].mean()), unsafe_allow_html=True)
            fig6 = px.scatter(filtered_df, x='timestamp', y='session_duration', color='customer_type', title="Session Duration Over Time")
            fig6.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig6, use_container_width=True)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown('<div class="metric-card">Total Demo Requests: {}</div>'.format(filtered_df['demo_request_flag'].sum()), unsafe_allow_html=True)
        with col2:
            demo_trend = filtered_df.groupby(filtered_df['timestamp'].dt.date)['demo_request_flag'].sum().reset_index()
            fig10 = px.line(demo_trend, x='timestamp', y='demo_request_flag', title="Demo Requests Over Time")
            fig10.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig10, use_container_width=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Revenue Trend by Customer Segment")
            revenue_by_segment = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'customer_segment'])['revenue'].sum().reset_index()
            fig15 = px.line(revenue_by_segment, x='timestamp', y='revenue', color='customer_segment', title="Revenue Trend by Customer Segment")
            fig15.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig15, use_container_width=True)
        with col2:
            st.write("")  # Empty column for spacing

        # New Chart: User Behavior Clustering
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("User Behavior Clusters")
            fig19 = px.scatter(filtered_df, x='session_duration', y='revenue', color='behavior_cluster', title="User Behavior Clusters")
            fig19.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig19, use_container_width=True)
        with col2:
            st.write("")  # Empty column for spacing

    with tabs[3]:  # Technical Insights
        st.subheader("Technical Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">Success Rate: {:.2f}%</div>'.format((filtered_df['sc-status'] == 200).mean() * 100), unsafe_allow_html=True)
            device_counts = filtered_df['device_type'].value_counts().reset_index()
            device_counts.columns = ['device_type', 'count']
            fig7 = px.bar(device_counts, x='device_type', y='count', title="Device Distribution")
            fig7.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig7, use_container_width=True)
        with col2:
            st.markdown('<div class="metric-card">OS Share: {}</div>'.format(filtered_df['operating_system'].mode()[0]), unsafe_allow_html=True)
            os_counts = filtered_df['operating_system'].value_counts().reset_index()
            os_counts.columns = ['operating_system', 'count']
            fig8 = px.pie(os_counts, names='operating_system', values='count', title="OS Distribution")
            fig8.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig8, use_container_width=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Session Duration by Device Type")
            fig16 = px.box(filtered_df, x='device_type', y='session_duration', title="Session Duration by Device Type")
            fig16.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig16, use_container_width=True)
        with col2:
            st.write("")  # Empty column for spacing

    with tabs[4]:  # Data & AI
        st.subheader("Data Quality & AI Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">Anomalies: {}</div>'.format(filtered_df['is_anomaly'].sum()), unsafe_allow_html=True)
            anomaly_counts = filtered_df['is_anomaly'].value_counts().reset_index()
            anomaly_counts.columns = ['is_anomaly', 'count']
            fig9 = px.bar(anomaly_counts, x='is_anomaly', y='count', title="Anomaly Detection")
            fig9.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig9, use_container_width=True)
        with col2:
            st.write("")  # Placeholder to maintain layout

        # Statistical Analysis
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Statistics")
            stats = filtered_df[['revenue', 'session_duration']].agg(['mean', 'median', 'std']).round(2)
            stats.loc['mode'] = filtered_df[['revenue', 'session_duration']].mode().iloc[0]
            st.table(stats)
        with col2:
            st.write("")

        # Anomaly Distribution by Region
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Anomaly Distribution by Region")
            anomaly_by_region = filtered_df.groupby('region')['is_anomaly'].sum().reset_index()
            fig17 = px.bar(anomaly_by_region, x='region', y='is_anomaly', title="Anomaly Distribution by Region")
            fig17.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig17, use_container_width=True)
        with col2:
            st.write("")  # Empty column for spacing

        # Raw Dataset Display
        st.subheader("Raw Dataset (Unfiltered)")
        st.dataframe(df, use_container_width=True)

        # Export Options
        col1, col2 = st.columns(2)  # Adjusted to 2 columns for CSV and PDF only
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button("Export to CSV", data=csv, file_name="global_sales.csv", mime="text/csv")
        with col2:
            pdf_buffer = generate_pdf_report(filtered_df)
            st.download_button("Export to PDF", data=pdf_buffer, file_name="global_sales_report.pdf", mime="application/pdf")

    st.markdown('<div class="footer">© 2025 AI-Solutions. All rights reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()