import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import random
import hashlib
import pycountry

# Configuration
CSV_FILE = "web_server_logs.csv"
st.set_page_config(
    layout="wide",
    page_title="AI-Solutions Global Sales Dashboard",
    page_icon="https://www.freepik.com/free-vector/global-technology-concept-with-globe-circuit_12152352.htm"
)

# Auto-refresh every 120 seconds
st_autorefresh(interval=120 * 1000, key="datarefresh")

# Custom CSS
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

# Data configurations from web_server_logs.py
COUNTRIES = [country.name for country in pycountry.countries]
COUNTRY_REGIONS = {
    "Algeria": "Africa", "Angola": "Africa", "Benin": "Africa", "Botswana": "Africa", "Burkina Faso": "Africa",
    "Burundi": "Africa", "Cabo Verde": "Africa", "Cameroon": "Africa", "Central African Republic": "Africa",
    "Chad": "Africa", "Comoros": "Africa", "Congo": "Africa", "Democratic Republic of the Congo": "Africa",
    "Djibouti": "Africa", "Egypt": "Africa", "Equatorial Guinea": "Africa", "Eritrea": "Africa", "Eswatini": "Africa",
    "Ethiopia": "Africa", "Gabon": "Africa", "Gambia": "Africa", "Ghana": "Africa", "Guinea": "Africa",
    "Guinea-Bissau": "Africa", "Ivory Coast": "Africa", "Kenya": "Africa", "Lesotho": "Africa", "Liberia": "Africa",
    "Libya": "Africa", "Madagascar": "Africa", "Malawi": "Africa", "Mali": "Africa", "Mauritania": "Africa",
    "Mauritius": "Africa", "Mozambique": "Africa", "Namibia": "Africa", "Niger": "Africa", "Nigeria": "Africa",
    "Rwanda": "Africa", "Sao Tome and Principe": "Africa", "Senegal": "Africa", "Seychelles": "Africa",
    "Sierra Leone": "Africa", "Somalia": "Africa", "South Africa": "Africa", "South Sudan": "Africa",
    "Sudan": "Africa", "Tanzania": "Africa", "Togo": "Africa", "Tunisia": "Africa", "Uganda": "Africa",
    "Zambia": "Africa", "Zimbabwe": "Africa",
    "Afghanistan": "Asia", "Bahrain": "Asia", "Bangladesh": "Asia", "Bhutan": "Asia", "Brunei Darussalam": "Asia",
    "Cambodia": "Asia", "China": "Asia", "Cyprus": "Asia", "East Timor": "Asia", "India": "Asia", "Indonesia": "Asia",
    "Iran": "Asia", "Iraq": "Asia", "Israel": "Asia", "Japan": "Asia", "Jordan": "Asia", "Kazakhstan": "Asia",
    "Kuwait": "Asia", "Kyrgyzstan": "Asia", "Laos": "Asia", "Lebanon": "Asia", "Malaysia": "Asia", "Maldives": "Asia",
    "Mongolia": "Asia", "Myanmar": "Asia", "Nepal": "Asia", "North Korea": "Asia", "Oman": "Asia", "Pakistan": "Asia",
    "Palestine": "Asia", "Philippines": "Asia", "Qatar": "Asia", "Saudi Arabia": "Asia", "Singapore": "Asia",
    "South Korea": "Asia", "Sri Lanka": "Asia", "Syria": "Asia", "Taiwan": "Asia", "Tajikistan": "Asia",
    "Thailand": "Asia", "Turkey": "Asia", "Turkmenistan": "Asia", "United Arab Emirates": "Asia", "Uzbekistan": "Asia",
    "Vietnam": "Asia", "Yemen": "Asia",
    "Albania": "Europe", "Andorra": "Europe", "Austria": "Europe", "Belarus": "Europe", "Belgium": "Europe",
    "Bosnia and Herzegovina": "Europe", "Bulgaria": "Europe", "Croatia": "Europe", "Czechia": "Europe",
    "Denmark": "Europe", "Estonia": "Europe", "Finland": "Europe", "France": "Europe", "Germany": "Europe",
    "Greece": "Europe", "Hungary": "Europe", "Iceland": "Europe", "Ireland": "Europe", "Italy": "Europe",
    "Latvia": "Europe", "Liechtenstein": "Europe", "Lithuania": "Europe", "Luxembourg": "Europe", "Malta": "Europe",
    "Moldova": "Europe", "Monaco": "Europe", "Montenegro": "Europe", "Netherlands": "Europe", "North Macedonia": "Europe",
    "Norway": "Europe", "Poland": "Europe", "Portugal": "Europe", "Romania": "Europe", "Russia": "Europe",
    "San Marino": "Europe", "Serbia": "Europe", "Slovakia": "Europe", "Slovenia": "Europe", "Spain": "Europe",
    "Sweden": "Europe", "Switzerland": "Europe", "Ukraine": "Europe", "United Kingdom": "Europe",
    "Vatican City": "Europe",
    "Antigua and Barbuda": "North America", "Bahamas": "North America", "Barbados": "North America",
    "Belize": "North America", "Canada": "North America", "Costa Rica": "North America", "Cuba": "North America",
    "Dominica": "North America", "Dominican Republic": "North America", "El Salvador": "North America",
    "Grenada": "North America", "Guatemala": "North America", "Haiti": "North America", "Honduras": "North America",
    "Jamaica": "North America", "Mexico": "North America", "Nicaragua": "North America", "Panama": "North America",
    "Saint Kitts and Nevis": "North America", "Saint Lucia": "North America", "Saint Vincent and the Grenadines": "North America",
    "Trinidad and Tobago": "North America", "United States": "North America", "Aruba": "North America",
    "Curaçao": "North America", "Sint Maarten": "North America", "Saint Martin": "North America", "Anguilla": "North America",
    "Bermuda": "North America", "British Virgin Islands": "North America", "Cayman Islands": "North America",
    "Greenland": "North America", "Montserrat": "North America", "Puerto Rico": "North America",
    "Turks and Caicos Islands": "North America", "United States Virgin Islands": "North America",
    "Australia": "Oceania", "Fiji": "Oceania", "Kiribati": "Oceania", "Marshall Islands": "Oceania",
    "Micronesia": "Oceania", "Nauru": "Oceania", "New Zealand": "Oceania", "Palau": "Oceania",
    "Papua New Guinea": "Oceania", "Samoa": "Oceania", "Solomon Islands": "Oceania", "Tonga": "Oceania",
    "Tuvalu": "Oceania", "Vanuatu": "Oceania", "American Samoa": "Oceania", "Cook Islands": "Oceania",
    "French Polynesia": "Oceania", "Guam": "Oceania", "New Caledonia": "Oceania", "Niue": "Oceania",
    "Norfolk Island": "Oceania", "Northern Mariana Islands": "Oceania", "Pitcairn Islands": "Oceania",
    "Tokelau": "Oceania", "Wallis and Futuna": "Oceania",
    "Argentina": "South America", "Bolivia": "South America", "Brazil": "South America", "Chile": "South America",
    "Colombia": "South America", "Ecuador": "South America", "Guyana": "South America", "Paraguay": "South America",
    "Peru": "South America", "Suriname": "South America", "Uruguay": "South America", "Venezuela": "South America",
    "Falkland Islands": "South America", "French Guiana": "South America"
}
for c in COUNTRIES:
    if c not in COUNTRY_REGIONS:
        COUNTRY_REGIONS[c] = "Unknown"
REGION_WEIGHTS = {"Africa": 0.2, "Asia": 0.3, "Europe": 0.2, "North America": 0.15, "Oceania": 0.05, "South America": 0.1, "Unknown": 0.0}
COUNTRY_WEIGHTS = [REGION_WEIGHTS.get(COUNTRY_REGIONS[c], 0.0) / sum(len([k for k, v in COUNTRY_REGIONS.items() if v == COUNTRY_REGIONS[c]]) for c in COUNTRIES) for c in COUNTRIES]
SALES_TEAM = ["Ms Catherine Jones", "Mr Daniel Mafia", "Ms Deliah Canny", "Mr Jack Melting", "Ms Emma Rain", "Mr Lee Thompson"]
REQUEST_TYPES = {
    "job_request": "/api/jobs/submit",
    "demo_booking": "/api/demo/schedule",
    "ai_assistant_inquiry": "/api/assistant/inquire",
    "promotion": "/api/event",
    "registration": "/api/signup",
    "support": "/api/contact",
    "pricing": "/api/pricing",
    "download": "/api/download",
    "feedback": "/api/feedback",
    "training": "/api/training",
    "partnership": "/api/partnership",
    "case_study": "/api/case_study",
    "whitepaper": "/api/whitepaper",
    "content": "/api/blog",
    "faq": "/api/faq",
    "purchase": "/api/checkout"
}
JOB_TYPES = {
    "/api/jobs/submit/software": "software",
    "/api/jobs/submit/engineering": "engineering",
    "/api/jobs/submit/design": "design",
    "/api/jobs/submit/ai_development": "ai_development",
    "/api/jobs/submit/data_analytics": "data_analytics",
    "/api/jobs/submit/project_management": "project_management",
    "/api/jobs/submit/devops": "devops",
    "/api/jobs/submit/quality_assurance": "quality_assurance",
    "/api/jobs/submit/ui_ux": "ui_ux",
    "/api/jobs/submit/cybersecurity": "cybersecurity"
}
METHODS = ["GET", "POST"]
MARKETING_CHANNELS = ["direct", "email", "social"]
CHANNEL_WEIGHTS = [0.5, 0.3, 0.2]
CUSTOMER_TYPES = ["new", "returning"]
CUSTOMER_WEIGHTS = [0.6, 0.4]
DEVICE_TYPES = ["desktop", "mobile", "tablet"]
DEVICE_WEIGHTS = [0.5, 0.4, 0.1]
OPERATING_SYSTEMS = ["Windows", "macOS", "Linux", "iOS", "Android"]
OS_WEIGHTS = [0.4, 0.2, 0.1, 0.15, 0.15]
CUSTOMER_SEGMENTS = ["enterprise", "SMB", "individual"]
SEGMENT_WEIGHTS = [0.3, 0.5, 0.2]
STATUS_CODES = [200, 404, 500]
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 5, 18)

# Generate sample data matching web_server_logs.py
def generate_sample_data(n_rows=1000):
    np.random.seed(42)
    date_range = [START_DATE + timedelta(days=x) for x in range((END_DATE - START_DATE).days)]

    def hash_ip(ip):
        return hashlib.sha256(ip.encode()).hexdigest()[:16]

    def infer_job_type(url, request_type):
        if request_type != "job_request":
            return ""
        for pattern, job_type in JOB_TYPES.items():
            if pattern in url:
                return job_type
        return ""

    logs = []
    for _ in range(n_rows):
        timestamp = np.random.choice(date_range)
        ip_address = ".".join(map(str, (np.random.randint(0, 256, 4))))
        country = np.random.choice(COUNTRIES, p=COUNTRY_WEIGHTS)
        region = COUNTRY_REGIONS[country]
        sales_team_member = np.random.choice(SALES_TEAM)
        request_type = np.random.choice(list(REQUEST_TYPES.keys()))
        url = REQUEST_TYPES[request_type]
        if request_type == "job_request":
            job_suffix = np.random.choice(list(JOB_TYPES.keys())).split("/api/jobs/submit/")[1]
            url = f"{url}/{job_suffix}"
        method = np.random.choice(METHODS)
        status_code = np.random.choice(STATUS_CODES)
        session_duration = round(np.random.uniform(30, 600), 2)
        demo_request = 1 if request_type == "demo_booking" else np.random.choice([0, 1])
        user_agent = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{np.random.randint(500, 600)}.36"
        customer_type = np.random.choice(CUSTOMER_TYPES, p=CUSTOMER_WEIGHTS)
        marketing_channel = np.random.choice(MARKETING_CHANNELS, p=CHANNEL_WEIGHTS)
        device_type = np.random.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS)
        operating_system = np.random.choice(OPERATING_SYSTEMS, p=OS_WEIGHTS)
        customer_segment = np.random.choice(CUSTOMER_SEGMENTS, p=SEGMENT_WEIGHTS)
        job_type = infer_job_type(url, request_type)
        if request_type == "job_request" and job_type:
            revenue_ranges = {
                "software": (150, 500),
                "engineering": (120, 450),
                "design": (100, 400),
                "ai_development": (200, 600),
                "data_analytics": (150, 500),
                "project_management": (100, 350),
                "devops": (120, 400),
                "quality_assurance": (80, 300),
                "ui_ux": (100, 400),
                "cybersecurity": (150, 550)
            }
            revenue_range = revenue_ranges.get(job_type, (100, 500))
            revenue = round(np.random.uniform(*revenue_range), 2)
        else:
            revenue = round(
                np.random.uniform(50, 200) if request_type == "demo_booking" else
                np.random.uniform(20, 100) if request_type == "ai_assistant_inquiry" else
                np.random.uniform(200, 600) if request_type == "purchase" else
                0, 2
            )
        cost = round(revenue * np.random.uniform(0.6, 0.8), 2)
        profit_loss = round(revenue * np.random.uniform(0.5, 0.9) if revenue > 0 else 0, 2)
        converted = 1 if np.random.random() < 0.6 else 0

        logs.append({
            "timestamp": timestamp.isoformat(),
            "c-ip": hash_ip(ip_address),
            "cs-method": method,
            "cs-uri-stem": url,
            "sc-status": status_code,
            "region": region,
            "sales_team_member": sales_team_member,
            "request_type": request_type,
            "job_type": job_type,
            "demo_request_flag": demo_request,
            "session_duration": session_duration,
            "revenue": revenue,
            "cost": cost,
            "profit_loss": profit_loss,
            "country": country,
            "user_agent": user_agent,
            "customer_type": customer_type,
            "marketing_channel": marketing_channel,
            "device_type": device_type,
            "operating_system": operating_system,
            "customer_segment": customer_segment,
            "converted": converted,
            "is_anomaly": 0
        })

    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None).dt.tz_localize('UTC')
    df['revenue'] = df['revenue'].clip(lower=0)
    df['cost'] = df['cost'].clip(lower=0)
    df['profit_loss'] = df['profit_loss'].clip(lower=-df['revenue'])
    df['session_duration'] = df['session_duration'].clip(lower=0, upper=600)
    df = df.drop_duplicates(subset=['timestamp', 'c-ip', 'request_type'], keep='last')
    try:
        df.to_csv(CSV_FILE, index=False)
    except Exception as e:
        st.warning(f"Could not save sample data to CSV: {e}")
    return df

# Load data from CSV or generate sample data
@st.cache_data
def load_data():
    import os
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            st.info("Loaded data from CSV.")
        except Exception as e:
            st.warning(f"Could not load CSV: {e}. Generating sample data.")
            df = pd.DataFrame()
    else:
        st.info("Generating sample data.")
        df = pd.DataFrame()
    
    if df.empty:
        df = generate_sample_data()
    
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
    df['is_anomaly'] = df['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)
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
    st.markdown('<div class="header"><img src="https://plus.unsplash.com/premium_photo-1682124651258-410b25fa9dc0?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTN8fGFydGlmaWNpYWwlMjBpbnRlbGxpZ2VuY2V8ZW58MHx8MHx8fDA%3D" alt="Logo"> <h1>AI-Solutions Global Sales</h1></div>', unsafe_allow_html=True)

    with st.expander("Help & Instructions"):
        st.markdown("""
        - **Filters**: Use the sidebar to filter data by date, region, sales person, or request type. Click 'Reset Filters' to clear.
        - **Charts**: Interact with charts to explore trends and insights.
        - **Export**: Download reports in CSV or PDF format.
        - **Data Source**: The dashboard uses locally generated or uploaded data for reliability.
        """)

    col1, col2 = st.columns([5, 1])
    with col1:
        st.write("")
    with col2:
        if st.button("Refresh", key="manual_refresh"):
            st.rerun()

    df = load_data()
    if df.empty:
        st.error("No data available. Please upload a CSV or try again.")
        return

    df = detect_anomalies(df)
    df = cluster_user_behavior(df)

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

    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV Logs", type="csv", key="upload")
    if uploaded_file:
        with st.spinner("Processing upload..."):
            try:
                df_upload = pd.read_csv(uploaded_file)
                required_columns = ['timestamp', 'c-ip', 'cs-method', 'cs-uri-stem', 'sc-status', 'revenue', 'cost', 'profit_loss', 'country', 'converted']
                missing_columns = [col for col in required_columns if col not in df_upload.columns]
                if missing_columns:
                    st.sidebar.error(f"Invalid CSV format. Missing columns: {', '.join(missing_columns)}")
                else:
                    df_upload['c-ip'] = df_upload['c-ip'].apply(hash_ip)
                    df_upload = df_upload.fillna({
                        'revenue': 0, 'cost': 0, 'profit_loss': 0, 'session_duration': 0,
                        'converted': 0, 'demo_request_flag': 0, 'job_type': ''
                    })
                    df_upload['revenue'] = df_upload['revenue'].clip(lower=0)
                    df_upload['cost'] = df_upload['cost'].clip(lower=0)
                    df_upload['profit_loss'] = df_upload['profit_loss'].clip(lower=-df_upload['revenue'])
                    df_upload['session_duration'] = df_upload['session_duration'].clip(lower=0, upper=600)
                    df_upload = df_upload.drop_duplicates(subset=['timestamp', 'c-ip', 'request_type'], keep='last')
                    df_upload.to_csv(CSV_FILE, index=False)
                    st.sidebar.success("File processed successfully.")
            except Exception as e:
                st.sidebar.error(f"Failed to process CSV: {str(e)}")

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
                pass
            elif "Requested" in request_types:
                filtered_df = filtered_df[filtered_df['demo_request_flag'] == 1]
            elif "Not Requested" in request_types:
                filtered_df = filtered_df[filtered_df['demo_request_flag'] == 0]

    tabs = st.tabs(["Overview", "Financial Insights", "Customer & Temporal", "Technical Insights", "Data & AI"])

    with tabs[0]:
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

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Conversion Rate by Region")
            conversion_by_region = filtered_df.groupby('region')['converted'].mean().reset_index()
            conversion_by_region['converted'] = conversion_by_region['converted'] * 100
            fig14 = px.bar(conversion_by_region, x='region', y='converted', title="Conversion Rate by Region (%)")
            fig14.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig14, use_container_width=True)
        with col2:
            st.write("")
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
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

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Revenue Heatmap by Region and Time")
            heatmap_data = filtered_df.groupby([filtered_df['timestamp'].dt.date, 'region'])['revenue'].sum().reset_index()
            fig18 = px.density_heatmap(heatmap_data, x='timestamp', y='region', z='revenue', title="Revenue Heatmap by Region and Time")
            fig18.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig18, use_container_width=True)
        with col2:
            st.write("")

    with tabs[2]:
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
            st.write("")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("User Behavior Clusters")
            fig19 = px.scatter(filtered_df, x='session_duration', y='revenue', color='behavior_cluster', title="User Behavior Clusters")
            fig19.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig19, use_container_width=True)
        with col2:
            st.write("")

    with tabs[3]:
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
            st.write("")

    with tabs[4]:
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
            st.write("")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Basic Statistics")
            stats = filtered_df[['revenue', 'session_duration']].agg(['mean', 'median', 'std']).round(2)
            stats.loc['mode'] = filtered_df[['revenue', 'session_duration']].mode().iloc[0]
            st.table(stats)
        with col2:
            st.write("")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Anomaly Distribution by Region")
            anomaly_by_region = filtered_df.groupby('region')['is_anomaly'].sum().reset_index()
            fig17 = px.bar(anomaly_by_region, x='region', y='is_anomaly', title="Anomaly Distribution by Region")
            fig17.update_layout(height=300, paper_bgcolor="white", font_color="#333")
            st.plotly_chart(fig17, use_container_width=True)
        with col2:
            st.write("")

        st.subheader("Raw Dataset (Unfiltered)")
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button("Export to CSV", data=csv, file_name="global_sales.csv", mime="text/csv")
        with col2:
            pdf_buffer = generate_pdf_report(filtered_df)
            st.download_button("Export to PDF", data=pdf_buffer, file_name="global_sales_report.pdf", mime="application/pdf")

    st.markdown('<div class="footer">© 2025 AI-Solutions. All rights reserved.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
