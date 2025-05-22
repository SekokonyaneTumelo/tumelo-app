from flask import Flask, jsonify, request, render_template
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
import os
import pycountry
import hashlib
from sklearn.ensemble import IsolationForest

app = Flask(__name__)
faker = Faker()

# Configuration
CSV_FILE = "web_server_logs.csv"
LOG_COUNT = 50  # Reduced for real-time simulation
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime.now()  # Dynamic end date for real-time
API_KEY = "ai-solutions-key-2025"

# Data configurations
# Get all countries and territories with full names
COUNTRIES = [country.name for country in pycountry.countries]

# Categorize all countries and territories by region based on UN geoscheme
COUNTRY_REGIONS = {
    # Africa
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

    # Asia
    "Afghanistan": "Asia", "Bahrain": "Asia", "Bangladesh": "Asia", "Bhutan": "Asia", "Brunei Darussalam": "Asia",
    "Cambodia": "Asia", "China": "Asia", "Cyprus": "Asia", "East Timor": "Asia", "India": "Asia", "Indonesia": "Asia",
    "Iran": "Asia", "Iraq": "Asia", "Israel": "Asia", "Japan": "Asia", "Jordan": "Asia", "Kazakhstan": "Asia",
    "Kuwait": "Asia", "Kyrgyzstan": "Asia", "Laos": "Asia", "Lebanon": "Asia", "Malaysia": "Asia", "Maldives": "Asia",
    "Mongolia": "Asia", "Myanmar": "Asia", "Nepal": "Asia", "North Korea": "Asia", "Oman": "Asia", "Pakistan": "Asia",
    "Palestine": "Asia", "Philippines": "Asia", "Qatar": "Asia", "Saudi Arabia": "Asia", "Singapore": "Asia",
    "South Korea": "Asia", "Sri Lanka": "Asia", "Syria": "Asia", "Taiwan": "Asia", "Tajikistan": "Asia",
    "Thailand": "Asia", "Turkey": "Asia", "Turkmenistan": "Asia", "United Arab Emirates": "Asia", "Uzbekistan": "Asia",
    "Vietnam": "Asia", "Yemen": "Asia",

    # Europe
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

    # North America
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

    # Oceania
    "Australia": "Oceania", "Fiji": "Oceania", "Kiribati": "Oceania", "Marshall Islands": "Oceania",
    "Micronesia": "Oceania", "Nauru": "Oceania", "New Zealand": "Oceania", "Palau": "Oceania",
    "Papua New Guinea": "Oceania", "Samoa": "Oceania", "Solomon Islands": "Oceania", "Tonga": "Oceania",
    "Tuvalu": "Oceania", "Vanuatu": "Oceania", "American Samoa": "Oceania", "Cook Islands": "Oceania",
    "French Polynesia": "Oceania", "Guam": "Oceania", "New Caledonia": "Oceania", "Niue": "Oceania",
    "Norfolk Island": "Oceania", "Northern Mariana Islands": "Oceania", "Pitcairn Islands": "Oceania",
    "Tokelau": "Oceania", "Wallis and Futuna": "Oceania",

    # South America
    "Argentina": "South America", "Bolivia": "South America", "Brazil": "South America", "Chile": "South America",
    "Colombia": "South America", "Ecuador": "South America", "Guyana": "South America", "Paraguay": "South America",
    "Peru": "South America", "Suriname": "South America", "Uruguay": "South America", "Venezuela": "South America",
    "Falkland Islands": "South America", "French Guiana": "South America"
}

# Add fallback for any missing countries
for c in COUNTRIES:
    if c not in COUNTRY_REGIONS:
        COUNTRY_REGIONS[c] = "Unknown"

# Assign weights based on region
REGION_WEIGHTS = {"Africa": 0.2, "Asia": 0.3, "Europe": 0.2, "North America": 0.15, "Oceania": 0.05, "South America": 0.1, "Unknown": 0.0}
COUNTRY_WEIGHTS = [REGION_WEIGHTS.get(COUNTRY_REGIONS[c], 0.0) / sum(len([k for k, v in COUNTRY_REGIONS.items() if v == COUNTRY_REGIONS[c]]) for c in COUNTRIES) for c in COUNTRIES]

REGIONS = list(REGION_WEIGHTS.keys())
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
CHANNEL_WEIGHTS = [0.5, 0.3, 0.2]  # Fixed: Changed from [0.5, "0.3", "0.2"] to numeric values
CUSTOMER_TYPES = ["new", "returning"]
CUSTOMER_WEIGHTS = [0.6, 0.4]
DEVICE_TYPES = ["desktop", "mobile", "tablet"]
DEVICE_WEIGHTS = [0.5, 0.4, 0.1]
OPERATING_SYSTEMS = ["Windows", "macOS", "Linux", "iOS", "Android"]
OS_WEIGHTS = [0.4, 0.2, 0.1, 0.15, 0.15]
CUSTOMER_SEGMENTS = ["enterprise", "SMB", "individual"]
SEGMENT_WEIGHTS = [0.3, 0.5, 0.2]
STATUS_CODES = [200, 404, 500]

def hash_ip(ip):
    """Hash IP address for security."""
    return hashlib.sha256(ip.encode()).hexdigest()[:16]

def infer_job_type(url, request_type):
    """Infer job type from URL if request_type is job_request."""
    if request_type != "job_request":
        return ""
    for pattern, job_type in JOB_TYPES.items():
        if pattern in url:
            return job_type
    return ""  # Default to empty if no specific job type matches

def generate_log_entry():
    """Generate a single synthetic log entry with sales data."""
    timestamp = faker.date_time_between(start_date=END_DATE - timedelta(days=7), end_date=END_DATE)
    ip_address = faker.ipv4_public()
    country = random.choices(COUNTRIES, weights=COUNTRY_WEIGHTS, k=1)[0]
    region = COUNTRY_REGIONS[country]
    sales_team_member = random.choice(SALES_TEAM)
    request_type = random.choice(list(REQUEST_TYPES.keys()))
    url = REQUEST_TYPES[request_type]
    # Add job type specific suffix for job requests
    if request_type == "job_request":
        job_suffix = random.choice(list(JOB_TYPES.keys())).split("/api/jobs/submit/")[1]
        url = f"{url}/{job_suffix}"
    method = random.choice(METHODS)
    status_code = random.choice(STATUS_CODES)
    session_duration = round(random.uniform(30, 600), 2)  # 30-600 seconds
    demo_request = 1 if request_type == "demo_booking" else random.choice([0, 1])
    user_agent = faker.user_agent()
    customer_type = random.choices(CUSTOMER_TYPES, weights=CUSTOMER_WEIGHTS, k=1)[0]
    marketing_channel = random.choices(MARKETING_CHANNELS, weights=CHANNEL_WEIGHTS, k=1)[0]
    device_type = random.choices(DEVICE_TYPES, weights=DEVICE_WEIGHTS, k=1)[0]
    operating_system = random.choices(OPERATING_SYSTEMS, weights=OS_WEIGHTS, k=1)[0]
    customer_segment = random.choices(CUSTOMER_SEGMENTS, weights=SEGMENT_WEIGHTS, k=1)[0]
    job_type = infer_job_type(url, request_type)
    # Revenue now depends on job_type for job_request, or request_type otherwise
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
        revenue = round(random.uniform(*revenue_range), 2)
    else:
        revenue = round(
            random.uniform(50, 200) if request_type == "demo_booking" else
            random.uniform(20, 100) if request_type == "ai_assistant_inquiry" else
            random.uniform(200, 600) if request_type == "purchase" else
            0, 2
        )
    cost = round(revenue * random.uniform(0.6, 0.8), 2)
    profit_loss = round(revenue * random.uniform(0.5, 0.9) if revenue > 0 else 0, 2)  # Profit as 50-90% of revenue
    converted = 1 if random.random() < 0.6 else 0

    return {
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
        "is_anomaly": 0  # Placeholder, updated by detect_anomalies
    }

def detect_anomalies(logs):
    """Detect anomalies using IsolationForest."""
    df = pd.DataFrame(logs)
    features = df[['sc-status', 'session_duration', 'revenue', 'demo_request_flag']].fillna(0)
    model = IsolationForest(contamination=0.1, random_state=42)
    df['is_anomaly'] = model.fit_predict(features)
    df['is_anomaly'] = df['is_anomaly'].apply(lambda x: 1 if x == -1 else 0)
    return df.to_dict(orient="records")

def save_logs_to_csv(logs):
    """Append cleaned logs to CSV file."""
    df = pd.DataFrame(logs)
    if os.path.exists(CSV_FILE):
        existing_df = pd.read_csv(CSV_FILE)
        existing_df = existing_df.fillna({
            'revenue': 0, 'cost': 0, 'profit_loss': 0, 'session_duration': 0,
            'converted': 0, 'demo_request_flag': 0, 'job_type': ''
        })
        df = pd.concat([existing_df, df], ignore_index=True)
    df = df.drop_duplicates(subset=['timestamp', 'c-ip', 'request_type'], keep='last')
    df.to_csv(CSV_FILE, index=False)

def generate_sales_data():
    """Generate synthetic sales data with raw transactions and apply data cleaning."""
    logs = [generate_log_entry() for _ in range(LOG_COUNT)]
    logs = detect_anomalies(logs)
    df = pd.DataFrame(logs)

    # Data Cleaning Steps
    df = df[df['is_anomaly'] == 0]

    df['revenue'] = df['revenue'].clip(lower=0)
    df['cost'] = df['cost'].clip(lower=0)
    df['profit_loss'] = df['profit_loss'].clip(lower=-df['revenue'])
    df['session_duration'] = df['session_duration'].clip(lower=0, upper=600)

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None).dt.tz_localize('UTC').dt.strftime('%Y-%m-%dT%H:%M:%S%Z')

    df = df.drop_duplicates(subset=['timestamp', 'c-ip', 'request_type'], keep='last')

    response = {
        "transactions": df.to_dict(orient='records')
    }
    return response

@app.route("/", methods=["GET"])
def home():
    """Serve the homepage using index.html."""
    return render_template("index.html")

@app.route("/api/data", methods=["GET"])
def get_data():
    """Generate and return synthetic sales data for real-time dashboard."""
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401
    response = generate_sales_data()
    save_logs_to_csv(response['transactions'])
    return jsonify(response)

@app.route("/api/logs", methods=["GET"])
def get_logs():
    """Alias for /api/data to maintain compatibility."""
    return get_data()

@app.route("/api/logs/recent", methods=["GET"])
def get_recent_logs():
    """Return recent logs from CSV with cleaning."""
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df = df.fillna({
            'revenue': 0, 'cost': 0, 'profit_loss': 0, 'session_duration': 0,
            'converted': 0, 'demo_request_flag': 0, 'job_type': ''
        })
        df['revenue'] = df['revenue'].clip(lower=0)
        df['cost'] = df['cost'].clip(lower=0)
        df['profit_loss'] = df['profit_loss'].clip(lower=-df['revenue'])
        df['session_duration'] = df['session_duration'].clip(lower=0, upper=600)
        df = df.drop_duplicates(subset=['timestamp', 'c-ip', 'request_type'], keep='last')
        recent_logs = df.tail(LOG_COUNT).to_dict(orient="records")
        return jsonify({"transactions": recent_logs})
    return jsonify({"transactions": []})

@app.route("/api/upload", methods=["POST"])
def upload_logs():
    """Process uploaded CSV logs with cleaning."""
    api_key = request.headers.get("X-API-Key")
    if api_key != API_KEY:
        return jsonify({"error": "Invalid API key"}), 401
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            required_columns = ['timestamp', 'c-ip', 'cs-method', 'cs-uri-stem', 'sc-status', 'revenue', 'cost', 'profit_loss', 'country', 'converted']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({"error": f"Invalid CSV format. Missing columns: {', '.join(missing_columns)}"}), 400
            df['c-ip'] = df['c-ip'].apply(hash_ip)
            df = df.fillna({
                'revenue': 0, 'cost': 0, 'profit_loss': 0, 'session_duration': 0,
                'converted': 0, 'demo_request_flag': 0, 'job_type': ''
            })
            df['revenue'] = df['revenue'].clip(lower=0)
            df['cost'] = df['cost'].clip(lower=0)
            df['profit_loss'] = df['profit_loss'].clip(lower=-df['revenue'])
            df['session_duration'] = df['session_duration'].clip(lower=0, upper=600)
            df = df.drop_duplicates(subset=['timestamp', 'c-ip', 'request_type'], keep='last')
            logs = df.to_dict(orient="records")
            logs = detect_anomalies(logs)
            logs = [log for log in logs if log['is_anomaly'] == 0]
            save_logs_to_csv(logs)
            return jsonify({"message": f"File processed successfully. {len(logs)} records added."})
        except Exception as e:
            return jsonify({"error": f"Failed to process CSV: {str(e)}"}), 400
    return jsonify({"error": "File must be CSV"}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)