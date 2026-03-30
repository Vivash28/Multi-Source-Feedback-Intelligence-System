# 📊 Multi-Source Feedback Intelligence System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Tests-Pytest-green?style=for-the-badge&logo=pytest&logoColor=white"/>
</p>

<p align="center">
  A production-ready Python system that aggregates app reviews from multiple platforms,
  performs AI-powered sentiment analysis, detects trends, prioritises issues,
  and generates professional PDF reports — all in one interactive dashboard.
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Setup Guide](#-setup-guide)
- [API Setup Instructions](#-api-setup-instructions)
- [Running the Dashboard](#-running-the-dashboard)
- [Dashboard Screenshots](#-dashboard-screenshots)
- [Configuration](#-configuration)
- [Supported Apps](#-supported-apps)
- [Running Tests](#-running-tests)
- [Docker Deployment](#-docker-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

The **Multi-Source Feedback Intelligence System** is a complete,
production-grade application that helps product teams, developers,
and business analysts understand what users are saying about their apps
— across multiple platforms — in one unified dashboard.

Instead of manually reading thousands of reviews, this system:

- **Automatically fetches** reviews from Google Play Store, Apple App Store,
  and CSV survey exports
- **Analyses sentiment** using a state-of-the-art DistilBERT model
  (HuggingFace Transformers)
- **Detects trends** in user satisfaction over time and raises automatic alerts
  when sentiment drops significantly
- **Prioritises issues** by ranking complaint keywords using a
  frequency × sentiment strength score
- **Generates PDF reports** with charts, tables, and AI-driven recommendations
  — ready to share with your team

---

## ✨ Features

### 🔗 Multi-Source Data Aggregation
| Source | Method | Reviews |
|---|---|---|
| Google Play Store | `google-play-scraper` library | Up to 500 per fetch |
| Apple App Store | iTunes RSS JSON feed | Up to 500 per fetch |
| CSV Survey Export | File upload with auto column detection | Unlimited |

### 🤖 AI Sentiment Analysis
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Outputs per review:
  - `sentiment_label` → POSITIVE or NEGATIVE
  - `confidence_score` → 0.0 to 1.0
  - `sentiment_score` → -1.0 to +1.0
- Singleton pattern — model loads once, reused across all requests
- Batched inference for performance (32 reviews per batch)

### 📈 Trend Detection
- Daily average sentiment calculation
- 7-day rolling average smoothing
- Automatic **Sentiment Alert** when sentiment drops >20% over 3 days
- Gap-filling for missing days using linear interpolation

### 🔥 Issue Prioritisation
- Extracts complaint keywords from negative reviews
- Calculates Priority Score:
  ```
  Priority Score = Frequency × Average Negative Sentiment Strength
  ```
- Ranks top 15 issues automatically
- Filters out stop words and irrelevant tokens

### 📊 Interactive Dashboard
- Date range filter
- Source filter (Google Play / App Store / CSV)
- Sentiment filter (Positive / Negative)
- KPI cards (total reviews, positive %, avg rating, avg sentiment)
- Sentiment trend line chart
- Sentiment distribution pie chart
- Top Issues priority table
- Raw reviews data table
- One-click PDF report generation

### 📄 PDF Report Generation
- Cover page with company name and date range
- Executive summary KPI table
- Sentiment trend chart
- Sentiment distribution pie chart
- Top issues ranked table
- Data-driven recommendations section
- Page headers and footers
- Generated in seconds using native ReportLab charts

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   DATA SOURCES                      │
│  Google Play API   App Store RSS   CSV Upload       │
└──────────┬──────────────┬──────────────┬────────────┘
           │              │              │
           ▼              ▼              ▼
┌─────────────────────────────────────────────────────┐
│                    FETCHERS                         │
│  GooglePlayFetcher  AppStoreFetcher  CSVLoader      │
│  (retry + backoff)  (pagination)    (auto-map)      │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│              NORMALISED DATAFRAME                   │
│   review_id | source | review_text | rating | date  │
└──────────────────────────┬──────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  SENTIMENT   │  │    TREND     │  │    ISSUE     │
│  ANALYSER    │  │   ANALYSER   │  │ PRIORITIZER  │
│ (DistilBERT) │  │ (rolling avg)│  │ (keyword NLP)│
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                  │
       └─────────────────┼──────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│             STREAMLIT DASHBOARD                     │
│   Filters | KPIs | Charts | Issues | PDF Export    │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               PDF REPORT GENERATOR                 │
│   Cover | Summary | Charts | Issues | Recommendations│
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
feedback_intelligence/
│
├── app.py                        # Streamlit dashboard entry point
├── config.py                     # All configuration constants
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker container definition
├── README.md                     # This file
│
├── fetchers/                     # Data source integrations
│   ├── __init__.py
│   ├── google_play.py            # Google Play Store scraper
│   ├── app_store.py              # Apple App Store RSS fetcher
│   └── csv_loader.py             # CSV survey file importer
│
├── analysis/                     # NLP and analytics engine
│   ├── __init__.py
│   ├── sentiment.py              # HuggingFace sentiment pipeline
│   ├── trend_analysis.py         # Daily/rolling trend + alert detection
│   └── issue_prioritizer.py      # Keyword extraction + priority scoring
│
├── reporting/                    # Report generation
│   ├── __init__.py
│   └── pdf_generator.py          # ReportLab PDF report builder
│
├── utils/                        # Shared utilities
│   ├── __init__.py
│   └── logger.py                 # Rotating file + console logger
│
├── tests/                        # Unit test suite
│   ├── __init__.py
│   ├── test_sentiment.py         # Sentiment analyser tests
│   ├── test_trends.py            # Trend detection tests
│   └── test_prioritization.py   # Issue prioritiser tests
│
├── data/                         # Auto-created: raw data storage
├── reports/                      # Auto-created: saved PDF reports
└── .cache/                       # Auto-created: model cache
```

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.10+ |
| Dashboard | Streamlit | 1.31.0 |
| Sentiment Model | HuggingFace Transformers (DistilBERT) | 4.37.2 |
| Deep Learning | PyTorch | 2.1.2 |
| Data Processing | Pandas | 2.1.4 |
| Charts (Dashboard) | Plotly | 5.18.0 |
| Charts (PDF) | ReportLab Graphics | 4.1.0 |
| PDF Generation | ReportLab Platypus | 4.1.0 |
| Google Play Scraper | google-play-scraper | 1.2.4 |
| App Store Fetcher | requests (iTunes RSS) | 2.31.0 |
| Testing | pytest + pytest-cov | 7.4.4 |
| Containerisation | Docker | latest |

---

## ⚙️ Setup Guide

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or higher
- pip (Python package manager)
- Git (optional)
- 4GB+ RAM (for DistilBERT model)
- Internet connection (for fetching reviews and downloading model)

---

### Step 1 — Clone or Create the Project

```bash
# Option A: If you have the project as a zip, extract it
# Option B: Create manually
mkdir feedback_intelligence
cd feedback_intelligence
```

---

### Step 2 — Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

---

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **Note:** PyTorch (~2GB) will take several minutes to download.
> For a smaller CPU-only install, use:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

### Step 4 — Download the AI Model (One-time)

```bash
python -c "
from transformers import pipeline
pipe = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english'
)
print('Model downloaded successfully!')
"
```

This downloads ~250MB to your HuggingFace cache. Only needed once.

---

### Step 5 — Verify Installation

```bash
python -c "
import streamlit, pandas, transformers, torch, plotly, reportlab
print('All packages installed correctly!')
"
```

---

## 🔑 API Setup Instructions

### Google Play Store

**No API key required!**

The system uses the `google-play-scraper` library which scrapes the
Play Store directly. You only need the **app package name**.

```
How to find the package name:
1. Open Google Play Store in browser
2. Search for your app
3. Click on the app
4. Look at the URL:
   https://play.google.com/store/apps/details?id=COM.PACKAGE.NAME
                                                  ^^^^^^^^^^^^^^^^
                                                  This is your package name
```

**Examples:**
```
Spotify   →  com.spotify.music
WhatsApp  →  com.whatsapp
Zomato    →  com.application.zomato
Amazon    →  com.amazon.mShop.android.shopping
```

**Rate Limiting:**
The scraper has a built-in retry mechanism with exponential backoff.
If you hit rate limits, reduce the review count or add a delay between
fetches. Settings in `config.py`:
```python
FETCH_MAX_RETRIES: int = 3
FETCH_RETRY_BACKOFF: float = 2.0
```

---

### Apple App Store

**No API key required!**

The system uses Apple's **public iTunes RSS JSON feed** — completely free
and no authentication needed.

```
Endpoint:
https://itunes.apple.com/{country}/rss/customerreviews/
page={page}/id={app_id}/sortby=mostrecent/json
```

```
How to find the App Store ID:
1. Open App Store in browser: https://apps.apple.com
2. Search for your app
3. Click on the app
4. Look at the URL:
   https://apps.apple.com/us/app/spotify/id324684580
                                                 ^^^^^^^^^
                                                 This is your App ID
```

**Examples:**
```
Spotify   →  324684580
WhatsApp  →  310633997
Zomato    →  434535394
Amazon    →  297606951
```

**Limitations:**
- Apple RSS feed provides a maximum of 10 pages × 50 reviews = 500 reviews
- Only the most recent reviews are available
- Reviews are language/country specific

---

### CSV Survey Data

**No API required — just upload your file!**

The CSV loader automatically detects column names. Your CSV just needs
a column containing review text. Supported column name variations:

| Field | Accepted Column Names |
|---|---|
| Review Text | review_text, text, review, content, comment, feedback, body, message |
| Rating | rating, score, stars |
| Date | date, timestamp, created_at, submitted_at |
| Review ID | review_id, id, ID |
| Source | source, channel |

**Minimum required CSV:**
```csv
review_text
"The app is amazing!"
"Keeps crashing on startup"
"Good but needs improvement"
```

**Full featured CSV:**
```csv
review_id,review_text,rating,date,source
1,"The app is amazing!",5,2025-01-15,survey
2,"Keeps crashing on startup",1,2025-01-14,survey
3,"Good but needs improvement",3,2025-01-13,survey
```

---

### Environment Variables (Optional)

Create a `.env` file in the project root to customise behaviour:

```bash
# .env file
COMPANY_NAME=Your Company Name      # Shown on PDF cover page
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR
PDF_LOGO_PATH=path/to/logo.png      # Optional logo for PDF
```

---

## 🚀 Running the Dashboard

```bash
# Make sure virtual environment is activated
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Start the dashboard
streamlit run app.py
```

The dashboard will open automatically at:
```
http://localhost:8501
```

---

### Using the Dashboard

**Step 1 — Select a Data Source**
In the left sidebar:
- ✅ Check **Enable Google Play** and enter a package name
- ✅ Check **Enable App Store** and enter an App ID
- ✅ Upload a **CSV file** with review data

**Step 2 — Set Review Count**
Use the slider to select how many reviews to fetch (50–500).
More reviews = better analysis but slower fetch.

**Step 3 — Apply Filters**
- **Date Range** → Filter reviews by date
- **Source** → Show only specific platforms
- **Sentiment** → Show only Positive or Negative reviews

**Step 4 — Fetch & Analyse**
Click **🔄 Fetch & Analyse** to:
1. Fetch reviews from selected sources
2. Run AI sentiment analysis
3. Compute trend data
4. Extract and prioritise issues

**Step 5 — Explore Results**
- View KPI cards at the top
- Analyse sentiment trend chart
- Check sentiment distribution pie chart
- Review top issues priority table
- Browse raw reviews in the expandable table

**Step 6 — Download PDF Report**
Click **Generate PDF Report** → then **⬇️ Download PDF**

---

## 📸 Dashboard Screenshots

### Main Dashboard
<img width="849" height="778" alt="image" src="https://github.com/user-attachments/assets/3fcd222c-6270-42ca-8bd4-5f6a048b358a" />

<img width="1916" height="784" alt="image" src="https://github.com/user-attachments/assets/012a67ff-9a78-4b75-aeab-35210e998aad" />

### PDF Report Preview
<img width="321" height="702" alt="image" src="https://github.com/user-attachments/assets/13354ec7-5c52-4486-9a3d-bb582f950137" />



## ⚙️ Configuration

All settings are in `config.py`. Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `GOOGLE_PLAY_DEFAULT_COUNT` | 150 | Reviews to fetch from Play Store |
| `APP_STORE_DEFAULT_COUNT` | 150 | Reviews to fetch from App Store |
| `SENTIMENT_BATCH_SIZE` | 32 | Reviews per AI inference batch |
| `ROLLING_WINDOW_DAYS` | 7 | Days for rolling average |
| `SENTIMENT_DROP_THRESHOLD` | 0.20 | Alert trigger (20% drop) |
| `SENTIMENT_DROP_WINDOW_DAYS` | 3 | Days to measure drop over |
| `TOP_N_ISSUES` | 15 | Number of top issues to show |
| `MIN_KEYWORD_FREQ` | 3 | Min frequency for keyword inclusion |
| `DEFAULT_DATE_RANGE_DAYS` | 365 | Default date filter range |
| `FETCH_MAX_RETRIES` | 3 | API retry attempts |
| `FETCH_RETRY_BACKOFF` | 2.0 | Exponential backoff multiplier |

---

## 📱 Supported Apps

### 🛒 Shopping
| App | Google Play | App Store |
|---|---|---|
| Amazon | `com.amazon.mShop.android.shopping` | `297606951` |
| Flipkart | `com.flipkart.android` | `742044692` |
| Myntra | `com.myntra.android` | `907394059` |

### 🍔 Food & Delivery
| App | Google Play | App Store |
|---|---|---|
| Swiggy | `in.swiggy.android` | `989540920` |
| Zomato | `com.application.zomato` | `434535394` |
| Uber Eats | `com.ubercab.eats` | `1058959277` |

### 💰 Finance
| App | Google Play | App Store |
|---|---|---|
| Google Pay | `com.google.android.apps.nbu.paisa.user` | `1191175610` |
| PhonePe | `com.phonepe.app` | `1170055821` |
| PayPal | `com.paypal.android.p2pmobile` | `283646709` |

### 🎬 Entertainment
| App | Google Play | App Store |
|---|---|---|
| Netflix | `com.netflix.mediaclient` | `363590051` |
| Spotify | `com.spotify.music` | `324684580` |
| YouTube | `com.google.android.youtube` | `544007664` |

### 💬 Social
| App | Google Play | App Store |
|---|---|---|
| WhatsApp | `com.whatsapp` | `310633997` |
| Instagram | `com.instagram.android` | `389801252` |
| Telegram | `org.telegram.messenger` | `686449807` |

---

## 🧪 Running Tests

```bash
# Run all tests with coverage report
pytest tests/ --cov=. --cov-report=term-missing -v

# Run a specific test file
pytest tests/test_sentiment.py -v
pytest tests/test_trends.py -v
pytest tests/test_prioritization.py -v

# Run with HTML coverage report
pytest tests/ --cov=. --cov-report=html
# Open htmlcov/index.html in browser
```

### Test Coverage Targets

| Module | Coverage |
|---|---|
| `analysis/sentiment.py` | >85% |
| `analysis/trend_analysis.py` | >85% |
| `analysis/issue_prioritizer.py` | >85% |
| Overall | >80% |

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t feedback-intelligence .

# Run the container
docker run -p 8501:8501 feedback-intelligence

# Run with environment variables
docker run -p 8501:8501 \
  -e COMPANY_NAME="Acme Corp" \
  -e LOG_LEVEL=INFO \
  feedback-intelligence

# Run in background
docker run -d -p 8501:8501 --name feedback-app feedback-intelligence
```

Open `http://localhost:8501` in your browser.

### Docker Commands

```bash
# View running containers
docker ps

# View logs
docker logs feedback-app

# Stop container
docker stop feedback-app

# Remove container
docker rm feedback-app

# Remove image
docker rmi feedback-intelligence
```

---

## 🔧 Troubleshooting

### Common Issues & Fixes

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Missing file or wrong folder | Check all files exist in correct folders |
| `No reviews match filters` | Date range mismatch | Set date range to 2014–2026 to capture all |
| `PDF generation failed` | Missing kaleido | `pip install kaleido==0.2.1` |
| `Model loading slow` | First time download | Wait ~2 mins, cached after first run |
| `Google Play rate limit` | Too many requests | Reduce count or wait 60 seconds |
| `App not found` | Wrong package name | Double-check package name on Play Store URL |
| `CSV column not found` | Column name mismatch | Rename column to `review_text` |
| `Port 8501 in use` | Another Streamlit running | Use `streamlit run app.py --server.port 8502` |
| `Cache showing old data` | Streamlit cache | Click 🗑️ Clear Cache button and restart |







