# GridLock Intelligence Platform

**AI-powered traffic enforcement intelligence for Bengaluru Traffic Police**

Built on 298,282 real violation records (Nov 2023 – Apr 2024), GridLock tells commanders *where* violations will surge, *when* to be there, *how* to get there, and *how much* revenue each deployment will recover.

---

## Features

| Tab | What It Does |
|---|---|
| 🏠 Executive | 30-second pitch for any stakeholder — animated stats, plain-English findings, Officer ROI |
| 📊 Overview | City-wide KPIs, violation type mix, monthly trend, AI intelligence feed |
| 🗺️ Hotspot Map | Live density heatmap of 298k violations across Bengaluru |
| ⛌ Junction Analysis | Ranks every intersection by danger — identifies priority AM/PM deployment points |
| 🎯 Enforcement Planner | CII-ranked zone table + LP Optimizer allocates officers to maximize fine recovery |
| 📈 Temporal Analysis | Prophet forecast predicts surge windows 24 hours ahead |
| 🔍 Zone Explorer | Deep-dive any zone — RandomForest risk classifier + anomaly detection |
| ⚡ Command Intel | IsolationForest anomaly alerts + SCITA camera blind spot map |
| 📋 Impact Report | Auto-generated PDF enforcement brief with all findings |
| 🤖 AI Assistant | Natural language Q&A powered by Llama 3.3-70B (Groq) |
| 🛣️ Route Optimizer | A* pathfinding routes officers through the city avoiding high-violation corridors |

---

## Tech Stack

- **Frontend:** Streamlit 1.38
- **Data:** Pandas, NumPy
- **Visualization:** Plotly
- **ML Models:** scikit-learn (IsolationForest, RandomForestClassifier), Prophet (forecasting), scipy (LP optimizer, Gaussian smoothing)
- **AI Assistant:** Groq API — Llama 3.3-70B Versatile
- **Pathfinding:** A* on an 80×80 violation-density grid with Gaussian smoothing
- **PDF Export:** fpdf2

---

## Danger Score (CII) Formula

Every zone is scored using the **Congestion Impact Index**:

```
CII = 0.35 × Violation Frequency
    + 0.30 × Junction Density
    + 0.20 × Severity Score
    + 0.15 × Peak Hour Rate
```

Zones scoring above 0.6 are classified **Critical**.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key

The AI Assistant tab requires a Groq API key (free at [console.groq.com](https://console.groq.com)).

Set it as an environment variable:

```bash
# Windows
setx GROQ_API_KEY "your_key_here"

# macOS / Linux
export GROQ_API_KEY="your_key_here"
```

> The AI Assistant also accepts the key directly in the app UI — no env var needed if you paste it there.

### 3. Add the dataset

Place the violation CSV file in the project root:

```
gridlock2/
  jan to may police violation_anonymized791b166.csv
  app.py
  ...
```

### 4. Run the app

```bash
streamlit run app.py --server.port 8502
```

Then open **http://localhost:8502**

**Windows shortcut:** run `start.ps1` — it loads the API key from your environment and launches the app automatically.

---

## Dataset

- **Source:** Bengaluru Traffic Police violation records (anonymized)
- **Period:** November 2023 – April 2024
- **Records:** 298,282 violations
- **Fields:** Date/time, location (lat/lon), station, violation type, vehicle type, junction flag, SCITA camera flag, fine amount

The CSV is excluded from this repository due to file size. Contact the team for access.

---

## Project Structure

```
gridlock2/
├── app.py              # Main Streamlit application (4,000+ lines)
├── requirements.txt    # Python dependencies
├── start.ps1           # Windows launcher script
├── .streamlit/
│   └── config.toml     # Dark theme + server config
└── README.md
```

---

## Team

Built for the Smart City Hackathon — Bengaluru, 2026.
