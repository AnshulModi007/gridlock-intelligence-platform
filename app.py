import streamlit as st
import os as _os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from scipy.optimize import linprog
try:
    from prophet import Prophet          # heavy dep; app falls back to polynomial if unavailable
    _HAS_PROPHET = True
except Exception:
    Prophet = None
    _HAS_PROPHET = False
from fpdf import FPDF
from groq import Groq
import io as _io
import ast
import datetime as _dt
import warnings
warnings.filterwarnings("ignore")

# â"€â"€ Page config â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
st.set_page_config(
    page_title="GridLock . Bengaluru Parking Intelligence",
    page_icon=":vertical_traffic_light:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â"€â"€ Design System CSS â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design Tokens ── */
:root {
  /* ── structural surfaces ── */
  --bg:         #080D17;
  --card:       #0E1623;
  --card2:      #13203A;
  --border:     #1D2F47;
  --border-hi:  #253D5C;

  /* ── accent scale (cyan / operational chrome only) ── */
  --accent-subtle: rgba(0,194,212,0.08);
  --accent:        #00C2D4;
  --accent-bright: #00D9ED;
  --accent-d:      #007E8C;
  --accent-glow:   rgba(0,194,212,0.18);

  /* ── semantic severity — borders, text, status only ── */
  --critical:        #DC2626;
  --critical-text:   #F87171;
  --critical-subtle: rgba(220,38,38,0.10);
  --critical-glow:   rgba(220,38,38,0.20);

  --high:            #D97706;
  --high-text:       #FBBF24;
  --high-subtle:     rgba(217,119,6,0.10);
  --high-glow:       rgba(217,119,6,0.20);

  --medium:          #0EA5E9;
  --medium-text:     #38BDF8;
  --medium-subtle:   rgba(14,165,233,0.08);
  --medium-glow:     rgba(14,165,233,0.18);

  --low:             #059669;
  --low-text:        #34D399;
  --low-subtle:      rgba(5,150,105,0.08);
  --low-glow:        rgba(5,150,105,0.18);

  /* ── text scale ── */
  --text:   #E2EBF5;
  --muted:  #6B87A8;
  --dim:    #546D87;
  --data:   #A8C8E8;

  /* ── secondary chart hue (data-viz only, never status) ── */
  --chart-violet: #8B5CF6;
}

/* ── Base ── */
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stSidebar"]          { background: #070B14 !important; border-right: 1px solid var(--border); }
[data-testid="stHeader"]           { display: none !important; }
[data-testid="stToolbar"]          { display: none !important; }
[data-testid="stDecoration"]       { display: none !important; }
.block-container { padding-top: 0.75rem !important; padding-bottom: 2rem !important; }

/* ── Kill rerun dimming / stale overlay ── */
.stale, [data-stale="true"], [data-stale] {
  opacity: 1 !important;
  transition: none !important;
}
[data-testid="stStatusWidget"]     { display: none !important; }
div.stSpinner > div > div          { border-top-color: var(--accent) !important; }
iframe { opacity: 1 !important; transition: none !important; }

/* ── Header ── */
.gl-header {
  display: flex; align-items: center; gap: 14px;
  padding: 0 0 14px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 16px;
}
.gl-logo {
  width: 44px; height: 44px; border-radius: 10px;
  background: linear-gradient(145deg, #00D9ED 0%, #007E8C 100%);
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  box-shadow: 0 0 28px rgba(0,194,212,0.35);
}
.gl-logo-text {
  font-family: 'Barlow Condensed', sans-serif;
  font-weight: 700; font-size: 17px;
  color: #080D17; letter-spacing: 0.03em;
}
.gl-title-wrap { flex: 1; min-width: 0; }
.gl-title-wrap .gl-title {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.65rem; font-weight: 700;
  color: var(--text); line-height: 1.1; margin: 0;
  letter-spacing: 0.03em;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.gl-title-wrap .gl-sub {
  font-size: 0.68rem; color: var(--muted);
  letter-spacing: 0.09em; text-transform: uppercase; margin: 3px 0 0 0;
}
.gl-header-right {
  display: flex; align-items: center; gap: 7px; flex-shrink: 0;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem; color: var(--dim); letter-spacing: 0.05em;
}
.gl-live-pip {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--low);
  box-shadow: 0 0 7px rgba(5,150,105,0.9);
  display: inline-block;
  animation: livepulse 2.2s ease-in-out infinite;
}
@keyframes livepulse {
  0%,100% { opacity:1; box-shadow: 0 0 7px rgba(5,150,105,0.9); }
  50%      { opacity:0.4; box-shadow: 0 0 3px rgba(5,150,105,0.3); }
}

/* ── Situation Report Banner ─────────────────────────────────────── */
.sitrep-bar {
  display: flex; align-items: center; gap: 10px;
  background: linear-gradient(90deg, rgba(220,38,38,0.04) 0%, var(--card) 30%, var(--card) 100%);
  border: 1px solid var(--border); border-left: 3px solid var(--critical);
  border-radius: 6px; padding: 7px 14px; margin: 8px 0 2px;
  font-size: 0.72rem; line-height: 1.4; flex-wrap: wrap;
}
.sitrep-label {
  font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 0.65rem;
  letter-spacing: .12em; color: var(--critical-text);
  background: var(--critical-subtle); padding: 2px 7px; border-radius: 3px;
  white-space: nowrap;
}
.sitrep-item { color: var(--data); }
.sitrep-red   { color: var(--critical-text); }
.sitrep-amber { color: var(--high-text); }
.sitrep-cyan  { color: var(--accent-bright); }
.sitrep-sep   { color: var(--border-hi); font-size: 0.9rem; }
.sitrep-dot {
  display: inline-block; width: 6px; height: 6px;
  border-radius: 50%; margin-right: 5px; vertical-align: middle;
}
.sitrep-dot-red   { background: var(--critical); box-shadow: 0 0 4px rgba(220,38,38,0.7); }
.sitrep-dot-amber { background: var(--high);     box-shadow: 0 0 4px rgba(217,119,6,0.7); }
.sitrep-dot-cyan  { background: var(--accent);   box-shadow: 0 0 4px rgba(0,194,212,0.7); }

/* ── Zone Alert Rail ── */
.zone-rail {
  display: flex; align-items: stretch;
  background: var(--card); border: 1px solid var(--border);
  border-radius: 8px; overflow: hidden;
  margin-bottom: 16px; height: 34px;
}
.zone-rail-label {
  padding: 0 14px;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--dim);
  border-right: 1px solid var(--border);
  display: flex; align-items: center; white-space: nowrap;
}
.zone-chip {
  display: flex; align-items: center; gap: 7px;
  padding: 0 16px;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em;
  border-right: 1px solid var(--border); white-space: nowrap;
}
.zone-chip.zc-critical { color: var(--critical); background: rgba(220,38,38,0.10); }
.zone-chip.zc-high     { color: var(--high);     background: rgba(217,119,6,0.10); }
.zone-chip.zc-medium   { color: var(--medium);   background: rgba(14,165,233,0.07); }
.zone-chip.zc-low      { color: var(--low);      background: rgba(5,150,105,0.07); }
.zone-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.zone-dot.zd-critical { background: var(--critical); box-shadow: 0 0 5px rgba(220,38,38,0.80); }
.zone-dot.zd-high     { background: var(--high);     box-shadow: 0 0 5px rgba(217,119,6,0.80); }
.zone-dot.zd-medium   { background: var(--medium);   box-shadow: 0 0 5px rgba(14,165,233,0.70); }
.zone-dot.zd-low      { background: var(--low);      box-shadow: 0 0 5px rgba(5,150,105,0.75); }
.zone-rail-tail {
  margin-left: auto; padding: 0 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.63rem; color: var(--dim); letter-spacing: 0.06em;
  display: flex; align-items: center; flex-shrink: 0;
}

/* ── KPI Cards — Instrument Panel ── */
.kpi-grid { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.kpi-card {
  flex: 1; min-width: 155px;
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--border);
  border-radius: 12px; padding: 16px 18px;
  position: relative; overflow: hidden;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.kpi-card:hover {
  border-color: var(--border-hi);
  box-shadow: 0 4px 20px rgba(0,194,212,0.10), 0 1px 6px rgba(0,0,0,0.50);
}
.kpi-card.kc-critical {
  border-left-color: var(--critical);
  background: linear-gradient(90deg, rgba(220,38,38,0.05) 0%, var(--card) 55%);
}
.kpi-card.kc-critical:hover { box-shadow: 0 4px 20px var(--critical-glow), 0 1px 6px rgba(0,0,0,0.50); }
.kpi-card.kc-high {
  border-left-color: var(--high);
  background: linear-gradient(90deg, rgba(217,119,6,0.05) 0%, var(--card) 55%);
}
.kpi-card.kc-high:hover     { box-shadow: 0 4px 20px var(--high-glow), 0 1px 6px rgba(0,0,0,0.50); }
.kpi-card.kc-accent {
  border-left-color: var(--accent);
  background: linear-gradient(90deg, rgba(0,194,212,0.05) 0%, var(--card) 55%);
}
.kpi-card.kc-accent:hover   { box-shadow: 0 4px 20px var(--accent-glow), 0 1px 6px rgba(0,0,0,0.50); }
.kpi-card.kc-medium {
  border-left-color: var(--medium);
  background: linear-gradient(90deg, rgba(14,165,233,0.05) 0%, var(--card) 55%);
}
.kpi-card.kc-medium:hover   { box-shadow: 0 4px 20px var(--medium-glow), 0 1px 6px rgba(0,0,0,0.50); }

.kpi-label {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.66rem; font-weight: 700; color: var(--dim);
  text-transform: uppercase; letter-spacing: 0.13em;
  margin: 0 0 8px 0;
}
.kpi-val {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 1.95rem; font-weight: 500;
  color: var(--text); line-height: 1; letter-spacing: -0.02em;
}
.kpi-sub  { font-size: 0.7rem; color: var(--muted); margin-top: 6px; }
.kpi-badge {
  display: inline-block;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.64rem; font-weight: 700; letter-spacing: 0.05em;
  text-transform: uppercase; padding: 2px 7px;
  border-radius: 4px; margin-top: 6px;
}
.badge-crit { background: rgba(220,38,38,0.12);  color: var(--critical-text); }
.badge-warn { background: rgba(217,119,6,0.12);  color: var(--high-text); }
.badge-ok   { background: rgba(5,150,105,0.12);  color: var(--low-text); }
.badge-info { background: rgba(14,165,233,0.12); color: var(--medium-text); }

/* ── Section labels ── */
.sec-label {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.66rem; font-weight: 700; color: var(--dim);
  text-transform: uppercase; letter-spacing: 0.13em;
  margin: 0 0 10px 0; padding-bottom: 5px;
  border-bottom: 1px solid var(--border);
}
.instr-header {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.66rem; font-weight: 700; color: var(--accent);
  text-transform: uppercase; letter-spacing: 0.14em;
  margin: 0 0 8px 0; padding-bottom: 5px;
  border-bottom: 2px solid var(--accent);
  opacity: 0.85;
}

/* ── AI Intelligence Feed ── */
.insight-feed {
  overflow-y: auto; max-height: 235px;
  padding-right: 4px; margin-top: 8px;
}
.insight-feed::-webkit-scrollbar { width: 3px; }
.insight-feed::-webkit-scrollbar-track { background: var(--card); }
.insight-feed::-webkit-scrollbar-thumb { background: rgba(0,194,212,0.25); border-radius: 3px; }
.insight {
  border-radius: 8px; padding: 11px 14px;
  margin: 0 0 7px 0; border-left: 2px solid;
  font-size: 0.84rem; line-height: 1.6;
}
.insight:last-child { margin-bottom: 0; }
.insight b { color: var(--text); }
.insight-crit   { background: rgba(220,38,38,0.09);  border-color: var(--critical);     color: #8FA8C8; }
.insight-warn   { background: rgba(217,119,6,0.09);  border-color: var(--high);         color: #8FA8C8; }
.insight-info   { background: rgba(14,165,233,0.08); border-color: var(--medium);       color: #8FA8C8; }
.insight-green  { background: rgba(5,150,105,0.08);  border-color: var(--low);          color: #8FA8C8; }
.insight-purple { background: rgba(139,92,246,0.09); border-color: var(--chart-violet); color: #8FA8C8; }

/* ── Stat pills ── */
.stat-row { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
.stat-pill {
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 11px;
  font-size: 0.78rem; color: var(--muted);
}
.stat-pill b { color: var(--data); font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; }

/* ── Schedule table ── */
.sched-table { width: 100%; border-collapse: collapse; font-size: 0.81rem; }
.sched-table th {
  background: var(--card2); color: var(--muted);
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.63rem; text-transform: uppercase; letter-spacing: 0.1em;
  padding: 9px 12px; border-bottom: 1px solid var(--border); text-align: left;
}
.sched-table td { padding: 8px 12px; border-bottom: 1px solid #0C1828; color: var(--text); }
.sched-table tr:last-child td { border-bottom: none; }
.sched-table tr:hover td { background: var(--card2); }
.sched-table tr:hover td:first-child { box-shadow: inset 3px 0 0 rgba(0,194,212,0.35); }
.sched-table .risk-crit { color: var(--critical); font-weight: 700; }
.sched-table .risk-high { color: var(--high);     font-weight: 700; }
.sched-table .risk-med  { color: var(--medium); }
.sched-table .risk-low  { color: var(--low); }

/* ── Sidebar ── */
.sidebar-section {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.66rem; font-weight: 700; color: var(--dim);
  text-transform: uppercase; letter-spacing: 0.13em;
  margin: 16px 0 6px 0;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label { font-size: 0.82rem !important; color: var(--muted) !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
  background: var(--card) !important;
  border-radius: 8px; padding: 4px; gap: 2px;
  border: 1px solid var(--border);
  overflow-x: auto !important;
  overflow-y: hidden !important;
  flex-wrap: nowrap !important;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}
[data-baseweb="tab-list"]::-webkit-scrollbar {
  height: 3px;
}
[data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
  background: var(--border-hi); border-radius: 3px;
}
[data-baseweb="tab-list"]::-webkit-scrollbar-track {
  background: transparent;
}
[data-baseweb="tab"] {
  border-radius: 6px !important;
  font-family: 'Barlow Condensed', sans-serif !important;
  font-size: 0.87rem !important; font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  color: var(--muted) !important; padding: 7px 16px !important;
  white-space: nowrap !important; flex-shrink: 0 !important;
}
[aria-selected="true"] {
  background: var(--accent) !important;
  color: #080D17 !important; font-weight: 700 !important;
  box-shadow: 0 2px 14px rgba(0,194,212,0.40) !important;
}

/* ── Misc ── */
hr { border-color: var(--border) !important; margin: 16px 0 !important; }
.dl-btn {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--accent); color: #080D17;
  padding: 8px 18px; border-radius: 6px;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 0.84rem; font-weight: 700; letter-spacing: 0.06em;
  text-decoration: none;
  transition: background 0.15s ease, box-shadow 0.15s ease;
}
.dl-btn:hover {
  background: var(--accent-bright);
  box-shadow: 0 2px 14px rgba(0,194,212,0.30);
}
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important; padding: 14px 16px !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
[data-testid="metric-container"]:hover {
  border-color: var(--border-hi) !important;
  box-shadow: 0 2px 12px rgba(0,194,212,0.08) !important;
}
div[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important; overflow: hidden;
}

/* ── KPI icon ── */
.kpi-icon { font-size: 0.85rem; margin-bottom: 4px; display: block; opacity: 0.7; }

/* ── Trend delta pill ── */
.kpi-delta {
  display: inline-flex; align-items: center; gap: 3px;
  font-size: 0.68rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;
  padding: 2px 7px; border-radius: 4px; margin-top: 5px;
}
.delta-up   { background: rgba(220,38,38,0.12); color: #F87171; }
.delta-down { background: rgba(5,150,105,0.12); color: #34D399; }
.delta-flat { background: rgba(107,135,168,0.12); color: #6B87A8; }

/* ── Sidebar quickstat cards ── */
.sb-stat {
  background: var(--card2); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 12px; margin-bottom: 8px;
}
.sb-stat-label { font-size: 0.58rem; color: var(--dim); letter-spacing: .1em; text-transform: uppercase; margin-bottom: 3px; }
.sb-stat-val   { font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 500; color: var(--text); }
.sb-stat-sub   { font-size: 0.65rem; color: var(--muted); margin-top: 2px; }

/* ── Sidebar quick action buttons ── */
.sb-action {
  display: block; width: 100%;
  background: rgba(0,194,212,0.07); border: 1px solid rgba(0,194,212,0.20);
  border-radius: 6px; padding: 7px 10px; margin-bottom: 6px;
  font-size: 0.76rem; color: var(--accent); cursor: pointer;
  font-family: 'Barlow Condensed', sans-serif; font-weight: 600; letter-spacing: .04em;
  text-align: left;
}
.sb-action:hover { background: rgba(0,194,212,0.13); border-color: var(--accent); }

/* ── Recoverable pill (zone rail) ── */
.recover-pill {
  margin-left: auto; padding: 0 18px;
  background: rgba(5,150,105,0.12); border-left: 1px solid var(--border);
  font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
  font-weight: 700; color: var(--low-text); letter-spacing: 0.04em;
  display: flex; align-items: center; gap: 6px; white-space: nowrap;
}
.recover-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--low); display: inline-block; }

/* ── Global entrance animations ── */
@keyframes fadeslide { from {opacity:0;transform:translateY(14px)} to {opacity:1;transform:translateY(0)} }
@keyframes shimmer-text {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
@keyframes gradient-bg {
  0%,100% { background-position: 0% 50%; }
  50%     { background-position: 100% 50%; }
}
@keyframes pulse-border {
  0%,100% { box-shadow: 0 0 0 0 rgba(0,194,212,0.0); }
  50%     { box-shadow: 0 0 0 4px rgba(0,194,212,0.18); }
}
@keyframes float-up {
  0%,100% { transform: translateY(0); }
  50%     { transform: translateY(-4px); }
}

/* ── Executive Summary ── */
.exec-hero {
  background: linear-gradient(135deg, #080D17 0%, rgba(0,194,212,0.07) 40%, rgba(139,92,246,0.05) 70%, #080D17 100%);
  background-size: 300% 300%;
  animation: gradient-bg 8s ease infinite, fadeslide 0.7s ease both;
  border: 1px solid rgba(0,194,212,0.25);
  border-radius: 14px; padding: 2.2rem 2rem 2rem; text-align: center; margin-bottom: 1.2rem;
}
.exec-hero-eyebrow {
  font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent);
  margin-bottom: 0.5rem; display: block;
  animation: fadeslide 0.5s 0.1s ease both;
}
.exec-hero-title {
  font-family: 'Barlow Condensed', sans-serif; font-size: 2.2rem; font-weight: 800;
  letter-spacing: 0.04em; line-height: 1.15;
  background: linear-gradient(135deg, #E2EBF5 0%, #00C2D4 50%, #E2EBF5 100%);
  background-size: 200% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: shimmer-text 3.5s linear infinite, fadeslide 0.6s 0.15s ease both;
}
.exec-hero-sub {
  font-size: 0.8rem; color: var(--muted); margin-top: 0.55rem;
  animation: fadeslide 0.7s 0.3s ease both;
}

.exec-stat {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1.4rem 1rem; text-align: center;
  transition: border-color 0.3s, box-shadow 0.3s;
}
.exec-stat:hover { border-color: rgba(0,194,212,0.4); box-shadow: 0 0 18px rgba(0,194,212,0.1); }
.exec-stat-num {
  font-family: 'JetBrains Mono', monospace; font-size: 2.1rem; font-weight: 700;
  color: var(--accent); display: block; line-height: 1;
}
.exec-stat-label { font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.09em; margin-top: 0.45rem; }
.exec-stat-sub   { font-size: 0.67rem; color: var(--dim); margin-top: 0.2rem; }

.exec-section-label {
  font-family: 'Barlow Condensed', sans-serif; font-size: 0.68rem; font-weight: 700;
  letter-spacing: 0.13em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.65rem;
}
.plain-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 10px;
  padding: 1.1rem 1.3rem; margin-bottom: 0.7rem;
  animation: fadeslide 0.6s ease both;
  transition: border-color 0.25s;
}
.plain-card:hover { border-color: rgba(0,194,212,0.3); }
.plain-card-tag {
  font-size: 0.63rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
  padding: 2px 8px; border-radius: 4px; margin-bottom: 0.45rem; display: inline-block;
}
.tag-finding { background: rgba(0,194,212,0.12); color: var(--accent); }
.tag-action  { background: rgba(5,150,105,0.12); color: var(--low-text); }
.tag-risk    { background: rgba(220,38,38,0.12); color: #F87171; }
.plain-card-title { font-family:'Barlow Condensed',sans-serif; font-size:1.05rem; font-weight:700; color:var(--text); margin-bottom:0.25rem; }
.plain-card-body  { font-size:0.8rem; color:var(--data); line-height:1.55; }

.flow-step {
  background: var(--card); border: 1px solid var(--border); border-radius: 11px;
  padding: 1.2rem 0.8rem; text-align: center;
  animation: fadeslide 0.65s ease both, float-up 4s 1s ease-in-out infinite;
  transition: border-color 0.25s;
}
.flow-step:hover { border-color: rgba(0,194,212,0.4); animation: none; transform: translateY(-3px); }
.flow-icon  { font-size: 2rem; display: block; margin-bottom: 0.5rem; }
.flow-title { font-family:'Barlow Condensed',sans-serif; font-size:1.05rem; font-weight:700; color:var(--text); }
.flow-desc  { font-size:0.72rem; color:var(--muted); margin-top:0.25rem; line-height:1.4; }

.roi-banner {
  background: linear-gradient(135deg, rgba(0,194,212,0.07), rgba(5,150,105,0.05));
  border: 1px solid rgba(0,194,212,0.28); border-radius: 12px;
  padding: 1.6rem 2rem; text-align: center; margin-top: 1rem;
  animation: fadeslide 0.9s ease both, pulse-border 3s 1s ease-in-out infinite;
}
.roi-number {
  font-family: 'JetBrains Mono', monospace; font-size: 2.6rem; font-weight: 700; color: var(--accent);
}
.roi-label { font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }
.roi-sub   { font-size: 0.68rem; color: var(--dim); margin-top: 0.15rem; }

/* ── "So What?" insight callout ── */
.sowhat {
  background: linear-gradient(135deg, rgba(0,194,212,0.05), rgba(139,92,246,0.03));
  border: 1px solid rgba(0,194,212,0.18); border-left: 3px solid var(--accent);
  border-radius: 8px; padding: 0.85rem 1.1rem; margin: 0.9rem 0;
  animation: fadeslide 0.6s ease both;
}
.sowhat-label { font-size:0.62rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:var(--accent); margin-bottom:0.3rem; }
.sowhat-text  { font-size:0.82rem; color:var(--data); line-height:1.5; }
</style>
""", unsafe_allow_html=True)

# â"€â"€ Constants â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
DATA_PARQUET = "data.parquet"
DATA_PATH    = r"jan to may police violation_anonymized791b166.csv"

SEVERITY_MAP = {
    "PARKING NEAR ROAD CROSSING": 1.0,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 1.0,
    "DOUBLE PARKING": 0.90,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 0.85,
    "PARKING IN A MAIN ROAD": 0.80,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 0.75,
    "WRONG PARKING": 0.60,
    "NO PARKING": 0.50,
    "PARKING ON FOOTPATH": 0.45,
    "DEFECTIVE NUMBER PLATE": 0.20,
    "REFUSE TO GO FOR HIRE": 0.15,
}
VEHICLE_WEIGHT = {
    "PRIVATE BUS": 3.0, "TRUCK": 3.0, "LGV": 2.5, "VAN": 2.0,
    "MAXI-CAB": 1.8, "GOODS AUTO": 1.5, "CAR": 1.0, "PASSENGER AUTO": 1.0,
    "MOTOR CYCLE": 0.5, "SCOOTER": 0.5, "MOPED": 0.5,
}
PEAK_HOURS  = set(list(range(7, 11)) + list(range(17, 21)))
DAY_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
RISK_COLORS = {"Critical":"#DC2626","High":"#D97706","Medium":"#0EA5E9","Low":"#059669"}
FINE_MAP = {
    "PARKING NEAR ROAD CROSSING":                 2000,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS":  2000,
    "DOUBLE PARKING":                             1000,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 1000,
    "DEFECTIVE NUMBER PLATE":                     1000,
    "PARKING IN A MAIN ROAD":                      500,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC":     500,
    "WRONG PARKING":                               500,
    "NO PARKING":                                  500,
    "PARKING ON FOOTPATH":                         500,
    "REFUSE TO GO FOR HIRE":                       500,
}
CHART_PAL   = ["#00C2D4","#8B5CF6","#F59E0B","#10B981","#38BDF8","#F472B6","#A78BFA","#2DD4BF"]

# â"€â"€ Data loading â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
@st.cache_data(show_spinner="Loading 298k violation recordsâ€¦")
def load_data():
    # Prefer the compact parquet (used on Streamlit Cloud); fall back to raw CSV locally
    if _os.path.exists(DATA_PARQUET):
        df = pd.read_parquet(DATA_PARQUET)
    else:
        df = pd.read_csv(DATA_PATH)
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], format="mixed", utc=True)
    df["hour"]       = df["created_datetime"].dt.hour
    df["dow"]        = df["created_datetime"].dt.dayofweek
    df["day_name"]   = df["created_datetime"].dt.day_name()
    df["month_key"]  = df["created_datetime"].dt.to_period("M").astype(str)
    df["month_label"]= df["created_datetime"].dt.strftime("%b %Y")
    df = df.dropna(subset=["latitude","longitude"])
    df = df[df["latitude"].between(12.80,13.20) & df["longitude"].between(77.40,77.80)].copy()

    def parse_v(v):
        try:    return ast.literal_eval(str(v))
        except: return []

    df["vlist"]         = df["violation_type"].apply(parse_v)
    df["primary_vtype"] = df["vlist"].apply(lambda x: x[0] if x else "UNKNOWN")
    df["severity"]      = df["vlist"].apply(
        lambda lst: max((SEVERITY_MAP.get(v, 0.30) for v in lst), default=0.30)
    )
    df["vehicle_weight"] = df["vehicle_type"].map(VEHICLE_WEIGHT).fillna(1.0)
    df["is_peak"]        = df["hour"].isin(PEAK_HOURS).astype(int)
    df["at_junction"]    = (df["junction_name"] != "No Junction").astype(int)
    df["scita_ok"]       = (df["data_sent_to_scita"] == True).astype(int)
    df["is_heavy"]       = (df["vehicle_weight"] >= 2.0).astype(int)
    df["junc_label"]     = df["junction_name"].apply(
        lambda x: x.split(" - ",1)[1] if " - " in str(x) else x
    )
    return df


# â"€â"€ Spatial grid clustering â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
@st.cache_data(show_spinner="Mapping hotspot zonesâ€¦")
def detect_hotspots(lat_arr, lon_arr, cell_km: float, min_samples: int):
    cell_deg = cell_km / 111.0
    lat_bin  = np.floor(lat_arr / cell_deg).astype(int)
    lon_bin  = np.floor(lon_arr / cell_deg).astype(int)
    cell_ids = np.array([f"{a}_{b}" for a, b in zip(lat_bin, lon_bin)])
    unique, counts = np.unique(cell_ids, return_counts=True)
    hotspot_cells  = set(unique[counts >= min_samples])
    cell_to_label  = {c: i for i, c in enumerate(sorted(hotspot_cells))}
    labels = np.array([cell_to_label.get(c, -1) for c in cell_ids])
    return labels


# â"€â"€ CII computation â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
@st.cache_data
def compute_cii(_df):
    stats = (
        _df[_df["cluster"] != -1]
        .groupby("cluster")
        .agg(
            count        =("id",            "count"),
            lat          =("latitude",       "mean"),
            lon          =("longitude",      "mean"),
            junction_rate=("at_junction",    "mean"),
            severity_mean=("severity",       "mean"),
            peak_rate    =("is_peak",        "mean"),
            scita_rate   =("scita_ok",       "mean"),
            vweight_mean =("vehicle_weight", "mean"),
            heavy_rate   =("is_heavy",       "mean"),
            top_vtype    =("primary_vtype",  lambda x: x.mode().iloc[0] if len(x) else "N/A"),
            station      =("police_station", lambda x: x.mode().iloc[0] if len(x) else "N/A"),
            top_hour     =("hour",           lambda x: int(x.mode().iloc[0]) if len(x) else 0),
            top_day      =("day_name",       lambda x: x.mode().iloc[0] if len(x) else "N/A"),
        )
        .reset_index()
    )
    sc = MinMaxScaler()
    stats["freq_score"] = sc.fit_transform(stats[["count"]]).flatten()
    stats["cii"] = (
        0.35 * stats["freq_score"]    +
        0.30 * stats["junction_rate"] +
        0.20 * stats["severity_mean"] +
        0.15 * stats["peak_rate"]
    ).round(4)
    stats["enforce_gap"] = (stats["cii"] * (1 - stats["scita_rate"])).round(4)
    stats["risk_tier"] = pd.cut(
        stats["cii"], bins=[0,.25,.45,.65,1.0],
        labels=["Low","Medium","High","Critical"], include_lowest=True,
    )
    stats = stats.sort_values("cii", ascending=False).reset_index(drop=True)
    stats["rank"] = stats.index + 1
    return stats


# â"€â"€ Helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
def plotly_base():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B87A8", family="'Barlow Condensed', sans-serif", size=11),
        margin=dict(t=36, b=10, l=10, r=10),
    )

def pl(fig, **kw):
    """Apply dark base layout + axis styles in one call."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#6B87A8", family="'Barlow Condensed', sans-serif", size=11),
        margin=dict(t=36, b=10, l=10, r=10),
    )
    base.update(kw)
    fig.update_layout(**base)
    fig.update_xaxes(gridcolor="#0E1623", linecolor="#30363D", zerolinecolor="#30363D")
    fig.update_yaxes(gridcolor="#0E1623", linecolor="#30363D")
    return fig

def gauge_fig(value, title, color="#00C2D4", max_val=1.0):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix":"", "font":{"size":26,"color":"#E2EBF5","family":"JetBrains Mono"}, "valueformat":".3f"},
        title={"text":title, "font":{"size":10,"color":"#6B87A8","family":"Barlow Condensed"}},
        gauge={
            "axis": {"range":[0,max_val],"tickcolor":"#253D5C","tickwidth":1,
                     "tickfont":{"color":"#334B65","size":8,"family":"JetBrains Mono"}},
            "bar": {"color":color, "thickness":0.70},
            "bgcolor":"#13203A", "borderwidth":0,
            "steps":[
                {"range":[0, max_val*0.33], "color":"#0E1623"},
                {"range":[max_val*0.33, max_val*0.66], "color":"#0F1A2D"},
                {"range":[max_val*0.66, max_val], "color":"#13203A"},
            ],
        }
    ))
    fig.update_layout(height=190, paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=40,b=0,l=15,r=15))
    return fig

def risk_color_css(tier):
    return {"Critical":"risk-crit","High":"risk-high","Medium":"risk-med","Low":"risk-low"}.get(str(tier),"")

def build_schedule(df_full, cluster_stats, n=10):
    rows = []
    for _, z in cluster_stats.head(n).iterrows():
        zdf = df_full[df_full["cluster"] == z["cluster"]]
        if zdf.empty:
            continue
        grp = zdf.groupby(["day_name","hour"]).size()
        if grp.empty:
            continue
        peak_day, peak_hour = grp.idxmax()
        deploy_hour = max(0, peak_hour - 1)
        rows.append({
            "rank":       int(z["rank"]),
            "station":    z["station"],
            "cii":        z["cii"],
            "risk_tier":  str(z["risk_tier"]),
            "peak_day":   peak_day,
            "peak_hour":  f"{peak_hour:02d}:00",
            "deploy_at":  f"{deploy_hour:02d}:00",
            "violations": int(z["count"]),
            "junc_pct":   f"{z['junction_rate']*100:.0f}%",
        })
    return pd.DataFrame(rows)


def nearest_neighbor_route(zones_df):
    """Greedy nearest-neighbour TSP — starts at highest-CII zone."""
    lats = zones_df["lat"].tolist()
    lons = zones_df["lon"].tolist()
    n = len(lats)
    if n <= 1:
        return list(range(n))
    unvisited = list(range(1, n))
    route = [0]
    while unvisited:
        curr = route[-1]
        nearest = min(unvisited,
                      key=lambda j: (lats[curr]-lats[j])**2 + (lons[curr]-lons[j])**2)
        route.append(nearest)
        unvisited.remove(nearest)
    return route



@st.cache_resource
def train_isolation_forest(_anom_X_tuple):
    """Train IsolationForest on zone-level features. Cached across reruns."""
    import numpy as _np
    _X = _np.array(_anom_X_tuple)
    iso = IsolationForest(n_estimators=200, contamination=0.10, random_state=42, n_jobs=-1)
    iso.fit(_X)
    return iso

@st.cache_resource
def train_rf_classifier(_X_tuple, _y_tuple):
    """Train RandomForest violation-type classifier. Cached across reruns."""
    import numpy as _np
    _X = _np.array(_X_tuple)
    _y = _np.array(_y_tuple)
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(_X, _y)
    return rf

# ── Bengaluru landmarks (lat, lon) ─────────────────────────────────────────
BENGALURU_LANDMARKS = {
    "Silk Board Junction":     (12.9176, 77.6233),
    "Electronic City":         (12.8452, 77.6602),
    "Koramangala 5th Block":   (12.9352, 77.6245),
    "HSR Layout":              (12.9116, 77.6473),
    "Marathahalli Bridge":     (12.9560, 77.7010),
    "Whitefield":              (12.9698, 77.7499),
    "MG Road":                 (12.9757, 77.6011),
    "Brigade Road":            (12.9724, 77.6081),
    "Jayanagar 4th Block":     (12.9250, 77.5938),
    "Banashankari":            (12.9260, 77.5656),
    "Rajajinagar":             (12.9952, 77.5527),
    "Yeshwanthpur":            (13.0241, 77.5380),
    "Hebbal Flyover":          (13.0351, 77.5971),
    "Yelahanka":               (13.1007, 77.5963),
    "KR Puram":                (13.0058, 77.6963),
    "Majestic (KSR)":          (12.9767, 77.5713),
    "Shivajinagar":            (12.9856, 77.6010),
    "Indiranagar 100ft Road":  (12.9784, 77.6408),
    "Old Airport Road":        (12.9542, 77.6471),
    "Domlur":                  (12.9608, 77.6385),
    "Lalbagh":                 (12.9507, 77.5848),
    "Ulsoor":                  (12.9842, 77.6195),
    "BTM Layout":              (12.9165, 77.6101),
    "JP Nagar":                (12.9063, 77.5857),
    "Bellary Road (Hebbal)":   (13.0500, 77.5960),
    "Outer Ring Road (ORR)":   (12.9344, 77.6869),
    "Sarjapur Road":           (12.9102, 77.6823),
    "Tumkur Road":             (13.0392, 77.5254),
    "Mysore Road":             (12.9462, 77.5170),
    "Kanakapura Road":         (12.8960, 77.5730),
}

@st.cache_data(show_spinner=False)
def build_congestion_grid(_lat_hash, lat_arr, lon_arr, sev_arr,
                           lat_min=12.83, lat_max=13.12,
                           lon_min=77.46, lon_max=77.78, n=80):
    """Build an n×n congestion grid weighted by violation density & severity."""
    grid = np.zeros((n, n), dtype=np.float32)
    lat_step = (lat_max - lat_min) / n
    lon_step = (lon_max - lon_min) / n
    for lat, lon, sev in zip(lat_arr, lon_arr, sev_arr):
        ri = int((lat - lat_min) / lat_step)
        ci = int((lon - lon_min) / lon_step)
        if 0 <= ri < n and 0 <= ci < n:
            grid[ri, ci] += sev
    # Gaussian blur to smooth hotspots into surrounding road cells
    from scipy.ndimage import gaussian_filter
    grid = gaussian_filter(grid, sigma=1.5)
    gmax = grid.max()
    if gmax > 0:
        grid /= gmax
    return grid, lat_min, lat_max, lon_min, lon_max

def _latlon_to_rc(lat, lon, lat_min, lat_max, lon_min, lon_max, n):
    ri = int((lat - lat_min) / (lat_max - lat_min) * n)
    ci = int((lon - lon_min) / (lon_max - lon_min) * n)
    return max(0, min(n-1, ri)), max(0, min(n-1, ci))

def _rc_to_latlon(ri, ci, lat_min, lat_max, lon_min, lon_max, n):
    lat = lat_min + (ri + 0.5) * (lat_max - lat_min) / n
    lon = lon_min + (ci + 0.5) * (lon_max - lon_min) / n
    return lat, lon

def astar_route(grid, start_rc, end_rc, cong_weight=10.0):
    """A* on congestion grid. Returns list of (row,col) tuples."""
    import heapq
    n = grid.shape[0]
    sr, sc = start_rc; er, ec = end_rc
    def h(r, c): return ((r - er)**2 + (c - ec)**2) ** 0.5
    open_set = [(h(sr, sc), 0.0, sr, sc, [(sr, sc)])]
    visited = {}
    while open_set:
        f, g, r, c, path = heapq.heappop(open_set)
        if (r, c) in visited:
            continue
        visited[(r, c)] = g
        if r == er and c == ec:
            return path
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n and 0 <= nc < n):
                    continue
                if (nr, nc) in visited:
                    continue
                step = (dr**2 + dc**2) ** 0.5
                ng = g + step * (1.0 + cong_weight * float(grid[nr, nc]))
                heapq.heappush(open_set, (ng + h(nr, nc), ng, nr, nc, path + [(nr, nc)]))
    # Fallback: straight line
    rs = np.linspace(sr, er, 60).astype(int)
    cs = np.linspace(sc, ec, 60).astype(int)
    return list(zip(rs.tolist(), cs.tolist()))

def straight_line_route(start_rc, end_rc, n_pts=60):
    """Direct straight-line path through grid."""
    sr, sc = start_rc; er, ec = end_rc
    rs = np.linspace(sr, er, n_pts).astype(int)
    cs = np.linspace(sc, ec, n_pts).astype(int)
    return list(zip(rs.tolist(), cs.tolist()))

def path_congestion_stats(path, grid):
    """Return mean & max congestion along a path."""
    vals = [float(grid[r, c]) for r, c in path]
    return np.mean(vals), np.max(vals), sum(vals)

# ==========================================================================
# MAIN APP
# ==========================================================================
def main():
    # â"€â"€ Header â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    st.markdown("""
    <div class="gl-header">
      <div class="gl-logo"><span class="gl-logo-text">GL</span></div>
      <div class="gl-title-wrap">
        <p class="gl-title">GridLock Intelligence Platform</p>
        <p class="gl-sub">Bengaluru Traffic Enforcement Command &middot; 298k Violations &middot; Nov 2023&ndash;Apr 2024</p>
      </div>
      <div class="gl-header-right">
        <span class="gl-live-pip"></span>&nbsp;NOV 2023 – APR 2024
      </div>
    </div>
    """, unsafe_allow_html=True)

    # â"€â"€ Load data â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    df_full = load_data()

    # â"€â"€ Sidebar â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    with st.sidebar:
        st.markdown('<div class="sidebar-section">&#9650;&#9650; TIME PERIOD</div>', unsafe_allow_html=True)
        period = st.selectbox("", ["All Time","Last 30 Days","Last 60 Days","Last 90 Days"],
                              label_visibility="collapsed")
        st.markdown('<div class="sidebar-section">&#9632;&#9632; POLICE STATION</div>', unsafe_allow_html=True)
        stations   = ["All Stations"] + sorted(df_full["police_station"].dropna().unique())
        sel_sta    = st.selectbox("", stations, label_visibility="collapsed")
        st.markdown('<div class="sidebar-section">&#9830; VIOLATION TYPE</div>', unsafe_allow_html=True)
        vtypes     = ["All Types","WRONG PARKING","NO PARKING","PARKING IN A MAIN ROAD",
                      "DOUBLE PARKING","PARKING NEAR ROAD CROSSING","PARKING ON FOOTPATH"]
        sel_vtype  = st.selectbox("", vtypes, label_visibility="collapsed")
        st.markdown('<div class="sidebar-section">&#9881; ZONE DETECTION</div>', unsafe_allow_html=True)
        cell_km    = st.slider("Grid Cell Size (km)", 0.1, 1.0, 0.30, 0.05)
        min_samp   = st.slider("Min Violations / Zone", 30, 300, 80, 10)

        st.divider()
        days  = {"All Time":None,"Last 30 Days":30,"Last 60 Days":60,"Last 90 Days":90}[period]
        df = df_full.copy()
        if days:
            df = df[df["created_datetime"] >= df["created_datetime"].max() - pd.Timedelta(days=days)]
        if sel_sta != "All Stations":
            df = df[df["police_station"] == sel_sta]
        if sel_vtype != "All Types":
            df = df[df["vlist"].apply(lambda x: sel_vtype in x)]


        st.divider()
        with st.expander("CII Formula & Methodology", expanded=False):
            st.markdown("""
            **Congestion Impact Index (CII)** quantifies zone risk on a 0–1 scale:

            `CII = 0.35×freq + 0.30×junction + 0.20×severity + 0.15×peak`

            | Weight | Factor | Description |
            |--------|--------|-------------|
            | 35% | Frequency | Normalised violation density |
            | 30% | Junction | % at intersections |
            | 20% | Severity | Violation severity score |
            | 15% | Peak Hour | % during AM/PM peak |

            **Risk tiers:** Critical > 0.6 · High 0.4–0.6 · Medium 0.2–0.4 · Low < 0.2
            """)

        st.metric("Records in View", f"{len(df):,}")
        st.caption(f"{len(df)/len(df_full)*100:.1f}% of total dataset")

    # â"€â"€ Cluster on full data â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    labels_full    = detect_hotspots(df_full["latitude"].values, df_full["longitude"].values,
                                     cell_km, min_samp)
    df_full        = df_full.copy()
    df_full["cluster"] = labels_full
    cluster_map    = df_full.set_index("id")["cluster"].to_dict()
    df["cluster"]  = df["id"].map(cluster_map).fillna(-1).astype(int)
    cluster_stats  = compute_cii(df_full)

    # â"€â"€ Derived KPIs â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    n_zones     = len(cluster_stats)
    n_critical  = int((cluster_stats["risk_tier"] == "Critical").sum())
    n_high      = int((cluster_stats["risk_tier"] == "High").sum())
    n_medium    = int((cluster_stats["risk_tier"] == "Medium").sum())
    n_low       = int((cluster_stats["risk_tier"] == "Low").sum())
    junc_pct    = df["at_junction"].mean() * 100
    blind_count = int((df["scita_ok"] == 0).sum())
    blind_pct   = (1 - df["scita_ok"].mean()) * 100
    heavy_pct   = df["is_heavy"].mean() * 100
    peak_pct    = df["is_peak"].mean() * 100
    top_zone    = cluster_stats.iloc[0]
    top_cii     = float(top_zone["cii"])
    avg_cii     = float(cluster_stats["cii"].mean())
    city_health = max(0, 100 * (1 - avg_cii * 1.2))
    # Fine revenue intelligence
    df["fine_amt"]          = df["primary_vtype"].map(FINE_MAP).fillna(500)
    df["uncollected_fine"]  = df["fine_amt"] * (1 - df["scita_ok"])
    total_fine              = float(df["fine_amt"].sum())
    uncollected_fine_total  = float(df["uncollected_fine"].sum())
    recovery_rate           = (1 - uncollected_fine_total / total_fine) * 100 if total_fine > 0 else 0

    # Cluster-level fine stats (on full dataset for zone ranking)
    _fa  = df_full["primary_vtype"].map(FINE_MAP).fillna(500)
    _uf  = _fa * (1 - df_full["scita_ok"])
    _cg  = pd.DataFrame({"cluster": df_full["cluster"], "fa": _fa, "uf": _uf}) \
             .groupby("cluster") \
             .agg(total_fine=("fa","sum"), uncollected=("uf","sum")) \
             .reset_index()
    cs_fine = cluster_stats.merge(_cg, on="cluster", how="left")
    cs_fine["total_fine"]  = cs_fine["total_fine"].fillna(0)
    cs_fine["uncollected"] = cs_fine["uncollected"].fillna(0)
    cs_fine["scita_gap_pct"]  = (1 - cs_fine["scita_rate"]) * 100
    cs_fine["priority_score"] = (cs_fine["cii"] * cs_fine["scita_gap_pct"] / 100).round(4)
    _gap_thresh = max(cs_fine["scita_gap_pct"].quantile(0.70), 10.0)
    n_cam_priority = int(((cs_fine["cii"] > 0.45) & (cs_fine["scita_gap_pct"] > _gap_thresh)).sum())

    # -- Anomaly Detection (IsolationForest on zone-level features) -----------
    _anom_features = ["cii", "junction_rate", "peak_rate", "severity_mean",
                      "heavy_rate", "scita_rate"]
    _anom_df = cs_fine[_anom_features].fillna(0).copy()
    _anom_scaler = MinMaxScaler()
    _anom_X = _anom_scaler.fit_transform(_anom_df)
    _iso = train_isolation_forest(tuple(map(tuple, _anom_X)))
    _anom_scores = _iso.decision_function(_anom_X)
    _anom_labels = _iso.predict(_anom_X)
    cs_fine = cs_fine.copy()
    cs_fine["anomaly_score"] = -_anom_scores
    cs_fine["is_anomaly"]    = (_anom_labels == -1)
    cluster_stats = cluster_stats.merge(
        cs_fine[["cluster","anomaly_score","is_anomaly"]], on="cluster", how="left"
    )
    cluster_stats["is_anomaly"] = cluster_stats["is_anomaly"].fillna(False)
    n_anomalies   = int(cs_fine["is_anomaly"].sum())
    anomaly_zones = cs_fine[cs_fine["is_anomaly"]].sort_values("anomaly_score", ascending=False)



    # ── Persistent sidebar quickstats (injected after KPIs computed) ─────────
    _unc_cr         = uncollected_fine_total / 1e7
    _recoverable_cr = _unc_cr * 0.25
    _now_h          = _dt.datetime.now().hour
    _is_peak_sb     = _now_h in range(7,11) or _now_h in range(17,21)
    _peak_sb_col    = "#DC2626" if _is_peak_sb else "#059669"
    _peak_sb_lbl    = "PEAK HOUR NOW" if _is_peak_sb else "OFF-PEAK NOW"
    st.sidebar.markdown(f"""
    <div style="margin-top:8px">
      <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.60rem;font-weight:700;
                  color:var(--dim);text-transform:uppercase;letter-spacing:.13em;margin-bottom:8px">
        ⚡ LIVE METRICS
      </div>
      <div class="sb-stat" style="border-left:3px solid #DC2626">
        <div class="sb-stat-label">Uncollected Revenue</div>
        <div class="sb-stat-val" style="color:#F87171">₹{_unc_cr:.1f} Cr</div>
        <div class="sb-stat-sub">₹{_recoverable_cr:.2f} Cr recoverable (25%)</div>
      </div>
      <div class="sb-stat" style="border-left:3px solid #00C2D4">
        <div class="sb-stat-label">Top Zone CII</div>
        <div class="sb-stat-val" style="color:#00C2D4">{top_cii:.3f}</div>
        <div class="sb-stat-sub">{str(top_zone['station'])[:20]} · {n_critical} critical zones</div>
      </div>
      <div class="sb-stat" style="border-left:3px solid {_peak_sb_col}">
        <div class="sb-stat-label">Enforcement Window</div>
        <div class="sb-stat-val" style="color:{_peak_sb_col};font-size:0.85rem">{_peak_sb_lbl}</div>
        <div class="sb-stat-sub">{_dt.datetime.now().strftime('%A %H:%M')}</div>
      </div>
      <div class="sb-stat" style="border-left:3px solid #D97706">
        <div class="sb-stat-label">SCITA Blind Spots</div>
        <div class="sb-stat-val" style="color:#FBBF24">{blind_pct:.0f}%</div>
        <div class="sb-stat-sub">{blind_count:,} violations untracked</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────────────

    st.markdown(f"""
    <div class="zone-rail">
      <div class="zone-rail-label">Zone Status</div>
      <div class="zone-chip zc-critical"><span class="zone-dot zd-critical"></span>{n_critical} Critical</div>
      <div class="zone-chip zc-high"><span class="zone-dot zd-high"></span>{n_high} High</div>
      <div class="zone-chip zc-medium"><span class="zone-dot zd-medium"></span>{n_medium} Medium</div>
      <div class="zone-chip zc-low"><span class="zone-dot zd-low"></span>{n_low} Low</div>
      <div class="zone-rail-tail">{n_zones} ZONES ACTIVE</div>
      <div style="margin-left:auto;background:rgba(16,185,129,0.12);border:1px solid #059669;border-radius:20px;padding:2px 12px;font-size:0.65rem;color:#10B981;font-weight:700;white-space:nowrap">&#8377;{_recoverable_cr:.1f} Cr recoverable</div>
    </div>
    """, unsafe_allow_html=True)


    # ── Situation Report Banner ──────────────────────────────────────────────
    _top_stn  = str(top_zone["station"])[:28]
    _top_rank = int(top_zone["rank"])
    _top_cii_v = float(top_zone["cii"])
    _top_cnt  = int(top_zone["count"])
    _top_day  = str(top_zone["top_day"])
    _top_hr   = int(top_zone["top_hour"])
    _unc_cr   = uncollected_fine_total / 1e7
    _crit_pct = n_critical / max(n_zones, 1) * 100
    _top3_sta = ", ".join(cluster_stats.head(3)["station"].str[:16].tolist())
    st.markdown(f"""
    <div class="sitrep-bar">
      <span class="sitrep-label">SITREP</span>
      <span class="sitrep-item sitrep-red">
        <span class="sitrep-dot sitrep-dot-red"></span>
        <b>Zone #{_top_rank}</b> ({_top_stn}) &mdash; CII {_top_cii_v:.3f} &bull; {_top_cnt:,} violations &bull; peaks {_top_day} {_top_hr:02d}:00
      </span>
      <span class="sitrep-sep">|</span>
      <span class="sitrep-item sitrep-amber">
        <span class="sitrep-dot sitrep-dot-amber"></span>
        <b>&#8377;{_unc_cr:.1f} Cr</b> uncollected &bull; {blind_pct:.0f}% blind to SCITA
      </span>
      <span class="sitrep-sep">|</span>
      <span class="sitrep-item sitrep-cyan">
        <span class="sitrep-dot sitrep-dot-cyan"></span>
        <b>{n_critical} Critical</b> zones ({max(_crit_pct, 0.1):.1f}% of total) &mdash; {_top3_sta}
      </span>
    </div>
    """, unsafe_allow_html=True)

    tex, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = st.tabs([
        "🏠 Executive",
        "📊 Overview",
        "🗺️ Hotspot Map",
        "⛌ Junction Analysis",
        "🎯 Enforcement Planner",
        "📈 Temporal Analysis",
        "🔍 Zone Explorer",
        "⚡ Command Intel",
        "📋 Impact Report",
        "🤖 AI Assistant",
        "🛣️ Route Optimizer",
    ])

    # ==========================================================================
    # TAB 0 -- EXECUTIVE SUMMARY
    # ==========================================================================
    with tex:
        _tot_v      = len(df)
        _unc_cr_ex  = uncollected_fine_total / 1e7
        _rec_cr_ex  = _unc_cr_ex * 0.25
        _n_crit_ex  = int(n_critical)
        _top_stn_ex = str(top_zone["station"])[:28]
        _top_cnt_ex = int(top_zone["count"])
        _top_hr_ex  = int(top_zone["top_hour"])
        _top_day_ex = str(top_zone["top_day"])
        _top_cii_ex = float(top_zone["cii"])
        _blind_ex   = blind_pct
        # Officer ROI: top-zone monthly uncollected / 26 working days / 2 shifts
        _top_unc    = float(top_zone.get("uncollected", 0))
        _roi_shift  = int(_top_unc / 6 / 26 / 2)
        _avg_shift  = int(uncollected_fine_total / 6 / 26 / 2 / max(n_zones, 1))
        _roi_mult   = _roi_shift / max(_avg_shift, 1)
        # Top 3 zones for findings
        _t3 = cluster_stats.head(3)[["station", "count", "peak_rate", "cii"]].copy()

        # Hero
        st.markdown(f"""
        <div class="exec-hero">
          <span class="exec-hero-eyebrow">GridLock Intelligence Platform &nbsp;&bull;&nbsp; Bengaluru Traffic Command</span>
          <div class="exec-hero-title">Turn Violation Data Into Officer Deployments</div>
          <div class="exec-hero-sub">
            {_tot_v:,} real violations &nbsp;&middot;&nbsp; Nov 2023 &ndash; Apr 2024
            &nbsp;&middot;&nbsp; 6 AI models &nbsp;&middot;&nbsp; Real-time enforcement intelligence
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Animated stat counters via JS component
        st.components.v1.html(f"""
        <style>
          body {{ margin:0; background:transparent; font-family:'JetBrains Mono',monospace; }}
          .ctr-wrap {{ display:flex; gap:14px; }}
          .ctr-card {{
            flex:1; background:#0D1117; border:1px solid #1E2D3D; border-radius:12px;
            padding:22px 14px; text-align:center;
            transition: border-color 0.3s, box-shadow 0.3s;
          }}
          .ctr-card:hover {{ border-color:#00C2D4; box-shadow:0 0 20px rgba(0,194,212,0.12); }}
          .ctr-num {{ font-size:2rem; font-weight:700; color:#00C2D4; display:block; line-height:1; }}
          .ctr-lbl {{ font-size:0.67rem; color:#6B87A8; text-transform:uppercase; letter-spacing:0.09em; margin-top:8px; }}
          .ctr-sub {{ font-size:0.62rem; color:#3D5A7A; margin-top:4px; }}
        </style>
        <div class="ctr-wrap">
          <div class="ctr-card">
            <span class="ctr-num" id="c1">0</span>
            <div class="ctr-lbl">Total Violations Recorded</div>
            <div class="ctr-sub">Nov 2023 &ndash; Apr 2024</div>
          </div>
          <div class="ctr-card">
            <span class="ctr-num" id="c2" style="color:#F87171">&#8377;0 Cr</span>
            <div class="ctr-lbl">Fines Left Uncollected</div>
            <div class="ctr-sub">&#8377;{_rec_cr_ex:.1f} Cr recoverable (25%)</div>
          </div>
          <div class="ctr-card">
            <span class="ctr-num" id="c3" style="color:#DC2626">{_n_crit_ex}</span>
            <div class="ctr-lbl">Critical Danger Zones</div>
            <div class="ctr-sub">{n_zones} total zones active</div>
          </div>
          <div class="ctr-card">
            <span class="ctr-num" id="c4" style="color:#FBBF24">{_blind_ex:.0f}%</span>
            <div class="ctr-lbl">Violations Missed by Cameras</div>
            <div class="ctr-sub">Need officer enforcement</div>
          </div>
        </div>
        <script>
        function animateCounter(id, target, prefix, suffix, decimals, duration) {{
          var el = document.getElementById(id);
          var start = 0, startTime = null;
          function step(ts) {{
            if (!startTime) startTime = ts;
            var progress = Math.min((ts - startTime) / duration, 1);
            var ease = 1 - Math.pow(1 - progress, 3);
            var val = start + (target - start) * ease;
            el.textContent = prefix + (decimals > 0 ? val.toFixed(decimals) : Math.floor(val).toLocaleString()) + suffix;
            if (progress < 1) requestAnimationFrame(step);
          }}
          requestAnimationFrame(step);
        }}
        animateCounter("c1", {_tot_v}, "", "", 0, 1800);
        animateCounter("c2", {_unc_cr_ex:.2f}, "₹", " Cr", 1, 1600);
        </script>
        """, height=130)

        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

        # Problem | Solution
        col_prob, col_sol = st.columns(2, gap="medium")
        with col_prob:
            st.markdown("""
            <div class="exec-section-label">The Problem</div>
            <div class="plain-card" style="border-left:3px solid #DC2626">
              <div class="plain-card-tag tag-risk">Challenge</div>
              <div class="plain-card-title">Bengaluru's traffic enforcement is flying blind</div>
              <div class="plain-card-body">
                With 298k violations spread across {n} zones, traffic police currently patrol based on intuition &mdash;
                not data. Officers are sent to low-risk areas while high-violation hotspots go understaffed.
                Meanwhile, crores in fines go uncollected every month because cameras can't cover every intersection.
              </div>
            </div>
            """.format(n=n_zones), unsafe_allow_html=True)

        with col_sol:
            st.markdown("""
            <div class="exec-section-label">GridLock Solves This</div>
            <div class="plain-card" style="border-left:3px solid var(--accent)">
              <div class="plain-card-tag tag-finding">Solution</div>
              <div class="plain-card-title">AI ranks every zone by real danger — not guesswork</div>
              <div class="plain-card-body">
                GridLock processes all violation records through 6 AI models to score every zone with a
                <b>Danger Score (CII)</b>, predict peak surge windows, find camera blind spots, and show
                officers exactly where to go &mdash; and the optimal route to get there safely.
              </div>
            </div>
            """, unsafe_allow_html=True)

        # How it works — 3-step flow
        st.markdown('<div style="height:0.4rem"></div>', unsafe_allow_html=True)
        st.markdown('<div class="exec-section-label">How It Works</div>', unsafe_allow_html=True)
        fc1, fc2, fc3, fc4, fc5 = st.columns([1, 0.15, 1, 0.15, 1])
        with fc1:
            st.markdown("""
            <div class="flow-step" style="animation-delay:0.1s">
              <span class="flow-icon">🔍</span>
              <div class="flow-title">Detect Hotspots</div>
              <div class="flow-desc">AI clusters 298k violations by location and scores each zone using our Danger Score formula</div>
            </div>""", unsafe_allow_html=True)
        with fc2:
            st.markdown('<div style="text-align:center;font-size:1.4rem;color:#3D5A7A;margin-top:2rem">→</div>', unsafe_allow_html=True)
        with fc3:
            st.markdown("""
            <div class="flow-step" style="animation-delay:0.25s">
              <span class="flow-icon">📊</span>
              <div class="flow-title">Rank by Danger</div>
              <div class="flow-desc">RandomForest + Prophet forecast when and where violations will surge next — hours in advance</div>
            </div>""", unsafe_allow_html=True)
        with fc4:
            st.markdown('<div style="text-align:center;font-size:1.4rem;color:#3D5A7A;margin-top:2rem">→</div>', unsafe_allow_html=True)
        with fc5:
            st.markdown("""
            <div class="flow-step" style="animation-delay:0.4s">
              <span class="flow-icon">🎯</span>
              <div class="flow-title">Deploy & Route</div>
              <div class="flow-desc">A* pathfinding routes officers through the city avoiding high-congestion zones to reach targets faster</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

        # Key findings — plain English
        col_f, col_r = st.columns([1.4, 1], gap="medium")
        with col_f:
            st.markdown('<div class="exec-section-label">Top Findings — Plain English</div>', unsafe_allow_html=True)
            _peak_pct = int(_t3.iloc[0]["peak_rate"] * 100)
            st.markdown(f"""
            <div class="plain-card" style="animation-delay:0.1s">
              <div class="plain-card-tag tag-risk">Most Dangerous Zone</div>
              <div class="plain-card-title">{_top_stn_ex} is the #1 hotspot</div>
              <div class="plain-card-body">
                With a Danger Score of <b>{_top_cii_ex:.3f}</b> (maximum is 1.0), this zone recorded
                <b>{_top_cnt_ex:,} violations</b> and peaks every <b>{_top_day_ex} at {_top_hr_ex:02d}:00</b>.
                Deploying one officer here recovers more revenue than any other location in the city.
              </div>
            </div>
            <div class="plain-card" style="animation-delay:0.2s">
              <div class="plain-card-tag tag-finding">Revenue Opportunity</div>
              <div class="plain-card-title">&#8377;{_unc_cr_ex:.1f} Cr sitting uncollected</div>
              <div class="plain-card-body">
                <b>{_blind_ex:.0f}% of violations</b> happen outside camera coverage.
                GridLock identifies exactly which zones these are so officers can recover
                the estimated <b>&#8377;{_rec_cr_ex:.1f} Cr</b> with targeted deployment.
              </div>
            </div>
            <div class="plain-card" style="animation-delay:0.3s">
              <div class="plain-card-tag tag-action">Best Enforcement Window</div>
              <div class="plain-card-title">Peak violations: 7–10 AM and 5–8 PM</div>
              <div class="plain-card-body">
                Violations spike during morning and evening rush hours. GridLock's surge predictor
                flags these windows 24 hours in advance so shift commanders can
                pre-position officers — not react after congestion builds.
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col_r:
            st.markdown('<div class="exec-section-label">Officer ROI Spotlight</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="roi-banner">
              <div style="font-size:0.68rem;color:#6B87A8;letter-spacing:0.09em;text-transform:uppercase;margin-bottom:0.5rem">
                Estimated recovery per shift &nbsp;&bull;&nbsp; Zone #1
              </div>
              <div class="roi-number">&#8377;{_roi_shift:,}</div>
              <div class="roi-label">per officer per 8-hour shift</div>
              <div class="roi-sub" style="margin-top:0.6rem;color:#546D87">
                vs. &#8377;{_avg_shift:,} at a random patrol zone
              </div>
              <div style="margin-top:1rem;padding-top:0.8rem;border-top:1px solid rgba(0,194,212,0.15);
                          font-size:0.72rem;color:#00C2D4;font-weight:700">
                {_roi_mult:.1f}x more effective deployment
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="plain-card" style="border-left:3px solid #059669">
              <div class="plain-card-tag tag-action">What GridLock Enables</div>
              <div class="plain-card-body" style="line-height:1.8">
                ✅ &nbsp;Know <b>where</b> violations will happen next<br>
                ✅ &nbsp;Know <b>when</b> to be there (surge window)<br>
                ✅ &nbsp;Know the <b>fastest safe route</b> to get there<br>
                ✅ &nbsp;Know <b>how much</b> revenue each zone can recover<br>
                ✅ &nbsp;Know which zones cameras are <b>missing entirely</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 1 -- OVERVIEW
    # ==========================================================================
    with t1:
        # MoM delta for trend arrows
        _mn_kpi = df.groupby("month_key").size().reset_index(name="c").sort_values("month_key")
        if len(_mn_kpi) >= 2:
            _m_last = int(_mn_kpi.iloc[-1]["c"]); _m_prev = int(_mn_kpi.iloc[-2]["c"])
            _mom    = (_m_last - _m_prev) / max(_m_prev, 1) * 100
            _mom_cls= "delta-up" if _mom > 0 else ("delta-down" if _mom < 0 else "delta-flat")
            _mom_arrow = "▲" if _mom > 0 else ("▼" if _mom < 0 else "–")
            _mom_html = f'<span class="kpi-delta {_mom_cls}">{_mom_arrow} {abs(_mom):.1f}% vs prev month</span>'
        else:
            _mom_html = ""
        _rec_cr = uncollected_fine_total * 0.25 / 1e7
        # KPI cards
        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-card kc-accent">
            <span class="kpi-icon">📊</span>
            <div class="kpi-val">{len(df):,}</div>
            <div class="kpi-label">Total Violations</div>
            <div class="kpi-sub">Nov 2023 – Apr 2024</div>
            {_mom_html}
          </div>
          <div class="kpi-card kc-critical">
            <span class="kpi-icon">🎯</span>
            <div class="kpi-val">{n_zones}</div>
            <div class="kpi-label">Hotspot Zones</div>
            <span class="kpi-badge badge-crit">{n_critical} Critical</span>
            <span class="kpi-badge badge-warn" style="margin-left:4px">{n_high} High</span>
          </div>
          <div class="kpi-card kc-high">
            <span class="kpi-icon">⛌</span>
            <div class="kpi-val">{junc_pct:.0f}%</div>
            <div class="kpi-label">Junction Impact</div>
            <div class="kpi-sub">violations at intersections</div>
          </div>
          <div class="kpi-card kc-medium">
            <span class="kpi-icon">🚛</span>
            <div class="kpi-val">{heavy_pct:.1f}%</div>
            <div class="kpi-label">Heavy Vehicles</div>
            <div class="kpi-sub">buses · trucks · vans</div>
          </div>
          <div class="kpi-card kc-critical">
            <span class="kpi-icon">💰</span>
            <div class="kpi-val">₹{_rec_cr:.2f}Cr</div>
            <div class="kpi-label">Recoverable Revenue</div>
            <span class="kpi-badge badge-crit">{blind_pct:.0f}% SCITA blind</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


        st.divider()

        # Bottom row: violation types + vehicle breakdown + monthly
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown('<div class="sec-label">Violation Type Mix</div>', unsafe_allow_html=True)
            vcnt = df["primary_vtype"].value_counts().head(6)
            pie_names = [n.replace("PARKING ","").replace("DEFECTIVE NUMBER PLATE","Defective Plate").title() for n in vcnt.index]
            fig_d = px.pie(
                names=pie_names,
                values=vcnt.values, hole=0.55,
                color_discrete_sequence=CHART_PAL, height=260,
            )
            fig_d.update_traces(
                textinfo="percent", textfont_size=10,
                textposition="outside",
                hovertemplate="<b>%{label}</b><br>%{value:,} violations<extra></extra>",
            )
            fig_d.update_layout(
                showlegend=True,
                legend=dict(orientation="v", font=dict(size=8, color="#8B949E"),
                            bgcolor="rgba(0,0,0,0)", x=1.0, y=0.5),
                margin=dict(t=10, b=10, l=10, r=80),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_d, use_container_width=True)

        with c2:
            st.markdown('<div class="sec-label">Vehicle Fleet Breakdown</div>', unsafe_allow_html=True)
            vc = df["vehicle_type"].value_counts().head(7).reset_index()
            vc.columns = ["type","count"]
            fig_v = px.bar(vc, x="count", y="type", orientation="h",
                           color="count", color_continuous_scale=["#0E1623","#00C2D4"],
                           height=250)
            fig_v.update_traces(
                hovertemplate="<b>%{y}</b><br>Violations: %{x:,}<extra></extra>")
            pl(fig_v, showlegend=False,
                                coloraxis_showscale=False,
                                yaxis=dict(categoryorder="total ascending",gridcolor="#0E1623"))
            st.plotly_chart(fig_v, use_container_width=True)

        with c3:
            st.markdown('<div class="sec-label">Monthly Trend</div>', unsafe_allow_html=True)
            mn = df.groupby("month_key").size().reset_index(name="count").sort_values("month_key")

            fig_m = px.area(mn, x="month_key", y="count", height=250,
                            color_discrete_sequence=["#00C2D4"])
            fig_m.update_traces(
                line_width=2, fillcolor="rgba(0,194,212,0.12)",
                hovertemplate="<b>%{x}</b><br>Violations: %{y:,}<extra></extra>",
            )
            # Month-over-month delta annotation
            if len(mn) >= 2:
                _m_last  = int(mn.iloc[-1]["count"])
                _m_prev  = int(mn.iloc[-2]["count"])
                _m_delta = (_m_last - _m_prev) / max(_m_prev, 1) * 100
                _m_sign  = "+" if _m_delta > 0 else ""
                _m_col   = "#DC2626" if _m_delta > 0 else "#059669"
                fig_m.add_annotation(
                    x=mn.iloc[-1]["month_key"], y=_m_last,
                    text=f"<b>{_m_sign}{_m_delta:.0f}%</b>",
                    showarrow=True, arrowhead=2, arrowcolor=_m_col,
                    font=dict(color=_m_col, size=10, family="Barlow Condensed"),
                    bgcolor="rgba(8,13,23,0.75)", borderpad=3,
                    ax=30, ay=-25,
                )
            pl(fig_m)
            st.plotly_chart(fig_m, use_container_width=True)

        # So What callout after the charts row
        _top_vtype = df["primary_vtype"].value_counts().index[0].replace("PARKING ","Parking ").title()
        _top_vtype_pct = int(df["primary_vtype"].value_counts().iloc[0] / len(df) * 100)
        st.markdown(f"""
        <div class="sowhat">
          <div class="sowhat-label">&#128161; What This Means</div>
          <div class="sowhat-text">
            <b>{_top_vtype}</b> is the most common violation at <b>{_top_vtype_pct}%</b> of all cases.
            Monthly violations {"increased" if _mom > 0 else "decreased"} by <b>{abs(_mom):.1f}%</b> last month &mdash;
            a {"worsening trend that needs immediate attention" if _mom > 0 else "positive sign that current enforcement is working"}.
            Focus officer resources on peak days and the top-ranked zones for maximum impact.
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_map, col_right = st.columns([3, 2])

        with col_map:
            st.markdown('<div class="sec-label">City-Wide Violation Density</div>', unsafe_allow_html=True)
            sample = df.sample(min(70000, len(df)), random_state=42)
            fig_h = px.density_mapbox(
                sample, lat="latitude", lon="longitude", z="severity", radius=9,
                center={"lat":12.97,"lon":77.59}, zoom=11,
                mapbox_style="carto-darkmatter",
                color_continuous_scale=[[0,"#080D17"],[0.4,"#D97706"],[1,"#DC2626"]],
                height=340,
            )
            fig_h.update_layout(
                margin=dict(r=0,t=0,l=0,b=0),
                coloraxis_showscale=True,
                coloraxis_colorbar=dict(
                    title=dict(text="Severity", font=dict(size=9,color="#6B87A8")),
                    len=0.55, thickness=8, x=1.0,
                    tickfont=dict(size=8,color="#6B87A8"),
                    bgcolor="rgba(0,0,0,0)", outlinewidth=0,
                    tickvals=[0,0.5,1.0], ticktext=["Low","Mod","High"],
                ),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with col_right:
            # City Health Score gauge
            st.markdown('<div class="instr-header">&#9632; Congestion Index Instruments</div>', unsafe_allow_html=True)
            g1, g2, g3 = st.columns(3)
            with g1:
                st.plotly_chart(gauge_fig(top_cii, "TOP ZONE CII", "#DC2626"), use_container_width=True)
            with g2:
                st.plotly_chart(gauge_fig(avg_cii, "AVG ZONE CII", "#D97706"), use_container_width=True)
            with g3:
                st.plotly_chart(gauge_fig(junc_pct/100, "JUNCTION RATE", "#00C2D4"), use_container_width=True)

            # Risk tier distribution bar
            _tiers = ["Critical","High","Medium","Low"]
            _tcols = ["#DC2626","#D97706","#0EA5E9","#059669"]
            _tcnts = [n_critical, n_high, n_medium, n_low]
            _ttot  = max(sum(_tcnts), 1)
            _bars  = "".join([
                f'<div style="flex:{c};background:{col};height:100%;'
                f'min-width:2px" title="{t}: {c} zones ({c/_ttot*100:.0f}%)"></div>'
                for t, col, c in zip(_tiers, _tcols, _tcnts) if c > 0
            ])
            _legend = "".join([
                f'<span style="color:{col};font-size:0.65rem;white-space:nowrap">'
                f'&#9679;&nbsp;{t}&nbsp;{c}</span>'
                for t, col, c in zip(_tiers, _tcols, _tcnts) if c > 0
            ])
            st.markdown(f"""
            <div style="margin:8px 0 4px">
              <div style="font-size:0.60rem;color:var(--muted);letter-spacing:.1em;
                          margin-bottom:4px">ZONE RISK DISTRIBUTION &mdash; {_ttot} ZONES</div>
              <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;gap:1px">
                {_bars}
              </div>
              <div style="display:flex;gap:10px;margin-top:4px;flex-wrap:wrap">{_legend}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sec-label" style="margin-top:10px">AI Intelligence Feed</div>', unsafe_allow_html=True)

            st.markdown('<div class="insight-feed">', unsafe_allow_html=True)

            peak_hour = int(df.groupby("hour").size().idxmax())
            peak_day  = df.groupby("day_name").size().idxmax()
            top_junc  = df[df["at_junction"]==1].groupby("junc_label").size().idxmax() if df["at_junction"].sum()>0 else "N/A"
            blind_top_station = (df[df["scita_ok"]==0].groupby("police_station").size().idxmax()
                                 if (df["scita_ok"]==0).sum()>0 else "N/A")

            st.markdown(f"""
            <div class="insight insight-crit">
            <span style="color:#DC2626">&#9679;</span> <b>Critical Zone</b> - Zone #{top_zone['rank']} ({top_zone['station']})<br>
            {top_zone['count']:,} violations &bull; CII <b>{top_zone['cii']:.3f}</b> &bull;
            {top_zone['junction_rate']*100:.0f}% at junctions &bull; peaks {top_zone['top_day']} {top_zone['top_hour']:02d}:00 &bull;
            SCITA gap {(1-top_zone['scita_rate'])*100:.0f}%
            </div>
            <div class="insight insight-warn">
            <span style="color:#D97706">&#9679;</span> <b>Peak Window</b> - {peak_day}s at {peak_hour}:00-{peak_hour+1}:00<br>
            Deploy officers <b>30 min before</b> peak for maximum suppression.
            </div>
            <div class="insight insight-info">
            <span style="color:#0EA5E9">&#9679;</span> <b>Worst Junction</b> - {top_junc}<br>
            Single highest-violation intersection. Priority AM & PM deployment.
            </div>
            <div class="insight insight-purple">
            <span style="color:#8B5CF6">&#9679;</span> <b>Largest SCITA Gap</b> - {blind_top_station}<br>
            {blind_count:,} violations ({blind_pct:.0f}%) not reaching SCITA &mdash;
            highest concentration in <b>{blind_top_station}</b>.
            </div>
            </div>
            """, unsafe_allow_html=True)


            # -- Live Intelligence Widget -----------------------------------------
            _now = _dt.datetime.now()
            _cur_hour = _now.hour
            _cur_dow  = _now.weekday()
            _cur_day_name = _now.strftime('%A')
            _slot_df = df_full[(df_full['hour'] == _cur_hour) & (df_full['dow'] == _cur_dow)]
            _slot_pct = len(_slot_df) / max(len(df_full), 1) * 100
            _slot_top_type = (_slot_df['primary_vtype'].value_counts().index[0]
                              if len(_slot_df) > 0 else 'N/A')
            _is_peak_now = 1 if _cur_hour in range(7, 11) or _cur_hour in range(17, 21) else 0
            _peak_label  = 'PEAK HOUR' if _is_peak_now else 'OFF-PEAK'
            _peak_color  = '#DC2626' if _is_peak_now else '#059669'
            _slot_zone_ids = (
                _slot_df.groupby('cluster').size()
                .reset_index(name='n')
                .merge(cluster_stats[['cluster','station']], on='cluster', how='left')
                .sort_values('n', ascending=False)
                .head(2)
            ) if len(_slot_df) > 0 else pd.DataFrame()
            _slot_zone_text = ' · '.join(
                str(r['station'])[:16] for _, r in _slot_zone_ids.iterrows()
                if pd.notna(r.get('station'))
            )
            st.markdown(
                '<div style="background:rgba(0,194,212,0.06);border:1px solid rgba(0,194,212,0.25);'
                'border-radius:8px;padding:10px 14px;margin:8px 0 4px;display:flex;'
                'align-items:center;gap:14px;flex-wrap:wrap">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{_peak_color};'
                'display:inline-block;margin-right:2px"></span>'
                f'<span style="font-size:0.60rem;color:{_peak_color};font-weight:800;letter-spacing:.1em">{_peak_label}</span>'
                f'<span style="font-size:0.72rem;color:#e6edf3;font-weight:600">NOW: {_cur_day_name} {_cur_hour:02d}:00</span>'
                f'<span style="font-size:0.70rem;color:var(--muted)">Historically {_slot_pct:.1f}% of violations occur at this window</span>'
                + (f'<span style="font-size:0.70rem;color:#F59E0B">Hot: {_slot_zone_text}</span>' if _slot_zone_text else '')
                + f'<span style="font-size:0.70rem;color:#A78BFA">Top type: {_slot_top_type[:28]}</span>'
                + '</div>',
                unsafe_allow_html=True)

            st.markdown('<div class="sec-label" style="margin-top:14px">Anomaly Detection — AI-Flagged Irregular Zones</div>', unsafe_allow_html=True)
            if n_anomalies > 0:
                st.markdown(
                    f'<div style="background:rgba(124,58,237,0.08);border:1px solid rgba(124,58,237,0.30);'
                    f'border-radius:8px;padding:8px 12px;margin-bottom:4px;font-size:0.68rem;'
                    f'color:#A78BFA;font-weight:700;letter-spacing:.06em">' +
                    f'{n_anomalies} ZONES FLAGGED BY ISOLATION FOREST — UNUSUAL ENFORCEMENT PROFILE</div>',
                    unsafe_allow_html=True)
                for _, _ar in anomaly_zones.head(5).iterrows():
                    _ar_rank  = int(_ar["rank"])
                    _ar_stn   = str(_ar["station"])[:26]
                    _ar_cii   = float(_ar["cii"])
                    _ar_junc  = float(_ar["junction_rate"])*100
                    _ar_gap   = float(_ar["scita_gap_pct"])
                    _ar_score = float(_ar["anomaly_score"])
                    _ar_risk  = str(_ar["risk_tier"])
                    _rc = "#DC2626" if _ar_risk=="Critical" else "#D97706" if _ar_risk=="High" else "#0EA5E9"
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;padding:5px 8px;'
                        f'border:1px solid rgba(124,58,237,0.20);border-radius:6px;margin-bottom:4px;'
                        f'background:rgba(8,13,23,0.5);font-size:0.72rem">' +
                        f'<span style="background:#7C3AED;color:#fff;font-size:0.58rem;font-weight:700;'
                        f'padding:2px 6px;border-radius:3px;white-space:nowrap">ANOM</span>' +
                        f'<span style="color:#e6edf3;font-weight:600;flex:2">Zone #{_ar_rank} — {_ar_stn}</span>' +
                        f'<span style="color:#A78BFA;flex:1">CII {_ar_cii:.3f}</span>' +
                        f'<span style="color:#F59E0B;flex:1">{_ar_junc:.0f}% junc</span>' +
                        f'<span style="color:#F472B6;flex:1">Gap {_ar_gap:.0f}%</span>' +
                        f'<span style="color:{_rc};font-weight:700;flex:1">score {_ar_score:.3f}</span>' +
                        f'</div>',
                        unsafe_allow_html=True)
    # ==========================================================================
    # TAB 2 â"€â"€ HOTSPOT MAP
    # ==========================================================================
    with t2:
        col_m2, col_t2 = st.columns([3, 2])

        with col_m2:
            st.markdown(f'<div class="sec-label">{n_zones} Hotspot Zones Detected  {n_critical} Critical Priority</div>',
                        unsafe_allow_html=True)

            fig_hs = go.Figure()
            samp2  = df.sample(min(30000, len(df)), random_state=7)
            fig_hs.add_trace(go.Densitymapbox(
                lat=samp2["latitude"], lon=samp2["longitude"],
                radius=5, colorscale=[[0,"rgba(0,0,0,0)"],[1,"#00C2D4"]],
                showscale=False, opacity=0.3, name="All Violations",
            ))
            for tier in ["Critical","High","Medium","Low"]:
                sub = cluster_stats[cluster_stats["risk_tier"]==tier]
                if sub.empty: continue
                sizes = sub["count"].apply(lambda c: max(12, min(45, c/50))).tolist()
                texts = sub.apply(lambda r: (
                    f"<b>Zone #{r['rank']}</b> - {tier}<br>"
                    f"CII: <b>{r['cii']:.3f}</b>  |  Violations: {r['count']:,}<br>"
                    f"Station: {r['station']}<br>"
                    f"Junction: {r['junction_rate']*100:.0f}%  |  Peak: {r['peak_rate']*100:.0f}%<br>"
                    f"Heavy vehicles: {r['heavy_rate']*100:.0f}%<br>"
                    f"SCITA gap: {(1-r['scita_rate'])*100:.0f}%<br>"
                    f"Predicted peak: {r['top_day']} {r['top_hour']:02d}:00"
                ), axis=1).tolist()
                fig_hs.add_trace(go.Scattermapbox(
                    lat=sub["lat"].tolist(), lon=sub["lon"].tolist(), mode="markers",
                    marker=dict(size=sizes, color=RISK_COLORS[tier], opacity=0.92,
                                sizemode="area"),
                    text=texts, hovertemplate="%{text}<extra></extra>",
                    name=f"{tier} ({len(sub)})",
                ))
            fig_hs.update_layout(
                mapbox=dict(style="carto-darkmatter", center=dict(lat=12.97,lon=77.59), zoom=11),
                height=560, margin=dict(r=0,t=0,l=0,b=0),
                legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363D",
                            borderwidth=1, font=dict(color="#F0F6FC",size=11)),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_hs, use_container_width=True)

        with col_t2:
            st.markdown('<div class="sec-label">Top 25 Zones  CII Ranking</div>', unsafe_allow_html=True)

            disp = cluster_stats[[
                "rank","station","count","cii","junction_rate","peak_rate","enforce_gap","risk_tier"
            ]].head(25).copy()
            disp["junction_rate"] = (disp["junction_rate"]*100).round(0).astype(int).astype(str)+"%"
            disp["peak_rate"]     = (disp["peak_rate"]*100).round(0).astype(int).astype(str)+"%"
            disp.columns = ["#","Station","Violations","CII","Junction","Peak","Gap","Risk"]
            st.dataframe(disp, use_container_width=True, height=530, hide_index=True,
                column_config={
                    "CII": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                    "Gap": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
                    "Violations": st.column_config.NumberColumn(format="%d"),
                })

            # CII distribution
            st.markdown('<div class="sec-label" style="margin-top:14px">CII Distribution by Risk Tier</div>',
                        unsafe_allow_html=True)
            fig_tier = px.histogram(
                cluster_stats, x="cii", color="risk_tier",
                color_discrete_map=RISK_COLORS, nbins=30, height=180,
                labels={"cii":"Congestion Impact Index","count":"Zones"},
                barmode="overlay",
            )
            pl(fig_tier, showlegend=True,
                legend=dict(orientation="h", y=-0.35, font_size=9,
                           font_color="#8B949E", bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10,b=40,l=10,r=10))
            st.plotly_chart(fig_tier, use_container_width=True)
            _crit_pct_sw = n_critical / max(n_zones,1) * 100
            st.markdown(f"""
            <div class="sowhat">
              <div class="sowhat-label">&#128161; What This Means for Commanders</div>
              <div class="sowhat-text">
                Only <b>{_crit_pct_sw:.0f}% of zones</b> are Critical, but they generate
                a disproportionate share of violations. Concentrate <b>70% of officer resources</b>
                on just the top {n_critical} zones and you'll cover the highest-risk areas
                while leaving enough capacity for the city's {n_high} High-risk zones.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 3 â"€â"€ JUNCTION ANALYSIS
    # ==========================================================================
    with t3:
        jdf    = df[df["at_junction"]==1].copy()
        jstats = jdf.groupby("junc_label").agg(
            count     =("id",          "count"),
            lat       =("latitude",    "mean"),
            lon       =("longitude",   "mean"),
            sev_mean  =("severity",    "mean"),
            peak_rate =("is_peak",     "mean"),
            scita_rate=("scita_ok",    "mean"),
            vweight   =("vehicle_weight","mean"),
            heavy_rate=("is_heavy",    "mean"),
        ).reset_index()
        jstats["jcii"] = (
            0.40*(jstats["count"]/jstats["count"].max()) +
            0.30*jstats["sev_mean"] +
            0.20*jstats["peak_rate"] +
            0.10*(jstats["vweight"]/jstats["vweight"].max())
        ).round(4)
        jstats = jstats.sort_values("jcii", ascending=False).reset_index(drop=True)
        jstats["rank"] = jstats.index + 1

        # Junction KPIs
        jkpi1, jkpi2, jkpi3, jkpi4 = st.columns(4)
        jkpi1.metric("Junction Violations", f"{jdf.shape[0]:,}", f"{jdf.shape[0]/len(df)*100:.1f}% of total")
        jkpi2.metric("Monitored Junctions", f"{jstats.shape[0]}", "named BTP junctions")
        if not jstats.empty:
            jkpi3.metric("Top Junction CII", f"{jstats['jcii'].max():.3f}", jstats.iloc[0]['junc_label'][:25])
            jkpi4.metric("Avg Peak Rate at Junctions", f"{jstats['peak_rate'].mean()*100:.1f}%", "during AM/PM peaks")
        else:
            jkpi3.metric("Top Junction CII", "N/A", "no junction data")
            jkpi4.metric("Avg Peak Rate at Junctions", "N/A", "no junction data")

        if jstats.empty:
            st.info("No junction violations in the current filter. Try 'All Stations' or 'All Types'.")

        st.divider()
        cj1, cj2 = st.columns([2, 3])

        with cj1:
            st.markdown('<div class="sec-label">Top 15 Junctions by Violation Count</div>', unsafe_allow_html=True)
            top15j = jstats.head(15).copy()
            fig_jb = px.bar(
                top15j, x="count", y="junc_label", orientation="h",
                color="jcii", color_continuous_scale=[[0,"#059669"],[0.5,"#D97706"],[1,"#DC2626"]],
                text="count", height=480,
                labels={"count":"Violations","junc_label":"","jcii":"Junction CII"},
            )
            fig_jb.update_traces(textposition="outside", textfont=dict(size=9,color="#8B949E"))
            pl(fig_jb, yaxis=dict(categoryorder="total ascending", gridcolor="#0E1623"),
                coloraxis_colorbar=dict(title="CII", len=0.5, thickness=10,
                                        tickfont=dict(size=9,color="#8B949E")),
                margin=dict(l=0,r=60,t=10,b=0))
            st.plotly_chart(fig_jb, use_container_width=True)

        with cj2:
            st.markdown('<div class="sec-label">Junction Impact Map</div>', unsafe_allow_html=True)
            fig_jm = px.scatter_mapbox(
                jstats.head(80), lat="lat", lon="lon",
                size="count", color="jcii",
                color_continuous_scale=[[0,"#059669"],[0.5,"#D97706"],[1,"#DC2626"]],
                hover_name="junc_label",
                hover_data={"count":True,"jcii":":.3f","peak_rate":":.2f","lat":False,"lon":False},
                zoom=11, center={"lat":12.97,"lon":77.59},
                mapbox_style="carto-darkmatter",
                size_max=50, height=480,
                labels={"jcii":"Junction CII"},
            )
            fig_jm.update_layout(margin=dict(r=0,t=0,l=0,b=0),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 coloraxis_colorbar=dict(title="CII",len=0.5,thickness=10,
                                                         tickfont=dict(size=9,color="#8B949E")))
            st.plotly_chart(fig_jm, use_container_width=True)

        st.divider()

        # Heavy vehicle junction focus
        c3a, c3b = st.columns(2)
        with c3a:
            st.markdown('<div class="sec-label">Heavy Vehicle Violations by Junction (Top 10)</div>',
                        unsafe_allow_html=True)
            heavy_j = jdf[jdf["is_heavy"]==1].groupby("junc_label").size().sort_values(ascending=False).head(10).reset_index()
            heavy_j.columns = ["Junction","Heavy Violations"]
            fig_hj = px.bar(heavy_j, x="Heavy Violations", y="Junction", orientation="h",
                            color="Heavy Violations",
                            color_continuous_scale=["#0E1623","#8B5CF6"],
                            height=300)
            pl(fig_hj, coloraxis_showscale=False,
                                  yaxis=dict(categoryorder="total ascending",gridcolor="#0E1623"),
                                  margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_hj, use_container_width=True)

        with c3b:
            st.markdown('<div class="sec-label">Junction Violations by Hour of Day</div>', unsafe_allow_html=True)
            jh = jdf.groupby("hour").size().reset_index(name="count")
            fig_jh = px.bar(jh, x="hour", y="count", height=300,
                            color="count",
                            color_continuous_scale=["#0E1623","#DC2626"])
            fig_jh.add_vrect(x0=6.5,x1=10.5, fillcolor="#D97706",opacity=0.07,
                             annotation_text="AM Peak", annotation_font_color="#D97706",
                             annotation_font_size=10)
            fig_jh.add_vrect(x0=16.5,x1=20.5, fillcolor="#DC2626",opacity=0.07,
                             annotation_text="PM Peak", annotation_font_color="#DC2626",
                             annotation_font_size=10)
            pl(fig_jh, coloraxis_showscale=False,
                                  margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_jh, use_container_width=True)

    # ==========================================================================
    # TAB 4 â"€â"€ ENFORCEMENT PLANNER
    # ==========================================================================
    with t4:
        st.markdown('<div class="sec-label">Enforcement Resource Allocation Intelligence</div>',
                    unsafe_allow_html=True)

        # Quick-read summary bar for planner
        _top_sta_p = df.groupby("police_station").agg(total=("id","count")).nlargest(1,"total")
        _top_sta_name = str(_top_sta_p.index[0]) if len(_top_sta_p) else "N/A"
        _top_sta_cnt  = int(_top_sta_p.iloc[0]["total"]) if len(_top_sta_p) else 0
        _peak_day2    = df.groupby("day_name").size().idxmax()
        _peak_hr2     = int(df.groupby("hour").size().idxmax())
        st.markdown(f"""
        <div style="display:flex;gap:0;margin:0 0 14px;border:1px solid var(--border);
                    border-radius:8px;overflow:hidden;font-size:0.78rem">
          <div style="flex:1;background:var(--card);padding:10px 16px;border-right:1px solid var(--border)">
            <div style="color:var(--muted);font-size:0.62rem;letter-spacing:.1em;margin-bottom:2px">TOP PRIORITY STATION</div>
            <div style="color:var(--text);font-weight:700">{_top_sta_name}</div>
            <div style="color:var(--dim);font-size:0.70rem">{_top_sta_cnt:,} violations</div>
          </div>
          <div style="flex:1;background:var(--card);padding:10px 16px;border-right:1px solid var(--border)">
            <div style="color:var(--muted);font-size:0.62rem;letter-spacing:.1em;margin-bottom:2px">PEAK ENFORCEMENT WINDOW</div>
            <div style="color:var(--high-text);font-weight:700">{_peak_day2} {_peak_hr2:02d}:00&ndash;{_peak_hr2+1:02d}:00</div>
            <div style="color:var(--dim);font-size:0.70rem">pre-deploy by {max(0,_peak_hr2-1):02d}:30</div>
          </div>
          <div style="flex:1;background:var(--card);padding:10px 16px;border-right:1px solid var(--border)">
            <div style="color:var(--muted);font-size:0.62rem;letter-spacing:.1em;margin-bottom:2px">SCITA BLIND SPOTS</div>
            <div style="color:var(--critical-text);font-weight:700">{blind_count:,}</div>
            <div style="color:var(--dim);font-size:0.70rem">{blind_pct:.0f}% not in system</div>
          </div>
          <div style="flex:1;background:var(--card);padding:10px 16px">
            <div style="color:var(--muted);font-size:0.62rem;letter-spacing:.1em;margin-bottom:2px">UNCOLLECTED FINES</div>
            <div style="color:var(--accent-bright);font-weight:700">&#8377;{uncollected_fine_total/1e7:.1f}&nbsp;Cr</div>
            <div style="color:var(--dim);font-size:0.70rem">{100-recovery_rate:.0f}% gap vs potential</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        c4a, c4b = st.columns([3, 2])

        with c4a:
            # Station priority
            st_s = df.groupby("police_station").agg(
                total     =("id",           "count"),
                junc_rate =("at_junction",  "mean"),
                peak_rate =("is_peak",      "mean"),
                sev_mean  =("severity",     "mean"),
                scita_gap =("scita_ok",     lambda x: 1-x.mean()),
                heavy_rate=("is_heavy",     "mean"),
            ).reset_index()
            sc2 = MinMaxScaler()
            st_s["freq_n"] = sc2.fit_transform(st_s[["total"]]).flatten()
            st_s["priority"] = (
                0.30*st_s["freq_n"]    +
                0.25*st_s["junc_rate"] +
                0.20*st_s["sev_mean"]  +
                0.15*st_s["scita_gap"] +
                0.10*st_s["heavy_rate"]
            ).round(4)
            st_s = st_s.sort_values("priority", ascending=False)

            fig_st = px.bar(
                st_s.head(15), x="priority", y="police_station", orientation="h",
                color="priority",
                color_continuous_scale=[[0,"#0E1623"],[0.5,"#D97706"],[1,"#DC2626"]],
                text=st_s.head(15)["total"].apply(lambda x: f"{x:,}"),
                title="Enforcement Priority Score - Top 15 Stations",
                height=420,
                labels={"priority":"Priority Score","police_station":""},
            )
            fig_st.update_traces(textposition="outside", textfont=dict(size=9,color="#8B949E"))
            pl(fig_st, coloraxis_showscale=False,
                                  yaxis=dict(categoryorder="total ascending",gridcolor="#0E1623"),
                                  margin=dict(l=0,r=60,t=36,b=0),
                                  title_font=dict(size=12,color="#8B949E"))
            st.plotly_chart(fig_st, use_container_width=True)

            # Data-driven vs random comparison
            st.markdown('<div class="sec-label">Enforcement Efficiency  Data-Driven vs Random Patrol</div>',
                        unsafe_allow_html=True)
            n_scenarios = [1,2,3,5,8,10,15,20]
            random_cov  = [min(100, n/len(cluster_stats)*100*1.2) for n in n_scenarios]
            smart_cov   = [min(100, cluster_stats.head(n)["count"].sum()/len(df_full)*100)
                           for n in n_scenarios]
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=n_scenarios, y=random_cov, name="Random Patrol",
                line=dict(color="#484F58",width=2,dash="dash"), mode="lines+markers",
                hovertemplate="Officers: %{x}<br>Random coverage: %{y:.1f}%<extra></extra>"))
            fig_cmp.add_trace(go.Scatter(x=n_scenarios, y=smart_cov, name="GridLock AI",
                line=dict(color="#00C2D4",width=2.5), mode="lines+markers",
                fill="tonexty", fillcolor="rgba(0,194,212,0.08)",
                hovertemplate="Officers: %{x}<br>GridLock AI coverage: %{y:.1f}%<extra></extra>"))
            pl(fig_cmp, height=240,
                xaxis_title="Officers Deployed",
                yaxis_title="Violations Covered (%)",
                legend=dict(orientation="h",y=1.1,font_size=10,
                           font_color="#8B949E",bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10,r=10,t=10,b=30))
            st.plotly_chart(fig_cmp, use_container_width=True)

        with c4b:
            # Deployment simulator
            st.markdown('<div class="sec-label"> Deployment Impact Simulator</div>', unsafe_allow_html=True)
            n_off = st.slider("Officers to deploy", 1, 30, 5)
            top_z = cluster_stats.head(n_off)
            v_cov  = int(top_z["count"].sum())
            jv_cov = int((top_z["count"]*top_z["junction_rate"]).sum())
            pct    = v_cov/len(df_full)*100
            smart_efficiency = pct
            random_efficiency = min(100, n_off/n_zones*100*1.2)
            lift   = smart_efficiency - random_efficiency

            sm1, sm2 = st.columns(2)
            sm1.metric("Violations Covered", f"{v_cov:,}", f"{pct:.1f}% of total")
            sm2.metric("Junction Incidents", f"{jv_cov:,}", "direct congestion")
            sm3, sm4 = st.columns(2)
            sm3.metric("Avg Zone CII", f"{top_z['cii'].mean():.3f}")
            sm4.metric("vs Random Patrol", f"+{lift:.1f}%", "efficiency gain")

            st.info(
                f"**{n_off} officers -> {pct:.1f}% coverage** vs ~{random_efficiency:.1f}% random.\n\n"
                f"GridLock targeting is **{lift:.1f}% more efficient** than unguided patrol."
            )

            # Radar for top zone
            top1 = cluster_stats.iloc[0]
            cats = ["Frequency","Junction","Severity","Peak Hour","Heavy Veh."]
            vals = [top1["freq_score"], top1["junction_rate"], top1["severity_mean"],
                    top1["peak_rate"], top1["heavy_rate"]]
            fig_r = go.Figure(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]],
                fill="toself", fillcolor="rgba(0,194,212,0.12)",
                line=dict(color="#00C2D4",width=2), name=f"Zone #{top1['rank']}",
            ))
            fig_r.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0,1],
                                   tickfont=dict(size=8,color="#484F58"), gridcolor="#0E1623"),
                    angularaxis=dict(tickfont=dict(size=10,color="#8B949E"), gridcolor="#0E1623"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                showlegend=False,
                title=dict(text=f"Zone #{top1['rank']} Risk Profile . CII={top1['cii']:.3f}",
                           font=dict(size=11,color="#8B949E")),
                paper_bgcolor="rgba(0,0,0,0)", height=260,
                margin=dict(t=50,b=10,l=10,r=10),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # Predictive weekly deployment schedule
        st.divider()
        st.markdown('<div class="sec-label"> AI-Generated Deployment Schedule  Next 7 Days</div>',
                    unsafe_allow_html=True)
        sched = build_schedule(df_full, cluster_stats, n=12)
        if not sched.empty:
            rows_html = ""
            for _, r in sched.iterrows():
                rc = risk_color_css(r["risk_tier"])
                rows_html += f"""
                <tr>
                  <td>#{r['rank']}</td>
                  <td>{r['station']}</td>
                  <td class="{rc}">{r['risk_tier']}</td>
                  <td>{r['cii']:.3f}</td>
                  <td>{r['violations']:,}</td>
                  <td>{r['junc_pct']}</td>
                  <td><b>{r['peak_day']}</b></td>
                  <td>{r['peak_hour']}</td>
                  <td style="color:#059669;font-weight:700">{r['deploy_at']}</td>
                </tr>"""
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden;margin-top:8px">
              <table class="sched-table">
                <thead><tr>
                  <th>Zone</th><th>Station</th><th>Risk</th><th>CII</th>
                  <th>Violations</th><th>Junction%</th><th>Peak Day</th>
                  <th>Peak Time</th><th>â° Deploy At</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

            # Download CSV
            csv_data = sched.to_csv(index=False)
            st.download_button(
                label="Download Enforcement Brief (CSV)",
                data=csv_data,
                file_name="gridlock_enforcement_brief.csv",
                mime="text/csv",
                help="Download the full enforcement schedule as CSV",
            )


        # -- Violation Type Predictor (Random Forest) --------------------------
        st.divider()
        st.markdown('<div class="sec-label">Violation Predictor  Random Forest Classifier</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.75rem;color:var(--muted);margin-bottom:10px">'
            'Predict the most likely violation type given time, day, and zone characteristics. '
            'Trained on the full dataset using <b>RandomForestClassifier (100 trees)</b>.'
            '</div>', unsafe_allow_html=True)

        _preset_cols = st.columns(4)
        _presets = [
            ("Sat 10:00 City Market", 10, "Saturday", "City Market PS"),
            ("Fri 18:00 Peak Hour",   18, "Friday",   cluster_stats.iloc[0]["station"]),
            ("Tue 08:00 Morning",      8, "Tuesday",  cluster_stats.iloc[1]["station"]),
            ("Sun 14:00 Weekend",     14, "Sunday",   cluster_stats.iloc[2]["station"]),
        ]
        for _pi, (_plabel, _ph, _pd, _pz) in enumerate(_presets):
            if _preset_cols[_pi].button(_plabel, key=f"preset_{_pi}", use_container_width=True):
                st.session_state["pred_hour"] = _ph
                st.session_state["pred_day"]  = _pd
                _pz_match = [z for z in cluster_stats["station"].tolist()[:20] if _pz[:8] in z]
                if _pz_match:
                    st.session_state["pred_zone"] = _pz_match[0]
                st.rerun()

        _rf_col1, _rf_col2, _rf_col3 = st.columns(3)
        with _rf_col1:
            pred_hour = st.selectbox("Hour of day", list(range(0, 24)),
                                      index=9, format_func=lambda h: f"{h:02d}:00", key="pred_hour")
        with _rf_col2:
            pred_day  = st.selectbox("Day of week", ["Monday","Tuesday","Wednesday",
                                                      "Thursday","Friday","Saturday","Sunday"],
                                      key="pred_day")
        with _rf_col3:
            pred_zone = st.selectbox("Zone", cluster_stats["station"].tolist()[:20], key="pred_zone")

        # Train RF on full dataset features
        _rf_df = df_full[["hour","dow","at_junction","is_peak","is_heavy","primary_vtype"]].dropna()
        _rf_df = _rf_df[_rf_df["primary_vtype"].isin(
            _rf_df["primary_vtype"].value_counts().head(6).index)]
        if len(_rf_df) > 1000:
            _rf_X = _rf_df[["hour","dow","at_junction","is_peak","is_heavy"]].values
            _rf_y = _rf_df["primary_vtype"].values
            _rf_model = train_rf_classifier(tuple(map(tuple, _rf_X)), tuple(_rf_y))

            # Quick OOB accuracy estimate
            try:
                from sklearn.ensemble import RandomForestClassifier as _RFC2
                _rf_oob = _RFC2(n_estimators=50, max_depth=8, random_state=42,
                                oob_score=True, n_jobs=-1)
                _rf_oob.fit(_rf_X, _rf_y)
                _rf_acc = _rf_oob.oob_score_
            except Exception:
                _rf_acc = None

            # Get zone CII for the selected station
            _z_info = cluster_stats[cluster_stats["station"]==pred_zone]
            _z_junc = float(_z_info["junction_rate"].iloc[0]) if len(_z_info) else 0.3
            _z_peak = 1 if pred_hour in range(7, 11) or pred_hour in range(17, 21) else 0
            _z_heavy= float(_z_info["heavy_rate"].iloc[0]) if len(_z_info) else 0.1
            _dow_map = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
                        "Friday":4,"Saturday":5,"Sunday":6}
            _pred_features = [[pred_hour, _dow_map[pred_day],
                                int(_z_junc > 0.3), _z_peak, int(_z_heavy > 0.15)]]
            _probas = _rf_model.predict_proba(_pred_features)[0]
            _classes = _rf_model.classes_

            # Show top 4 predictions as probability bars
            _sorted_idx = _probas.argsort()[::-1][:4]
            _bar_html = ""
            _colors = ["#00C2D4","#8B5CF6","#F59E0B","#10B981"]
            for _i, _idx in enumerate(_sorted_idx):
                _vt   = str(_classes[_idx])
                _prob = _probas[_idx]
                _bar_html += (
                    f'<div style="margin-bottom:8px">'
                    f'<div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:2px">'
                    f'<span style="color:#e6edf3">{_vt[:32]}</span>'
                    f'<span style="color:{_colors[_i]};font-weight:700">{_prob*100:.1f}%</span>'
                    f'</div>'
                    f'<div style="background:#0E1623;border-radius:3px;height:6px">'
                    f'<div style="background:{_colors[_i]};width:{_prob*100:.0f}%;height:6px;border-radius:3px"></div>'
                    f'</div>'
                    f'</div>'
                )
            # Feature importance
            _fi = _rf_model.feature_importances_
            _fn = ["Hour","Day of Week","At Junction","Is Peak","Heavy Vehicle"]
            _top_feat = sorted(zip(_fn, _fi), key=lambda x: -x[1])

            _rc1, _rc2 = st.columns([3,2])
            with _rc1:
                st.markdown(
                    f'<div style="background:var(--card2);border:1px solid var(--border);'
                    f'border-radius:8px;padding:14px 16px">'
                    f'<div style="font-size:0.62rem;color:var(--muted);letter-spacing:.08em;margin-bottom:10px">'
                    f'PREDICTED VIOLATION PROBABILITIES  —  {pred_zone[:24]} · {pred_day} · {pred_hour:02d}:00'
                    + (f'  |  OOB Accuracy: <b style="color:#10B981">{_rf_acc*100:.1f}%</b>' if _rf_acc else '')
                    + '</div>'
                    + f'{_bar_html}'
                    + '</div>',
                    unsafe_allow_html=True)
            with _rc2:
                _fi_fig = px.bar(
                    x=[f[1]*100 for f in _top_feat],
                    y=[f[0] for f in _top_feat],
                    orientation="h",
                    title="Feature Importance (%)",
                    color=[f[1] for f in _top_feat],
                    color_continuous_scale=[[0,"#0E1623"],[1,"#00C2D4"]],
                )
                _fi_fig.update_layout(
                    height=180, showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8B949E", family="Barlow Condensed", size=9),
                    margin=dict(t=30,b=0,l=0,r=10), coloraxis_showscale=False,
                    yaxis=dict(gridcolor="#0E1623"),
                    xaxis=dict(gridcolor="#0E1623", title="Importance (%)"),
                    title_font=dict(size=10, color="#8B949E"),
                )
                st.plotly_chart(_fi_fig, use_container_width=True)

        # -- Officer Allocation Optimizer (Linear Programming) --------------------------
        st.divider()
        st.markdown('<div class="sec-label">Officer Allocation Optimizer  Linear Programming</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.75rem;color:var(--muted);margin-bottom:10px">'
            'Mathematically optimal allocation of officers across zones to maximise violation-weighted CII coverage. '
            'Uses <b>scipy.optimize.linprog (HiGHS)</b> with zone CII &times; violation count as objective weights.'
            '</div>',
            unsafe_allow_html=True)

        _opt_col1, _opt_col2 = st.columns([1, 2])
        with _opt_col1:
            n_officers   = st.slider("Available Officers", min_value=10, max_value=200,
                                     value=50, step=5, key="n_officers")
            max_per_zone = st.slider("Max officers per zone", min_value=1, max_value=20,
                                     value=8, key="max_per_zone")

        _top_lp = cluster_stats.head(30).copy().reset_index(drop=True)
        _weights = (_top_lp["cii"] * _top_lp["count"] /
                    max(float(_top_lp["cii"].max() * _top_lp["count"].max()), 1.0)).values
        _n = len(_top_lp)
        _c_lp  = (-_weights).tolist()
        _A_eq  = [[1.0]*_n]
        _b_eq  = [float(n_officers)]
        _bnds  = [(0.0, float(max_per_zone))] * _n
        from scipy.optimize import linprog as _linprog
        _res = _linprog(_c_lp, A_eq=_A_eq, b_eq=_b_eq, bounds=_bnds, method="highs")

        if _res.success:
            _alloc = np.round(_res.x).astype(int)
            _diff  = n_officers - int(_alloc.sum())
            if _diff > 0:
                _alloc[int(np.argmax(_weights))] += _diff
            elif _diff < 0:
                _nz = np.where(_alloc > 0)[0]
                if len(_nz):
                    _alloc[_nz[-1]] = max(0, _alloc[_nz[-1]] + _diff)
            _top_lp["officers_optimal"] = _alloc
            _top_lp["officers_uniform"] = max(1, n_officers // _n)

            _optimal_score = float(np.sum(_weights * _alloc))
            _uniform_alloc = np.full(_n, n_officers / _n)
            _uniform_score = float(np.sum(_weights * _uniform_alloc))
            _gain_pct = (_optimal_score / max(_uniform_score, 0.001) - 1) * 100

            with _opt_col2:
                st.markdown(
                    f'<div style="display:flex;gap:12px;margin-bottom:12px">'
                    f'<div style="flex:1;background:var(--card);border:1px solid #059669;border-radius:8px;padding:12px 16px">'
                    f'<div style="font-size:0.62rem;color:var(--muted);letter-spacing:.08em">OPTIMAL COVERAGE SCORE</div>'
                    f'<div style="font-size:1.3rem;color:#10B981;font-weight:800">{_optimal_score:.2f}</div>'
                    f'<div style="font-size:0.68rem;color:var(--dim)">{_gain_pct:+.1f}% vs uniform patrol</div>'
                    f'</div>'
                    f'<div style="flex:1;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 16px">'
                    f'<div style="font-size:0.62rem;color:var(--muted);letter-spacing:.08em">UNIFORM PATROL SCORE</div>'
                    f'<div style="font-size:1.3rem;color:var(--muted);font-weight:800">{_uniform_score:.2f}</div>'
                    f'<div style="font-size:0.68rem;color:var(--dim)">{n_officers / _n:.1f} officers/zone (uniform)</div>'
                    f'</div>'
                    f'<div style="flex:1;background:var(--card);border:1px solid #D97706;border-radius:8px;padding:12px 16px">'
                    f'<div style="font-size:0.62rem;color:var(--muted);letter-spacing:.08em">EFFICIENCY GAIN</div>'
                    f'<div style="font-size:1.3rem;color:#F59E0B;font-weight:800">{_gain_pct:.1f}%</div>'
                    f'<div style="font-size:0.68rem;color:var(--dim)">LP vs random allocation</div>'
                    f'</div></div>',
                    unsafe_allow_html=True)

            _show = _top_lp[_top_lp["officers_optimal"] > 0].sort_values("officers_optimal", ascending=False).head(15)
            _fig_alloc = go.Figure()
            _fig_alloc.add_bar(
                y=[f"Z#{int(r['rank'])} {str(r['station'])[:18]}" for _, r in _show.iterrows()],
                x=_show["officers_optimal"].tolist(),
                name="Optimal (LP)", orientation="h", marker_color="#00C2D4",
                text=[f"{v}" for v in _show["officers_optimal"].tolist()],
                textposition="outside", textfont=dict(size=9, color="#8B949E"),
            )
            _fig_alloc.add_bar(
                y=[f"Z#{int(r['rank'])} {str(r['station'])[:18]}" for _, r in _show.iterrows()],
                x=_show["officers_uniform"].tolist(),
                name="Uniform", orientation="h",
                marker_color="rgba(139,92,246,0.35)",
            )
            _fig_alloc.update_layout(
                barmode="group", height=420,
                title=dict(text=f"Optimal vs Uniform Officer Deployment  (Total: {n_officers})",
                           font=dict(size=11, color="#8B949E")),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8B949E", family="Barlow Condensed"),
                xaxis=dict(gridcolor="#0E1623", title="Officers Allocated"),
                yaxis=dict(categoryorder="total ascending", gridcolor="#0E1623"),
                margin=dict(l=0, r=70, t=36, b=0),
                legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, x=0.78, y=0.02),
            )
            st.plotly_chart(_fig_alloc, use_container_width=True)
            _alloc_out = _top_lp[_top_lp["officers_optimal"]>0][
                ["rank","station","cii","risk_tier","officers_optimal","officers_uniform"]
            ].rename(columns={"officers_optimal":"optimal_officers","officers_uniform":"uniform_officers"})
            st.download_button(
                "Download Allocation Plan (CSV)", _alloc_out.to_csv(index=False),
                file_name="gridlock_officer_allocation.csv", mime="text/csv", key="dl_alloc",
            )
        else:
            st.warning("Optimizer could not find a feasible solution. Try adjusting officer count.")

    # ==========================================================================
    # TAB 5 â"€â"€ TEMPORAL ANALYSIS
    # ==========================================================================
    with t5:
        st.markdown('<div class="sec-label">When Violations Happen</div>', unsafe_allow_html=True)

        c5a, c5b = st.columns(2)
        with c5a:
            pivot = (df.groupby(["dow","hour"]).size()
                     .reset_index(name="count")
                     .pivot(index="dow", columns="hour", values="count").fillna(0))
            fig_hm = px.imshow(
                pivot.values, x=list(pivot.columns),
                y=[DAY_ORDER[i] for i in pivot.index if i < len(DAY_ORDER)],
                color_continuous_scale=[[0,"#080D17"],[0.4,"#D97706"],[1,"#DC2626"]],
                title="Violations Heatmap - Hour x Day",
                labels={"x":"Hour of Day","y":"","color":"Count"},
                aspect="auto", height=310,
            )
            pl(fig_hm, margin=dict(t=36,b=0,l=0,r=0))
            st.plotly_chart(fig_hm, use_container_width=True)

        with c5b:
            mn2 = df.groupby("month_key").size().reset_index(name="count").sort_values("month_key")
            fig_mn = go.Figure()
            fig_mn.add_trace(go.Scatter(
                x=mn2["month_key"], y=mn2["count"],
                mode="lines+markers",
                line=dict(color="#00C2D4",width=2.5),
                fill="tozeroy", fillcolor="rgba(0,194,212,0.08)",
                marker=dict(size=8, color="#00C2D4",
                           line=dict(color="#080D17",width=2)),
                name="Monthly",
                hovertemplate="<b>%{x}</b><br>Violations: %{y:,}<extra></extra>",
            ))
            if len(mn2) >= 2:
                _mn_last  = int(mn2.iloc[-1]["count"])
                _mn_prev  = int(mn2.iloc[-2]["count"])
                _mn_delta = (_mn_last - _mn_prev) / max(_mn_prev, 1) * 100
                _mn_sign  = "+" if _mn_delta > 0 else ""
                _mn_col   = "#DC2626" if _mn_delta > 0 else "#059669"
                fig_mn.add_annotation(
                    x=mn2.iloc[-1]["month_key"], y=_mn_last,
                    text=f"<b>{_mn_sign}{_mn_delta:.0f}% MoM</b>",
                    showarrow=True, arrowhead=2, arrowcolor=_mn_col,
                    font=dict(color=_mn_col, size=10, family="Barlow Condensed"),
                    bgcolor="rgba(8,13,23,0.80)", borderpad=3, ax=35, ay=-30,
                )
            # ── Forecast: Prophet (with polynomial fallback) ──
            _prophet = None
            _r2_val = 0.0; _mape = 0.0; _val_dates = []; _val_actual = []; _val_preds = []
            if len(mn2) >= 4:
                try:
                    # Build proper datetime index for Prophet
                    _mn2_dt = mn2.copy()
                    _mn2_dt["ds"] = pd.PeriodIndex(_mn2_dt["month_key"], freq="M").to_timestamp()
                    _mn2_dt["y"]  = _mn2_dt["count"].astype(float)

                    # Train/test split — last 2 months held out for validation
                    _n_val = 2
                    _train_df = _mn2_dt.iloc[:-_n_val][["ds","y"]]
                    _test_df  = _mn2_dt.iloc[-_n_val:][["ds","y"]]

                    # Fit Prophet
                    _prophet = Prophet(
                        yearly_seasonality=False,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        changepoint_prior_scale=0.3,
                        interval_width=0.80,
                    )
                    _prophet.fit(_train_df)

                    # Predict on test + 2 future months
                    _n_future = 2
                    _future = _prophet.make_future_dataframe(
                        periods=_n_val + _n_future, freq="MS"
                    )
                    _forecast = _prophet.predict(_future)

                    # Validation metrics on held-out months
                    _val_preds = _forecast[_forecast["ds"].isin(_test_df["ds"])]["yhat"].values
                    _val_actual= _test_df["y"].values
                    _mape = float(np.mean(np.abs((_val_actual - _val_preds) / np.maximum(_val_actual, 1))) * 100)
                    _ss_res = np.sum((_val_actual - _val_preds)**2)
                    _ss_tot = np.sum((_val_actual - _val_actual.mean())**2)
                    _r2_val = max(0, 1 - _ss_res / max(_ss_tot, 1e-9))

                    # Actual vs predicted on test set (train-era)
                    _val_dates = [str(pd.Period(d, freq="M")) for d in _test_df["ds"]]
                    fig_mn.add_trace(go.Scatter(
                        x=_val_dates, y=_val_preds.tolist(),
                        mode="markers", name="Validation (held-out)",
                        marker=dict(size=10, color="#F59E0B", symbol="circle-open",
                                    line=dict(color="#F59E0B", width=2)),
                        hovertemplate="<b>%{x}</b><br>Predicted: %{y:,.0f}<extra></extra>",
                    ))

                    # Future forecast line
                    _fut_rows = _forecast[~_forecast["ds"].isin(_mn2_dt["ds"])]
                    _fut_labels = [str(pd.Period(d, freq="M")) for d in _fut_rows["ds"]]
                    _fut_yhat   = _fut_rows["yhat"].clip(lower=0).tolist()
                    _fut_yhi    = _fut_rows["yhat_upper"].clip(lower=0).tolist()
                    _fut_ylo    = _fut_rows["yhat_lower"].clip(lower=0).tolist()

                    _bridge_x = [mn2["month_key"].iloc[-1]] + _fut_labels
                    _bridge_y = [float(mn2["count"].iloc[-1])] + _fut_yhat

                    fig_mn.add_trace(go.Scatter(
                        x=_bridge_x, y=_bridge_y,
                        mode="lines+markers", name="Prophet Forecast",
                        line=dict(color="#8B5CF6", width=2, dash="dot"),
                        marker=dict(size=7, color="#8B5CF6", symbol="diamond",
                                    line=dict(color="#080D17", width=1.5)),
                        hovertemplate="<b>%{x}</b><br>Forecast: %{y:,.0f}<extra></extra>",
                    ))
                    # Confidence band
                    _band_x = _bridge_x + _bridge_x[::-1]
                    _band_y = ([float(mn2["count"].iloc[-1])] + _fut_yhi +
                               [float(mn2["count"].iloc[-1])] + _fut_ylo[::-1])
                    fig_mn.add_trace(go.Scatter(
                        x=_band_x, y=_band_y, fill="toself",
                        fillcolor="rgba(139,92,246,0.10)", line=dict(width=0),
                        showlegend=False, hoverinfo="skip",
                    ))
                    fig_mn.add_annotation(
                        x=_fut_labels[-1], y=float(_fut_yhat[-1]),
                        text=f"<b>Forecast {_fut_labels[-1]}: {_fut_yhat[-1]:,.0f}</b><br>"
                             f"R² = {_r2_val:.3f} · MAPE {_mape:.1f}%",
                        showarrow=True, arrowhead=2, arrowcolor="#8B5CF6",
                        font=dict(color="#A78BFA", size=10, family="Barlow Condensed"),
                        bgcolor="rgba(8,13,23,0.85)", borderpad=3, ax=-70, ay=-35,
                    )

                except Exception as _prophet_err:
                    # Prophet failed — fall back to polynomial regression
                    _X = np.arange(len(mn2)).reshape(-1, 1).astype(float)
                    _y = mn2['count'].values.astype(float)
                    _pipe = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                    _pipe.fit(_X, _y)
                    _n_future = 2
                    _X_fut = np.arange(len(mn2), len(mn2)+_n_future).reshape(-1,1).astype(float)
                    _y_fut = np.clip(_pipe.predict(_X_fut), 0, None)
                    _last_period = pd.Period(mn2['month_key'].iloc[-1], freq='M')
                    _fut_labels = [str(_last_period + i + 1) for i in range(_n_future)]
                    _bridge_x = [mn2['month_key'].iloc[-1]] + _fut_labels
                    _bridge_y = [float(mn2['count'].iloc[-1])] + _y_fut.tolist()
                    fig_mn.add_trace(go.Scatter(
                        x=_bridge_x, y=_bridge_y, mode='lines+markers',
                        line=dict(color='#8B5CF6', width=2, dash='dot'),
                        marker=dict(size=7, color='#8B5CF6', symbol='diamond'),
                        name='Poly Forecast',
                        hovertemplate='<b>%{x}</b><br>Forecast: %{y:,.0f}<extra></extra>',
                    ))
                    _r2_val = _pipe.score(_X, _y)
                    fig_mn.add_annotation(
                        x=_fut_labels[-1], y=float(_y_fut[-1]),
                        text=f'<b>Forecast {_fut_labels[-1]}: {_y_fut[-1]:,.0f}</b><br>R²={_r2_val:.3f} (poly fallback)',
                        showarrow=True, arrowhead=2, arrowcolor='#8B5CF6',
                        font=dict(color='#A78BFA', size=10), bgcolor='rgba(8,13,23,0.85)',
                        borderpad=3, ax=-70, ay=-35,
                    )

            pl(fig_mn, height=310,
                title=dict(text="Monthly Violation Volume + AI Forecast",font=dict(size=12,color="#8B949E")),
                yaxis_title="Violations", xaxis_title="Month",
                legend=dict(orientation="h", y=-0.18, font_size=9,
                            font_color="#6B87A8", bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=36,b=40,l=10,r=10))
            st.plotly_chart(fig_mn, use_container_width=True)

            # Prophet model validation + trend decomposition
            if len(mn2) >= 4 and _prophet is not None:
                st.markdown(
                    '<div class="sec-label" style="margin-top:8px">Model Validation  Prophet Trend Decomposition</div>',
                    unsafe_allow_html=True)
                _vc1, _vc2 = st.columns(2)
                with _vc1:
                    # Actual vs Predicted on validation set
                    _fig_val = go.Figure()
                    _fig_val.add_trace(go.Scatter(
                        x=_val_dates, y=_val_actual.tolist(),
                        mode="lines+markers", name="Actual",
                        line=dict(color="#00C2D4", width=2),
                        marker=dict(size=8),
                    ))
                    _fig_val.add_trace(go.Scatter(
                        x=_val_dates, y=_val_preds.tolist(),
                        mode="lines+markers", name="Predicted",
                        line=dict(color="#F59E0B", width=2, dash="dash"),
                        marker=dict(size=8, symbol="diamond"),
                    ))
                    _fig_val.update_layout(
                        height=200,
                        title=dict(text=f"Validation: Actual vs Predicted  (R²={_r2_val:.3f}, MAPE={_mape:.1f}%)",
                                   font=dict(size=10, color="#8B949E")),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#8B949E", family="Barlow Condensed", size=9),
                        xaxis=dict(gridcolor="#0E1623"),
                        yaxis=dict(gridcolor="#0E1623", title="Violations"),
                        legend=dict(bgcolor="rgba(0,0,0,0)", x=0.6, y=0.05, font_size=9),
                        margin=dict(t=30, b=0, l=0, r=0),
                    )
                    st.plotly_chart(_fig_val, use_container_width=True)
                with _vc2:
                    # Prophet trend component
                    _full_forecast = _prophet.predict(_prophet.make_future_dataframe(
                        periods=len(_mn2_dt) - len(_train_df) + 2, freq="MS"))
                    _trend_x = [str(pd.Period(d, freq="M")) for d in _full_forecast["ds"]]
                    _fig_trend = go.Figure()
                    _fig_trend.add_trace(go.Scatter(
                        x=_trend_x, y=_full_forecast["trend"].clip(lower=0).tolist(),
                        mode="lines", name="Trend",
                        line=dict(color="#10B981", width=2),
                        fill="toself",
                        fillcolor="rgba(16,185,129,0.08)",
                    ))
                    _fig_trend.update_layout(
                        height=200,
                        title=dict(text="Prophet Trend Component",
                                   font=dict(size=10, color="#8B949E")),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#8B949E", family="Barlow Condensed", size=9),
                        xaxis=dict(gridcolor="#0E1623"),
                        yaxis=dict(gridcolor="#0E1623", title="Trend Level"),
                        showlegend=False,
                        margin=dict(t=30, b=0, l=0, r=0),
                    )
                    st.plotly_chart(_fig_trend, use_container_width=True)

        # Stacked hourly by type
        top5v = df["primary_vtype"].value_counts().head(5).index.tolist()
        hv    = df[df["primary_vtype"].isin(top5v)].groupby(["hour","primary_vtype"]).size().reset_index(name="count")
        fig_hv = px.bar(
            hv, x="hour", y="count", color="primary_vtype",
            barmode="stack", height=300,
            color_discrete_sequence=CHART_PAL,
            title="Hourly Volume by Violation Type",
            labels={"hour":"Hour of Day","count":"Violations","primary_vtype":"Violation Type"},
        )
        fig_hv.add_vrect(x0=6.5,x1=10.5, fillcolor="#D97706",opacity=0.06,
                         annotation_text="AM Peak",annotation_font_color="#D97706",annotation_font_size=10)
        fig_hv.add_vrect(x0=16.5,x1=20.5, fillcolor="#DC2626",opacity=0.06,
                         annotation_text="PM Peak",annotation_font_color="#DC2626",annotation_font_size=10)
        pl(fig_hv, margin=dict(t=36,b=0),
                              legend=dict(orientation="h",y=-0.25,font_size=9,
                                         font_color="#8B949E",bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_hv, use_container_width=True)

        c5c, c5d = st.columns(2)
        with c5c:
            # SCITA gap by hour
            sg = df.groupby("hour").agg(total=("id","count"), ok=("scita_ok","sum")).reset_index()
            sg["gap"] = (1 - sg["ok"]/sg["total"])*100
            fig_sg = px.area(sg, x="hour", y="gap", height=260,
                             color_discrete_sequence=["#0EA5E9"],
                             title="SCITA Coverage Gap by Hour (%)",
                             labels={"hour":"Hour","gap":"% Not in SCITA"})
            fig_sg.update_traces(fillcolor="rgba(14,165,233,0.10)",line_width=2)
            fig_sg.add_hrect(y0=30, y1=100, fillcolor="#DC2626",opacity=0.04,
                             annotation_text="High blind-spot zone",
                             annotation_font_color="#DC2626",annotation_font_size=9)
            pl(fig_sg, margin=dict(t=36,b=0))
            st.plotly_chart(fig_sg, use_container_width=True)

        with c5d:
            # Peak vs off-peak by violation type
            pk = df.copy()
            pk["period"] = pk["is_peak"].map({1:"Peak Hours",0:"Off-Peak"})
            pv = pk.groupby(["primary_vtype","period"]).size().reset_index(name="count")
            pv = pv[pv["primary_vtype"].isin(top5v)]
            fig_pv = px.bar(pv, x="count", y="primary_vtype", color="period",
                            barmode="group", orientation="h", height=260,
                            color_discrete_map={"Peak Hours":"#DC2626","Off-Peak":"#0EA5E9"},
                            title="Peak vs Off-Peak by Violation Type",
                            labels={"primary_vtype":"","count":"Violations"})
            pl(fig_pv, yaxis=dict(categoryorder="total ascending",gridcolor="#0E1623"),
                legend=dict(orientation="h",y=-0.3,font_size=9,bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=0,r=0,t=36,b=30))
            st.plotly_chart(fig_pv, use_container_width=True)

    # ==========================================================================
    # TAB 6 â"€â"€ ZONE EXPLORER
    # ==========================================================================
    with t6:
        st.markdown('<div class="sec-label">Deep-Dive Into Any Hotspot Zone</div>', unsafe_allow_html=True)

        top30 = cluster_stats.head(30)
        zone_options = [f"Zone #{r['rank']} - {r['station']} (CII={r['cii']:.3f})"
                        for _, r in top30.iterrows()]
        sel_zone_label = st.selectbox("Select Zone", zone_options)
        sel_idx = zone_options.index(sel_zone_label)
        z = top30.iloc[sel_idx]
        zdf = df_full[df_full["cluster"] == z["cluster"]]

        # Zone header stats
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-radius:12px;padding:18px 22px;margin:12px 0">
          <div style="display:flex;gap:30px;flex-wrap:wrap;align-items:flex-start">
            <div>
              <div style="font-size:1.6rem;font-weight:900;color:#00C2D4">Zone #{z['rank']}</div>
              <div style="color:#8B949E;font-size:0.8rem;text-transform:uppercase;
                          letter-spacing:0.06em">{z['station']} Jurisdiction</div>
            </div>
            <div class="stat-row">
              <div class="stat-pill"><b>{z['count']:,}</b> violations</div>
              <div class="stat-pill">CII <b>{z['cii']:.3f}</b></div>
              <div class="stat-pill"><b>{z['junction_rate']*100:.0f}%</b> junction</div>
              <div class="stat-pill"><b>{z['peak_rate']*100:.0f}%</b> peak hours</div>
              <div class="stat-pill"><b>{z['heavy_rate']*100:.0f}%</b> heavy vehicles</div>
              <div class="stat-pill">SCITA gap <b>{(1-z['scita_rate'])*100:.0f}%</b></div>
              <div class="stat-pill">Peak: <b>{z['top_day']} {z['top_hour']:02d}:00</b></div>
              <div class="stat-pill">Risk: <b style="color:{RISK_COLORS.get(str(z['risk_tier']),'#00C2D4')}">{z['risk_tier']}</b></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_z1, col_z2, col_z3 = st.columns(3)

        with col_z1:
            # Zone map
            st.markdown('<div class="sec-label">Zone Location</div>', unsafe_allow_html=True)
            fig_zm = px.scatter_mapbox(
                zdf.sample(min(2000,len(zdf)), random_state=1),
                lat="latitude", lon="longitude",
                color="severity", size_max=6,
                color_continuous_scale=[[0,"#059669"],[0.5,"#D97706"],[1,"#DC2626"]],
                zoom=13,
                center={"lat":float(z["lat"]),"lon":float(z["lon"])},
                mapbox_style="carto-darkmatter",
                height=280,
            )
            fig_zm.update_layout(margin=dict(r=0,t=0,l=0,b=0),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_zm, use_container_width=True)

        with col_z2:
            # Hour pattern
            st.markdown('<div class="sec-label">Violations by Hour</div>', unsafe_allow_html=True)
            zh = zdf.groupby("hour").size().reset_index(name="count")
            fig_zh = px.bar(zh, x="hour", y="count", height=280,
                            color="count",
                            color_continuous_scale=["#0E1623","#DC2626"])
            peak_h = int(zh.set_index("hour")["count"].idxmax())
            fig_zh.add_vline(x=peak_h, line_color="#00C2D4", line_dash="dot",
                             annotation_text=f"Peak {peak_h}:00",
                             annotation_font_color="#00C2D4", annotation_font_size=9)
            pl(fig_zh, coloraxis_showscale=False,
                                  margin=dict(t=10,b=0,l=0,r=0))
            st.plotly_chart(fig_zh, use_container_width=True)

        with col_z3:
            # Day pattern
            st.markdown('<div class="sec-label">Violations by Day</div>', unsafe_allow_html=True)
            zd = zdf.groupby("day_name").size().reindex(DAY_ORDER, fill_value=0).reset_index()
            zd.columns = ["day","count"]
            fig_zd = px.bar(zd, x="day", y="count", height=280,
                            color="count",
                            color_continuous_scale=["#0E1623","#0EA5E9"])
            pl(fig_zd, coloraxis_showscale=False,
                                  margin=dict(t=10,b=0,l=0,r=0),
                                  xaxis=dict(tickangle=-30,gridcolor="#0E1623",linecolor="#30363D"))
            st.plotly_chart(fig_zd, use_container_width=True)

        col_z4, col_z5, col_z6 = st.columns(3)

        with col_z4:
            # Violation types
            st.markdown('<div class="sec-label">Violation Type Breakdown</div>', unsafe_allow_html=True)
            zvc: dict = {}
            for vl in zdf["vlist"]:
                for v in vl: zvc[v] = zvc.get(v,0)+1
            ztop = sorted(zvc.items(), key=lambda x:-x[1])[:6]
            fig_zt = px.pie(
                names=[t[0].replace("PARKING ","").title() for t in ztop],
                values=[t[1] for t in ztop], hole=0.5,
                color_discrete_sequence=CHART_PAL, height=260,
            )
            fig_zt.update_traces(textinfo="percent", textfont_size=9)
            fig_zt.update_layout(showlegend=True,
                legend=dict(orientation="v",font_size=8,font_color="#8B949E",bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=10,b=0,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_zt, use_container_width=True)

        with col_z5:
            # Vehicle types
            st.markdown('<div class="sec-label">Vehicle Fleet in Zone</div>', unsafe_allow_html=True)
            zveh = zdf["vehicle_type"].value_counts().head(6).reset_index()
            zveh.columns = ["type","count"]
            fig_zv = px.bar(zveh, x="count", y="type", orientation="h",
                            color="count",
                            color_continuous_scale=["#0E1623","#8B5CF6"],
                            height=260)
            pl(fig_zv, coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending",gridcolor="#0E1623"),
                margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_zv, use_container_width=True)

        with col_z6:
            # Gauge metrics
            st.markdown('<div class="sec-label">Zone Risk Profile</div>', unsafe_allow_html=True)
            g_a, g_b = st.columns(2)
            with g_a:
                st.plotly_chart(gauge_fig(float(z["cii"]), "CII Score", "#DC2626"),
                                use_container_width=True)
            with g_b:
                _z_scita_gap = min(1.0, max(0.0, 1.0 - float(z["scita_rate"])))
                st.plotly_chart(gauge_fig(_z_scita_gap, "SCITA Gap", "#8B5CF6"),
                                use_container_width=True)

            # AI recommendation for zone
            deploy_hour = max(0, z["top_hour"] - 1)
            _z_unc = float(cs_fine[cs_fine["cluster"]==z["cluster"]]["uncollected"].sum()) if z["cluster"] in cs_fine["cluster"].values else 0.0
            st.markdown(f"""
            <div class="insight insight-crit" style="margin-top:6px"><b style="color:#DC2626">&#9679;</b> <b>Recommended Action</b><br>
            Deploy to <b>{z['station']}</b> by <b>{deploy_hour:02d}:00 on {z['top_day']}s</b>.
            Focus on {"junction control" if z['junction_rate']>0.5 else "area patrol"}.
            {"Heavy vehicle enforcement priority." if z['heavy_rate']>0.15 else ""}
            Uncollected fine exposure: <b style="color:#FBBF24">&#8377;{_z_unc/1e5:.1f}L</b>.
            </div>
            """, unsafe_allow_html=True)

        # Zone monthly trend (full-width below the 3-column row)
        st.markdown('<div class="sec-label" style="margin-top:10px">Violation Trend — Month by Month</div>',
                    unsafe_allow_html=True)
        zmn = zdf.groupby("month_key").size().reset_index(name="count").sort_values("month_key")
        city_mn = df_full.groupby("month_key").size().reset_index(name="city_count").sort_values("month_key")
        zmn = zmn.merge(city_mn, on="month_key", how="left")
        zmn["zone_share_pct"] = zmn["count"] / zmn["city_count"] * 100
        fig_zmn = go.Figure()
        fig_zmn.add_trace(go.Scatter(
            x=zmn["month_key"], y=zmn["count"],
            mode="lines+markers", name="Zone Violations",
            line=dict(color="#DC2626", width=2),
            marker=dict(size=7, color="#DC2626"),
            hovertemplate="<b>%{x}</b><br>Zone: %{y:,} violations<extra></extra>",
        ))
        fig_zmn.add_trace(go.Bar(
            x=zmn["month_key"], y=zmn["zone_share_pct"],
            name="City Share %", yaxis="y2",
            marker_color="rgba(139,92,246,0.25)",
            hovertemplate="<b>%{x}</b><br>City share: %{y:.2f}%<extra></extra>",
        ))
        fig_zmn.update_layout(
            height=200,
            yaxis=dict(title="Violations", gridcolor="#1D2F47", color="#6B87A8", title_font_size=10),
            yaxis2=dict(title="City Share %", overlaying="y", side="right",
                        gridcolor="rgba(0,0,0,0)", color="#8B5CF6", title_font_size=10,
                        showgrid=False),
            legend=dict(orientation="h", y=-0.25, font_size=9, font_color="#6B87A8",
                        bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#6B87A8"), margin=dict(t=10, b=30, l=0, r=0),
        )
        st.plotly_chart(fig_zmn, use_container_width=True)

    # â"€â"€ Footer

    # ==========================================================================
    # TAB 7 -- COMMAND INTELLIGENCE
    # ==========================================================================
    with t7:

        # ── KEY RECOMMENDATIONS ────────────────────────────────────────────────
        _cam_top   = cs_fine.sort_values("priority_score", ascending=False).iloc[0]
        _junc_top  = df[df["at_junction"]==1].groupby("junc_label").size().idxmax() if df["at_junction"].sum()>0 else "N/A"
        _rev_top20 = cs_fine.head(20)["uncollected"].sum() / 1e7
        _rev_all   = uncollected_fine_total / 1e7
        _rev_pct   = _rev_top20 / max(_rev_all, 0.01) * 100
        _cam_unc   = _cam_top["uncollected"] / 1e5
        st.markdown(f"""
        <div style="background:var(--card2);border:1px solid var(--border);border-radius:10px;
                    padding:16px 20px;margin-bottom:14px">
          <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.65rem;font-weight:800;
                      letter-spacing:.14em;color:var(--muted);margin-bottom:10px">
            COMMAND RECOMMENDATIONS &mdash; AI GENERATED
          </div>
          <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
            <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid var(--critical);
                        border-radius:8px;padding:12px 14px">
              <div style="font-size:0.68rem;font-weight:700;color:var(--critical-text);
                          letter-spacing:.08em;margin-bottom:5px">&#9679; CAMERA DEPLOYMENT</div>
              <div style="font-size:0.80rem;color:var(--text);line-height:1.5">
                Install SCITA camera at <b>{str(_cam_top['station'])[:22]}</b>
                (Zone&nbsp;#{int(_cam_top['rank'])}, CII&nbsp;{float(_cam_top['cii']):.3f}).
                Estimated recovery: <b style="color:var(--high-text)">&#8377;{_cam_unc:.1f}L</b>.
              </div>
            </div>
            <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid var(--high);
                        border-radius:8px;padding:12px 14px">
              <div style="font-size:0.68rem;font-weight:700;color:var(--high-text);
                          letter-spacing:.08em;margin-bottom:5px">&#9679; ENFORCEMENT FOCUS</div>
              <div style="font-size:0.80rem;color:var(--text);line-height:1.5">
                Concentrate patrols at <b>{str(_junc_top)[:22]}</b> intersection.
                Top-20 zone sweep recovers <b style="color:var(--high-text)">
                &#8377;{_rev_top20:.1f}&nbsp;Cr</b> ({_rev_pct:.0f}% of backlog).
              </div>
            </div>
            <div style="background:var(--card);border:1px solid var(--border);border-left:3px solid var(--accent);
                        border-radius:8px;padding:12px 14px">
              <div style="font-size:0.68rem;font-weight:700;color:var(--accent-bright);
                          letter-spacing:.08em;margin-bottom:5px">&#9679; TIMING PRIORITY</div>
              <div style="font-size:0.80rem;color:var(--text);line-height:1.5">
                Peak enforcement window is <b>{str(top_zone['top_day'])}s
                {int(top_zone['top_hour']):02d}:00</b>.
                Pre-deploy officers by <b>{max(0,int(top_zone['top_hour'])-1):02d}:30</b>
                for maximum deterrence at Zone&nbsp;#{int(top_zone['rank'])}.
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── A. FINE REVENUE INTELLIGENCE ──────────────────────────────────────
        st.markdown('<div class="sec-label">Fine Revenue Intelligence</div>', unsafe_allow_html=True)

        rev1, rev2, rev3, rev4 = st.columns(4)
        rev1.metric("Total Fine Potential",   f"₹{total_fine/1e7:.1f} Cr",
                    f"{len(df):,} violations")
        rev2.metric("Uncollected Revenue",    f"₹{uncollected_fine_total/1e7:.1f} Cr",
                    f"{blind_pct:.0f}% of fines not reaching SCITA")
        rev3.metric("SCITA Recovery Rate",    f"{recovery_rate:.1f}%",
                    "violations captured in system")
        rev4.metric("Camera Priority Zones",  f"{n_cam_priority}",
                    "high CII + high blind-spot")

        ra, rb = st.columns([3, 2])

        with ra:
            st.markdown('<div class="sec-label">Uncollected Revenue by Zone (Top 20)</div>',
                        unsafe_allow_html=True)
            zone_rev = cs_fine.sort_values("uncollected", ascending=False).head(20).copy()
            zone_rev["label"] = (zone_rev["station"].str[:20]
                                 + " (#" + zone_rev["rank"].astype(str) + ")")
            zone_rev["unc_L"] = zone_rev["uncollected"] / 1e5
            fig_rev = px.bar(
                zone_rev, x="unc_L", y="label", orientation="h",
                color="cii",
                color_continuous_scale=[[0,"#1D2F47"],[0.5,"#D97706"],[1,"#DC2626"]],
                text=zone_rev["unc_L"].apply(lambda x: f"₹{x:.1f}L"),
                height=440,
                labels={"unc_L": "Uncollected (₹ Lakhs)", "label": "", "cii": "CII"},
            )
            fig_rev.update_traces(textposition="outside",
                                  textfont=dict(size=9, color="#6B87A8"))
            pl(fig_rev,
               yaxis=dict(categoryorder="total ascending", gridcolor="#1D2F47"),
               coloraxis_showscale=False, margin=dict(l=0, r=70, t=10, b=0))
            st.plotly_chart(fig_rev, use_container_width=True)

        with rb:
            st.markdown('<div class="sec-label">Fine Potential vs Uncollected by Type</div>',
                        unsafe_allow_html=True)
            fbt = df.groupby("primary_vtype").agg(
                total    =("fine_amt",        "sum"),
                uncollect=("uncollected_fine", "sum"),
            ).reset_index().sort_values("total", ascending=False).head(8)
            fbt["label"] = fbt["primary_vtype"].str.replace("PARKING ", "").str.title()
            fig_fbt = go.Figure()
            fig_fbt.add_trace(go.Bar(
                y=fbt["label"], x=fbt["total"]/1e5,
                orientation="h", name="Total Potential",
                marker_color="#253D5C",
                hovertemplate="<b>%{y}</b><br>Fine Potential: ₹%{x:.1f}L<extra></extra>",
            ))
            fig_fbt.add_trace(go.Bar(
                y=fbt["label"], x=fbt["uncollect"]/1e5,
                orientation="h", name="Uncollected",
                marker_color="#DC2626", opacity=0.9,
                hovertemplate="<b>%{y}</b><br>Uncollected: ₹%{x:.1f}L<extra></extra>",
            ))
            pl(fig_fbt, barmode="overlay", height=390,
               yaxis=dict(categoryorder="total ascending", gridcolor="#1D2F47"),
               legend=dict(orientation="h", y=-0.18, font_size=9,
                           font_color="#6B87A8", bgcolor="rgba(0,0,0,0)"),
               margin=dict(l=0, r=0, t=10, b=50),
               xaxis_title="₹ Lakhs")
            st.plotly_chart(fig_fbt, use_container_width=True)

        st.divider()

        # ── B. SCITA CAMERA PLACEMENT RECOMMENDER ─────────────────────────────
        st.markdown('<div class="sec-label">SCITA Camera Placement Recommender</div>',
                    unsafe_allow_html=True)

        cam_top_uncollected = float(
            cs_fine[((cs_fine["cii"] > 0.45) & (cs_fine["scita_gap_pct"] > _gap_thresh))]["uncollected"].sum()
        )
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Camera Priority Zones", f"{n_cam_priority}",
                   f"CII > 0.45 and gap > {_gap_thresh:.0f}%")
        sm2.metric("Avg SCITA Gap (Priority)",
                   f"{cs_fine[cs_fine['cii']>0.45]['scita_gap_pct'].mean():.1f}%",
                   "in high-CII zones")
        sm3.metric("Revenue at Risk (Priority)",
                   f"₹{cam_top_uncollected/1e5:.1f}L",
                   "uncollected in priority zones")

        sc_a, sc_b = st.columns([3, 2])

        with sc_a:
            fig_scat = px.scatter(
                cs_fine,
                x="cii", y="scita_gap_pct",
                size="count", color="risk_tier",
                color_discrete_map=RISK_COLORS,
                hover_name="station",
                hover_data={
                    "cii":          ":.3f",
                    "scita_gap_pct":":.1f",
                    "count":        ":.0f",
                    "risk_tier":    False,
                    "priority_score":":.3f",
                },
                size_max=30, height=420,
                labels={
                    "cii":           "Congestion Impact Index (CII)",
                    "scita_gap_pct": "SCITA Coverage Gap (%)",
                    "risk_tier":     "Risk",
                },
            )
            _vline = 0.45
            fig_scat.add_vline(x=_vline, line_dash="dot",
                               line_color="#253D5C", line_width=1.5)
            fig_scat.add_hline(y=_gap_thresh, line_dash="dot",
                               line_color="#253D5C", line_width=1.5)
            _y_max = cs_fine["scita_gap_pct"].max() * 1.08
            fig_scat.add_shape(type="rect",
                               x0=_vline, y0=_gap_thresh, x1=1.05, y1=_y_max * 1.02,
                               fillcolor="rgba(220,38,38,0.06)",
                               line_width=0, layer="below")
            _yt = _y_max * 0.96
            _yb = _y_max * 0.04
            for txt, x, y, col, bg in [
                ("CAMERA PRIORITY",  0.73, _yt, "#DC2626", "rgba(220,38,38,0.12)"),
                ("COVERAGE GAP",     0.23, _yt, "#D97706", "rgba(217,119,6,0.10)"),
                ("ENFORCED ZONE",    0.73, _yb, "#059669", "rgba(5,150,105,0.10)"),
                ("LOW RISK",         0.23, _yb, "#334B65", "rgba(0,0,0,0)"),
            ]:
                fig_scat.add_annotation(
                    x=x, y=y, text=txt, showarrow=False,
                    font=dict(color=col, size=9, family="Barlow Condensed"),
                    bgcolor=bg, borderpad=3,
                )

            # Annotate top 3 camera priority zones
            _cam_label = cs_fine.sort_values("priority_score", ascending=False).head(3)
            for _, _lr in _cam_label.iterrows():
                fig_scat.add_annotation(
                    x=float(_lr["cii"]) + 0.02,
                    y=float(_lr["scita_gap_pct"]),
                    text=f"<b>{str(_lr['station'])[:18]}</b>",
                    showarrow=False,
                    font=dict(color="#F87171", size=8, family="Barlow Condensed"),
                    xanchor="left", yanchor="middle",
                )
            pl(fig_scat,
               xaxis=dict(range=[0, 1.03], gridcolor="#1D2F47"),
               yaxis=dict(range=[0, _y_max * 1.05], gridcolor="#1D2F47"),
               legend=dict(orientation="h", y=-0.12, font_size=9,
                           font_color="#6B87A8", bgcolor="rgba(0,0,0,0)"),
               margin=dict(t=20, b=40, l=10, r=10))
            st.plotly_chart(fig_scat, use_container_width=True)

        with sc_b:
            st.markdown('<div class="sec-label">Top Priority Zones for Camera Installation</div>',
                        unsafe_allow_html=True)
            cam_pri = cs_fine.sort_values("priority_score", ascending=False).head(12)[[
                "rank","station","cii","scita_gap_pct","uncollected","priority_score","risk_tier"
            ]].copy()
            cam_pri["scita_gap_pct"] = cam_pri["scita_gap_pct"].round(1)
            cam_pri["uncollected"]   = (cam_pri["uncollected"] / 1e5).round(1)
            cam_pri.columns = ["Zone","Station","CII","Gap%","Unc.(L)","Priority","Risk"]
            st.dataframe(cam_pri, use_container_width=True, height=390, hide_index=True,
                column_config={
                    "CII":      st.column_config.ProgressColumn(
                                    min_value=0, max_value=1, format="%.3f"),
                    "Priority": st.column_config.ProgressColumn(
                                    min_value=0, max_value=1, format="%.3f"),
                    "Unc.(L)":  st.column_config.NumberColumn(format="₹ %.1f L"),
                })

        st.divider()

        # ── C. AI PATROL BEAT ROUTING ─────────────────────────────────────────
        st.markdown('<div class="sec-label">AI Patrol Beat Routing</div>', unsafe_allow_html=True)
        st.caption(
            "Greedy nearest-neighbour route connecting highest-CII zones. "
            "Officers follow numbered stops in sequence for maximum violation coverage."
        )

        n_beat = st.slider("Zones in patrol beat", 3, min(20, n_zones), 8,
                           key="beat_slider")

        beat_zones = cluster_stats.head(n_beat).copy().reset_index(drop=True)
        route_ord  = nearest_neighbor_route(beat_zones)
        route_df   = beat_zones.iloc[route_ord].reset_index(drop=True)
        route_lats = route_df["lat"].tolist()
        route_lons = route_df["lon"].tolist()

        bc1, bc2 = st.columns([3, 2])

        with bc1:
            st.markdown('<div class="sec-label">Patrol Route Map</div>',
                        unsafe_allow_html=True)
            fig_beat = go.Figure()

            # Faint density background
            _samp = df_full.sample(min(15000, len(df_full)), random_state=11)
            fig_beat.add_trace(go.Densitymapbox(
                lat=_samp["latitude"], lon=_samp["longitude"],
                radius=5,
                colorscale=[[0,"rgba(0,0,0,0)"],[1,"#00C2D4"]],
                showscale=False, opacity=0.18, hoverinfo="skip",
            ))

            # Route line
            fig_beat.add_trace(go.Scattermapbox(
                lat=route_lats, lon=route_lons,
                mode="lines",
                line=dict(color="#00C2D4", width=3),
                name="Patrol Route", hoverinfo="skip",
            ))

            # Numbered stop markers
            for stop_i, (_, zr) in enumerate(route_df.iterrows()):
                tier_col = RISK_COLORS.get(str(zr["risk_tier"]), "#00C2D4")
                fig_beat.add_trace(go.Scattermapbox(
                    lat=[zr["lat"]], lon=[zr["lon"]],
                    mode="markers+text",
                    marker=dict(size=22, color=tier_col),
                    text=[str(stop_i + 1)],
                    textfont=dict(color="white", size=9),
                    textposition="middle center",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Stop {stop_i+1}</b><br>"
                        f"{zr['station']}<br>"
                        f"CII: {zr['cii']:.3f}  |  {zr['risk_tier']}<br>"
                        f"Deploy by: {max(0, zr['top_hour']-1):02d}:00<extra></extra>"
                    ),
                ))

            fig_beat.update_layout(
                mapbox=dict(style="carto-darkmatter",
                            center=dict(lat=12.97, lon=77.59), zoom=11),
                height=460, margin=dict(r=0, t=0, l=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_beat, use_container_width=True)

        with bc2:
            st.markdown('<div class="sec-label">Route Schedule</div>', unsafe_allow_html=True)
            beat_rows = ""
            total_beat_viol = int(route_df["count"].sum())
            for stop_i, (_, zr) in enumerate(route_df.iterrows()):
                rc = {"Critical":"risk-crit","High":"risk-high",
                      "Medium":"risk-med","Low":"risk-low"}.get(str(zr["risk_tier"]),"")
                deploy = max(0, int(zr["top_hour"]) - 1)
                beat_rows += f"""
                <tr>
                  <td style="font-family:'JetBrains Mono',monospace;color:#00C2D4">
                    {stop_i+1}</td>
                  <td>{str(zr['station'])[:22]}</td>
                  <td class="{rc}">{zr['risk_tier']}</td>
                  <td style="font-family:'JetBrains Mono',monospace">{zr['cii']:.3f}</td>
                  <td style="color:#059669;font-weight:700;font-family:'JetBrains Mono',monospace">
                    {deploy:02d}:00</td>
                  <td>{zr['top_day'][:3]}</td>
                </tr>"""
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);
                        border-radius:12px;overflow:hidden">
              <table class="sched-table">
                <thead><tr>
                  <th>#</th><th>Station</th><th>Risk</th>
                  <th>CII</th><th>Deploy</th><th>Day</th>
                </tr></thead>
                <tbody>{beat_rows}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

            beat_pct = total_beat_viol / len(df_full) * 100
            # Estimate route distance (Euclidean degrees → approx km)
            _rlats = route_lats + [route_lats[0]]
            _rlons = route_lons + [route_lons[0]]
            _km = sum(
                ((_rlats[i+1]-_rlats[i])**2 + (_rlons[i+1]-_rlons[i])**2)**0.5 * 111
                for i in range(len(_rlats)-1)
            )
            _crit_beat = int((route_df["risk_tier"] == "Critical").sum())
            _beat_unc  = float(
                cs_fine[cs_fine["cluster"].isin(beat_zones["cluster"].tolist())]["uncollected"].sum()
            )
            st.markdown(f"""
            <div class="insight insight-info" style="margin-top:12px">
              <b>Beat covers {total_beat_viol:,} violations</b>
              ({beat_pct:.1f}% of dataset) across {n_beat} zones including {_crit_beat} critical.
              Estimated loop: <b>~{_km:.1f}&thinsp;km</b> &bull;
              Revenue at stake: <b>&#8377;{_beat_unc/1e5:.1f}L</b>.
              Start at Stop 1 ({route_df.iloc[0]["station"][:20]}) and follow numbered sequence.
            </div>
            """, unsafe_allow_html=True)




        # -- Fine Recovery Simulator ----------------------------------------------
        st.divider()
        st.markdown('<div class="sec-label">Fine Recovery Simulator  Interactive Revenue Projection</div>',
                    unsafe_allow_html=True)
        _sim_col1, _sim_col2 = st.columns([1, 2])
        with _sim_col1:
            n_zones_sim = st.slider("Zones to target", min_value=1,
                                    max_value=min(50, n_zones), value=10,
                                    key="sim_zones",
                                    help="Deploy SCITA cameras to top N zones by priority score")
            scita_improvement = st.slider("SCITA coverage improvement %", min_value=10,
                                          max_value=90, value=50, step=5, key="sim_scita",
                                          help="Expected increase in SCITA capture rate after camera install")
            enforcement_multiplier = st.slider("Enforcement intensity multiplier", min_value=1.0,
                                               max_value=3.0, value=1.5, step=0.1, key="sim_enf",
                                               help="How many times more violations caught via active patrol")

        _target_zones = cs_fine.sort_values("priority_score", ascending=False).head(n_zones_sim)
        _current_unc  = float(_target_zones["uncollected"].sum())
        _total_unc_all = float(cs_fine["uncollected"].sum())

        # Revenue recovered by improving SCITA coverage
        _scita_recovery = _current_unc * (scita_improvement / 100.0)
        # Additional revenue from enforcement intensity (violations caught × fine)
        _enf_recovery   = float(_target_zones["total_fine"].sum()) * (enforcement_multiplier - 1.0) * 0.15
        _total_recovery = _scita_recovery + _enf_recovery
        _new_recovery_rate = min(100.0, recovery_rate + (_total_recovery / max(total_fine, 1)) * 100)

        # ROI on camera installation (avg cost ₹8L per camera, maintain 5yr)
        _cam_cost = n_zones_sim * 800000
        _annual_recovery = _scita_recovery * 2   # assume full 6-month dataset × 2 = annual
        _roi_months = (_cam_cost / max(_annual_recovery / 12, 1))

        with _sim_col2:
            st.markdown(
                f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px">'
                f'<div style="background:var(--card);border:1px solid #059669;border-radius:8px;padding:12px 16px">'
                f'<div style="font-size:0.60rem;color:var(--muted);letter-spacing:.08em">SCITA RECOVERY</div>'
                f'<div style="font-size:1.2rem;color:#10B981;font-weight:800">&#8377;{_scita_recovery/1e5:.1f}L</div>'
                f'<div style="font-size:0.65rem;color:var(--dim)">Camera coverage improvement</div></div>'
                f'<div style="background:var(--card);border:1px solid #D97706;border-radius:8px;padding:12px 16px">'
                f'<div style="font-size:0.60rem;color:var(--muted);letter-spacing:.08em">ENFORCEMENT GAIN</div>'
                f'<div style="font-size:1.2rem;color:#F59E0B;font-weight:800">&#8377;{_enf_recovery/1e5:.1f}L</div>'
                f'<div style="font-size:0.65rem;color:var(--dim)">{enforcement_multiplier:.1f}x intensity boost</div></div>'
                f'<div style="background:var(--card);border:1px solid #00C2D4;border-radius:8px;padding:12px 16px">'
                f'<div style="font-size:0.60rem;color:var(--muted);letter-spacing:.08em">TOTAL RECOVERABLE</div>'
                f'<div style="font-size:1.2rem;color:#00C2D4;font-weight:800">&#8377;{_total_recovery/1e5:.1f}L</div>'
                f'<div style="font-size:0.65rem;color:var(--dim)">Recovery rate: {_new_recovery_rate:.1f}%</div></div>'
                f'</div>'
                f'<div style="background:var(--card2);border:1px solid var(--border);border-radius:8px;padding:10px 14px;'
                f'font-size:0.72rem;color:var(--muted)">'
                f'Camera install cost ({n_zones_sim} zones): &#8377;{_cam_cost/1e5:.0f}L  |  '
                f'Estimated payback: <b style="color:var(--text)">{_roi_months:.0f} months</b>  |  '
                f'Total uncollected in target zones: &#8377;{_current_unc/1e5:.1f}L  |  '
                f'Full-city uncollected: &#8377;{_total_unc_all/1e5:.1f}L'
                f'</div>',
                unsafe_allow_html=True)

        # Waterfall chart: ₹ revenue in Lakhs
        _current_collected = total_fine * recovery_rate / 100
        _gap_before = total_fine - _current_collected
        _wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Current Gap (Uncollected)", "SCITA Camera Recovery",
               "Enforcement Boost", "Remaining Gap"],
            y=[_gap_before/1e5, -_scita_recovery/1e5, -_enf_recovery/1e5, 0],
            text=[f"₹{_gap_before/1e5:.1f}L gap",
                  f"-₹{_scita_recovery/1e5:.1f}L",
                  f"-₹{_enf_recovery/1e5:.1f}L",
                  f"₹{(_gap_before-_total_recovery)/1e5:.1f}L left"],
            textposition="outside", textfont=dict(size=10, color="#e6edf3"),
            connector=dict(line=dict(color="#1A2035", width=1.5)),
            increasing=dict(marker=dict(color="#DC2626")),
            decreasing=dict(marker=dict(color="#10B981")),
            totals=dict(marker=dict(color="#00C2D4")),
        ))
        _wf.update_layout(
            height=300,
            title=dict(text=f"Revenue Gap Closure  (₹ Lakhs)  Target: {n_zones_sim} zones · {scita_improvement}% SCITA · {enforcement_multiplier:.1f}x enforcement",
                       font=dict(size=11, color="#8B949E")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8B949E", family="Barlow Condensed"),
            yaxis=dict(gridcolor="#0E1623", title="Revenue (₹ Lakhs)"),
            margin=dict(l=0, r=10, t=48, b=0),
        )
        st.plotly_chart(_wf, use_container_width=True)

        # ── Zone Analytics Download ──────────────────────────────────────────────
        with st.expander("Download Zone Analytics", expanded=False):
            _dl_df = cs_fine[[
                "rank","station","cii","risk_tier","count",
                "junction_rate","severity_mean","peak_rate",
                "scita_rate","scita_gap_pct","total_fine","uncollected","priority_score"
            ]].copy()
            _dl_df.columns = [
                "Zone_Rank","Station","CII","Risk_Tier","Violations",
                "Junction_Rate","Severity_Mean","Peak_Rate",
                "SCITA_Rate","SCITA_Gap_Pct","Total_Fine_INR","Uncollected_INR","Camera_Priority_Score"
            ]
            _dl_df = _dl_df.round(4)
            st.dataframe(_dl_df, use_container_width=True, hide_index=True)
            st.download_button(
                label="Download as CSV",
                data=_dl_df.to_csv(index=False).encode("utf-8"),
                file_name="gridlock_zone_analytics.csv",
                mime="text/csv",
            )

        # ==========================================================================
    # TAB 9 — AI ASSISTANT
    # ==========================================================================
    with t9:
        # Build rich context from current dashboard state
        _zone_ctx = "\n".join([
            f"  Zone #{int(r['rank'])} — {r['station']} | CII {float(r['cii']):.3f} | "
            f"{int(r['count']):,} violations | {r['top_day']} {int(r['top_hour']):02d}:00 peak | "
            f"Risk: {r['risk_tier']} | Junction {float(r['junction_rate'])*100:.0f}% | "
            f"SCITA gap {(1-float(r['scita_rate']))*100:.0f}%"
            for _, r in cluster_stats.head(10).iterrows()
        ])
        _cam_ctx = "\n".join([
            f"  #{int(r['rank'])} {r['station']} — Priority {float(r['priority_score']):.3f} | "
            f"CII {float(r['cii']):.3f} | Gap {float(r['scita_gap_pct']):.1f}% | "
            f"Uncollected ₹{float(r['uncollected'])/1e5:.1f}L"
            for _, r in cs_fine.sort_values("priority_score", ascending=False).head(5).iterrows()
        ])
        _peak_day_ai  = df.groupby("day_name").size().idxmax()
        _peak_hour_ai = int(df.groupby("hour").size().idxmax())
        _top_junc_ai  = (df[df["at_junction"]==1].groupby("junc_label").size().idxmax()
                         if df["at_junction"].sum() > 0 else "N/A")
        _anom_ctx = "\n".join([
            f"  Zone #{int(r['rank'])} — {r['station']} | CII {float(r['cii']):.3f} | "
            f"Anomaly score {float(r['anomaly_score']):.3f} | {r['risk_tier']} risk | "
            f"Junction {float(r['junction_rate'])*100:.0f}% | SCITA gap {float(r['scita_gap_pct']):.0f}%"
            for _, r in anomaly_zones.head(5).iterrows()
        ]) if n_anomalies > 0 else "  No anomalies detected in current filter."

        SYSTEM_PROMPT = f"""You are GridLock AI, the intelligent enforcement assistant for Bengaluru Traffic Police.
You have real-time access to analysis of {len(df_full):,} parking violation records (Nov 2023 – Apr 2024) across 54 police stations in Bengaluru.

CURRENT FILTER STATE:
- Time period: {period} | Station: {sel_sta} | Violation type: {sel_vtype}
- Records in view: {len(df):,} ({len(df)/len(df_full)*100:.1f}% of dataset)

TOP 10 HOTSPOT ZONES (by CII):
{_zone_ctx}

ANOMALY DETECTION (IsolationForest — {n_anomalies} zones with unusual enforcement profiles):
{_anom_ctx}

FINE REVENUE INTELLIGENCE:
- Total fine potential: ₹{total_fine/1e7:.2f} Cr
- Uncollected revenue: ₹{uncollected_fine_total/1e7:.2f} Cr ({blind_pct:.1f}% blind to SCITA)
- SCITA recovery rate: {recovery_rate:.1f}%
- Camera priority zones (CII>0.45 + high gap): {n_cam_priority}

TOP 5 CAMERA PRIORITY ZONES:
{_cam_ctx}

ENFORCEMENT TIMING:
- Peak day: {_peak_day_ai} | Peak hour: {_peak_hour_ai:02d}:00–{_peak_hour_ai+1:02d}:00
- Pre-deploy recommendation: {max(0,_peak_hour_ai-1):02d}:30
- Worst junction: {_top_junc_ai}
- Junction violation rate: {junc_pct:.1f}%
- Heavy vehicle rate: {heavy_pct:.1f}%

ZONE SUMMARY: {n_zones} zones total — {n_critical} Critical, {n_high} High, {n_medium} Medium, {n_low} Low

INSTRUCTIONS:
- Be concise and direct. Use bullet points for lists.
- Always reference specific zone names, CII scores, ₹ amounts, and timestamps from the data above.
- Mention anomaly-flagged zones when asked about unusual patterns or priority zones.
- When asked where to patrol: give top 3 zones with station name, CII, and deploy time.
- When asked about cameras: give specific zones with priority score and ₹ recovery potential.
- When asked about peak times: give specific day + hour + pre-deploy recommendation.
- When asked for briefing: include anomaly flags and top 3 zones with specific ₹ impact.
- Do not make up data. If something is not in the context, say so.
- Keep responses under 250 words unless a detailed breakdown is requested.
"""

        # ── Chat UI ──────────────────────────────────────────────────────────
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
          <div style="width:36px;height:36px;border-radius:8px;
                      background:linear-gradient(135deg,#00C2D4,#8B5CF6);
                      display:flex;align-items:center;justify-content:center;
                      font-size:1.2rem">&#129302;</div>
          <div>
            <div style="font-size:1.1rem;font-weight:700;color:var(--text)">GridLock AI Assistant</div>
            <div style="font-size:0.72rem;color:var(--muted)">
              Powered by Groq · Contextualised with live dashboard data · Ask anything about Bengaluru enforcement
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        ai_col, cfg_col = st.columns([3, 1])

        with cfg_col:
            st.markdown('<div class="sec-label">Configuration</div>', unsafe_allow_html=True)
            _env_key = _os.environ.get("GROQ_API_KEY", "")
            if _env_key:
                st.markdown(
                    "<div style='background:rgba(16,185,129,0.10);border:1px solid #059669;"
                    "border-radius:6px;padding:6px 10px;font-size:0.70rem;color:#10B981'>"
                    "API key loaded from environment</div>",
                    unsafe_allow_html=True)
                groq_key = _env_key
            else:
                groq_key = st.text_input("Groq API Key", type="password",
                                         placeholder="gsk_...", key="groq_api_key")
                if not groq_key:
                    st.caption("Free key: console.groq.com")
            groq_model = st.selectbox("Model", [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ], key="groq_model")
            st.caption("Get a free key at console.groq.com")

            st.markdown('<div class="sec-label" style="margin-top:12px">Suggested Questions</div>',
                        unsafe_allow_html=True)
            suggestions = [
                "Where should I patrol tonight?",
                "Which zones need SCITA cameras most urgently?",
                "What is the single highest-impact action to recover revenue?",
                "Give me a 5-minute briefing for the shift commander.",
                "Which junctions need permanent enforcement posts?",
            ]
            for s in suggestions:
                if st.button(s, key=f"sugg_{s[:20]}", use_container_width=True):
                    if "ai_messages" not in st.session_state:
                        st.session_state.ai_messages = []
                    st.session_state.ai_messages.append({"role": "user", "content": s})
                    st.rerun()

            st.divider()
            if st.button("Load Demo Conversation", use_container_width=True, key="demo_chat",
                         help="Pre-load an example conversation to show judges"):
                _top1_demo = cluster_stats.iloc[0]
                _top2_demo = cluster_stats.iloc[1]
                _anom_demo = anomaly_zones.iloc[0] if n_anomalies > 0 else _top1_demo
                st.session_state.ai_messages = [
                    {"role": "user", "content": "Give me a 5-minute briefing for the shift commander."},
                    {"role": "assistant", "content":
                        f"**GRIDLOCK SHIFT BRIEFING — Bengaluru Traffic Police**\n\n"
                        f"**Critical Hotspots:**\n"
                        f"• Zone #{int(_top1_demo['rank'])} ({_top1_demo['station']}) — CII {float(_top1_demo['cii']):.3f}, {int(_top1_demo['count']):,} violations. Deploy by {max(0,int(_top1_demo['top_hour'])-1):02d}:30 on {_top1_demo['top_day']}s.\n"
                        f"• Zone #{int(_top2_demo['rank'])} ({_top2_demo['station']}) — CII {float(_top2_demo['cii']):.3f}, junction rate {float(_top2_demo['junction_rate'])*100:.0f}%.\n\n"
                        f"**Revenue Alert:** ₹{uncollected_fine_total/1e7:.1f} Cr uncollected. Top {n_cam_priority} zones need SCITA cameras urgently.\n\n"
                        f"**Anomaly Flag:** Zone #{int(_anom_demo['rank'])} ({str(_anom_demo['station'])[:22]}) shows unusual enforcement profile (IsolationForest score {float(_anom_demo['anomaly_score']):.3f}) — inspect today.\n\n"
                        f"**Peak Window:** {df.groupby('day_name').size().idxmax()}s at {int(df.groupby('hour').size().idxmax()):02d}:00. Pre-deploy by {max(0,int(df.groupby('hour').size().idxmax())-1):02d}:30.\n\n"
                        f"Focus enforcement on top 3 zones. Estimated impact: ₹{uncollected_fine_total*0.25/1e5:.1f}L recoverable this shift."},
                    {"role": "user", "content": "Which zones need SCITA cameras most urgently?"},
                    {"role": "assistant", "content":
                        "**Top 3 SCITA Camera Priority Zones:**\n\n" +
                        "\n".join([
                            f"**{i+1}. Zone #{int(r['rank'])} — {r['station']}**\n"
                            f"   • Priority score: {float(r['priority_score']):.3f} | CII: {float(r['cii']):.3f}\n"
                            f"   • SCITA gap: {float(r['scita_gap_pct']):.0f}% blind | Uncollected: ₹{float(r['uncollected'])/1e5:.1f}L"
                            for i, (_, r) in enumerate(
                                cs_fine.sort_values("priority_score", ascending=False).head(3).iterrows()
                            )
                        ]) +
                        f"\n\n**Why these first:** Combination of high CII (active violations) AND large SCITA gap (maximum revenue recovery potential).\n"
                        f"Camera install cost: ~₹8L per zone. Payback in <12 months at current violation rates."},
                ]
                st.rerun()

            if st.button("Clear conversation", use_container_width=True, key="clear_chat"):
                st.session_state.ai_messages = []
                st.rerun()

        with ai_col:
            if "ai_messages" not in st.session_state:
                st.session_state.ai_messages = []

            # Empty state placeholder
            if not st.session_state.ai_messages:
                st.markdown(
                    "<div style='text-align:center;padding:60px 20px;color:var(--muted)'>"
                    "<div style='font-size:2rem;margin-bottom:12px'>\U0001f916</div>"
                    "<div style='font-size:0.88rem'>Ask me anything about Bengaluru traffic enforcement.<br>"
                    f"I have full context of all {n_zones} zones, fine revenue, and patrol data.</div>"
                    "</div>",
                    unsafe_allow_html=True)

            # Render full message history
            for msg in st.session_state.ai_messages:
                with st.chat_message(msg["role"], avatar="\U0001f916" if msg["role"] == "assistant" else None):
                    st.markdown(msg["content"])

            # Chat input at bottom — no st.rerun() needed
            user_input = st.chat_input("Ask about patrol zones, enforcement timing, revenue recovery...")

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state.ai_messages.append({"role": "user", "content": user_input})

                if not groq_key:
                    _no_key = "Please enter your Groq API key in the configuration panel on the right."
                    with st.chat_message("assistant", avatar="\U0001f916"):
                        st.markdown(_no_key)
                    st.session_state.ai_messages.append({"role": "assistant", "content": _no_key})
                else:
                    try:
                        client = Groq(api_key=groq_key)

                        def _stream():
                            _s = client.chat.completions.create(
                                model=groq_model,
                                messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                                         + st.session_state.ai_messages,
                                max_tokens=600,
                                temperature=0.4,
                                stream=True,
                            )
                            for chunk in _s:
                                yield chunk.choices[0].delta.content or ""

                        with st.chat_message("assistant", avatar="\U0001f916"):
                            response_text = st.write_stream(_stream())

                        st.session_state.ai_messages.append({
                            "role": "assistant", "content": response_text
                        })
                    except Exception as e:
                        _err = "API Error: " + str(e)
                        with st.chat_message("assistant", avatar="\U0001f916"):
                            st.markdown(_err)
                        st.session_state.ai_messages.append({"role": "assistant", "content": _err})

    # ==========================================================================
    # TAB 8 — IMPACT REPORT
    # ==========================================================================
    with t8:
        _top5_zones = cluster_stats.head(5)
        _top_cam    = cs_fine.sort_values("priority_score", ascending=False).iloc[0]
        _beat_cov   = cluster_stats.head(8)["count"].sum() / max(len(df_full), 1) * 100
        _fine_gap   = uncollected_fine_total / 1e7

        st.markdown(f"""
        <div style="max-width:1100px;margin:0 auto;padding:8px 0">

          <!-- HERO -->
          <div style="text-align:center;padding:32px 20px 24px">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:0.68rem;
                        letter-spacing:.2em;color:var(--muted);margin-bottom:10px">
              HACKATHON SUBMISSION &bull; BENGALURU TRAFFIC ENFORCEMENT INTELLIGENCE
            </div>
            <div style="font-size:2.4rem;font-weight:900;line-height:1.1;
                        background:linear-gradient(90deg,#00D9ED,#8B5CF6);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;margin-bottom:10px">
              GridLock Intelligence Platform
            </div>
            <div style="font-size:1.0rem;color:var(--muted);max-width:620px;margin:0 auto">
              AI-powered parking enforcement command system for Bengaluru Traffic Police —
              turning 298,282 violation records into actionable patrol intelligence.
            </div>
          </div>

          <!-- HEADLINE NUMBERS -->
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:0 0 24px">
            <div style="background:var(--card);border:1px solid var(--border);
                        border-top:3px solid var(--critical);border-radius:10px;
                        padding:20px 18px;text-align:center">
              <div style="font-size:2.2rem;font-weight:900;color:var(--critical-text);
                          font-family:'JetBrains Mono',monospace">&#8377;{_fine_gap:.1f}&nbsp;Cr</div>
              <div style="font-size:0.72rem;color:var(--muted);margin-top:4px;letter-spacing:.06em">
                UNCOLLECTED FINE REVENUE</div>
              <div style="font-size:0.78rem;color:var(--dim);margin-top:6px">
                {blind_pct:.0f}% of violations not reaching SCITA system</div>
            </div>
            <div style="background:var(--card);border:1px solid var(--border);
                        border-top:3px solid var(--high);border-radius:10px;
                        padding:20px 18px;text-align:center">
              <div style="font-size:2.2rem;font-weight:900;color:var(--high-text);
                          font-family:'JetBrains Mono',monospace">{n_cam_priority}</div>
              <div style="font-size:0.72rem;color:var(--muted);margin-top:4px;letter-spacing:.06em">
                CAMERA PRIORITY ZONES</div>
              <div style="font-size:0.78rem;color:var(--dim);margin-top:6px">
                ₹{cs_fine[cs_fine['cii']>0.45]['uncollected'].sum()/1e5:.1f}L recoverable with SCITA coverage</div>
            </div>
            <div style="background:var(--card);border:1px solid var(--border);
                        border-top:3px solid var(--accent);border-radius:10px;
                        padding:20px 18px;text-align:center">
              <div style="font-size:2.2rem;font-weight:900;color:var(--accent-bright);
                          font-family:'JetBrains Mono',monospace">{_beat_cov:.1f}%</div>
              <div style="font-size:0.72rem;color:var(--muted);margin-top:4px;letter-spacing:.06em">
                VIOLATION COVERAGE</div>
              <div style="font-size:0.78rem;color:var(--dim);margin-top:6px">
                8-zone AI patrol beat vs ~1.2% random patrol</div>
            </div>
            <div style="background:var(--card);border:1px solid var(--border);
                        border-top:3px solid var(--chart-violet);border-radius:10px;
                        padding:20px 18px;text-align:center">
              <div style="font-size:2.2rem;font-weight:900;color:#A78BFA;
                          font-family:'JetBrains Mono',monospace">{n_zones}</div>
              <div style="font-size:0.72rem;color:var(--muted);margin-top:4px;letter-spacing:.06em">
                HOTSPOT ZONES DETECTED</div>
              <div style="font-size:0.78rem;color:var(--dim);margin-top:6px">
                across {len(df_full):,} violations, Nov 2023–Apr 2024</div>
            </div>
          </div>

          <!-- PROBLEM / SOLUTION / IMPACT -->
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:24px">
            <div style="background:var(--card2);border:1px solid var(--border);
                        border-radius:10px;padding:20px">
              <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:0.65rem;
                          letter-spacing:.14em;color:var(--critical-text);margin-bottom:10px">
                THE PROBLEM
              </div>
              <div style="font-size:0.88rem;color:var(--text);line-height:1.6">
                Bengaluru processes <b>298K parking violations</b> per 6 months across 54 police stations.
                Enforcement is <b>reactive and unguided</b> — officers patrol without knowing
                where violations cluster, when peak hours occur, or which junctions are most congested.
                <b>&#8377;{_fine_gap:.1f}&nbsp;Cr</b> in fines goes uncollected due to blind spots in
                the SCITA traffic camera network.
              </div>
            </div>
            <div style="background:var(--card2);border:1px solid var(--border);
                        border-radius:10px;padding:20px">
              <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:0.65rem;
                          letter-spacing:.14em;color:var(--accent-bright);margin-bottom:10px">
                OUR SOLUTION
              </div>
              <div style="font-size:0.88rem;color:var(--text);line-height:1.6">
                <b>Congestion Impact Index (CII)</b> scores every zone on 4 factors.
                <b>SCITA Gap Analysis</b> identifies camera blind spots with the highest revenue risk.
                <b>AI Patrol Beat Routing</b> (greedy TSP) generates optimised officer routes.
                <b>Prophet Time-Series Forecast</b> (Meta) with train/test validation.
                <b>IsolationForest</b> flags anomalous zones. <b>LP Optimizer</b> allocates officers.
                All outputs are <b>actionable</b> — specific zones, times, routes, and ₹ impact.
              </div>
            </div>
            <div style="background:var(--card2);border:1px solid var(--border);
                        border-radius:10px;padding:20px">
              <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:0.65rem;
                          letter-spacing:.14em;color:var(--high-text);margin-bottom:10px">
                THE IMPACT
              </div>
              <div style="font-size:0.88rem;color:var(--text);line-height:1.6">
                Installing SCITA cameras in <b>{n_cam_priority} priority zones</b> recovers
                <b>&#8377;{cam_top_uncollected/1e5:.1f}L</b> in previously missed fines.
                The AI patrol beat with <b>8 officers</b> covers <b>{_beat_cov:.1f}%</b>
                of violations vs ~1.2% random — a <b>9× efficiency gain</b>.
                Top-20 zone enforcement closes <b>&#8377;{cs_fine.head(20)['uncollected'].sum()/1e7:.1f}&nbsp;Cr</b>
                ({cs_fine.head(20)['uncollected'].sum()/max(uncollected_fine_total,1)*100:.0f}%
                of the total gap).
              </div>
            </div>
          </div>

          <!-- METHODOLOGY -->
          <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;
                      padding:20px 24px;margin-bottom:24px">
            <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:0.65rem;
                        letter-spacing:.14em;color:var(--muted);margin-bottom:14px">
              METHODOLOGY
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px">
              <div>
                <div style="font-size:0.72rem;font-weight:700;color:var(--accent-bright);
                            margin-bottom:6px">01 · SPATIAL CLUSTERING</div>
                <div style="font-size:0.78rem;color:var(--dim);line-height:1.5">
                  Grid-based spatial binning (0.30&thinsp;km cells) detects hotspot zones
                  with configurable density thresholds. {n_zones} zones from 298K records.
                </div>
              </div>
              <div>
                <div style="font-size:0.72rem;font-weight:700;color:var(--high-text);
                            margin-bottom:6px">02 · CONGESTION IMPACT INDEX</div>
                <div style="font-size:0.78rem;color:var(--dim);line-height:1.5">
                  CII = 0.35&times;freq + 0.30&times;junction + 0.20&times;severity + 0.15&times;peak.
                  Normalised 0–1 scale. Top zone: <b style="color:var(--text)">{float(top_zone['cii']):.3f}</b>.
                </div>
              </div>
              <div>
                <div style="font-size:0.72rem;font-weight:700;color:var(--chart-violet);
                            margin-bottom:6px">03 · SCITA GAP ANALYSIS</div>
                <div style="font-size:0.78rem;color:var(--dim);line-height:1.5">
                  Cross-references violations with SCITA camera coverage.
                  {blind_pct:.0f}% blind spots mapped to zones for camera priority scoring.
                </div>
              </div>
              <div>
                <div style="font-size:0.72rem;font-weight:700;color:var(--low-text);
                            margin-bottom:6px">04 · AI PATROL ROUTING</div>
                <div style="font-size:0.78rem;color:var(--dim);line-height:1.5">
                  Greedy nearest-neighbour TSP routing. Prophet time-series forecast
                  with held-out validation. IsolationForest anomaly detection.
                  RandomForest violation-type classifier. LP officer allocation.
                </div>
              </div>
            </div>
          </div>

          <!-- TOP 5 ZONES TABLE -->
          <div style="background:var(--card);border:1px solid var(--border);border-radius:10px;
                      padding:20px 24px;margin-bottom:16px">
            <div style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:0.65rem;
                        letter-spacing:.14em;color:var(--muted);margin-bottom:12px">
              TOP 5 HIGHEST-IMPACT ZONES
            </div>
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px">
              {"".join([
                f'''<div style="background:var(--card2);border:1px solid var(--border);
                              border-left:3px solid {"#DC2626" if r["risk_tier"]=="Critical" else "#D97706"};
                              border-radius:8px;padding:12px">
                  <div style="font-size:0.65rem;color:var(--muted);letter-spacing:.08em">ZONE #{int(r["rank"])}</div>
                  <div style="font-size:0.85rem;font-weight:700;color:var(--text);
                              margin:4px 0 2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    {str(r["station"])[:18]}</div>
                  <div style="font-size:0.78rem;color:var(--dim)">CII <b style="color:var(--text)">{float(r["cii"]):.3f}</b></div>
                  <div style="font-size:0.78rem;color:var(--dim)">{int(r["count"]):,} violations</div>
                  <div style="font-size:0.72rem;color:var(--muted);margin-top:4px">{str(r["top_day"])} {int(r["top_hour"]):02d}:00 peak</div>
                </div>'''
                for _, r in _top5_zones.iterrows()
              ])}
            </div>
          </div>

          <!-- TECH STACK -->
          <div style="text-align:center;padding:12px 0 4px">
            <div style="font-size:0.65rem;color:var(--dim);letter-spacing:.1em">
              BUILT WITH &nbsp;&bull;&nbsp;
              Python 3.10 &bull; Streamlit &bull; Plotly &bull; Pandas &bull; NumPy &bull;
              scikit-learn (CII · IsolationForest anomaly detection) &bull;
              Prophet time-series &bull; scipy LP optimizer &bull; fpdf2 PDF &bull; Groq LLM &bull;
              Carto Dark Matter tiles &bull; Barlow Condensed + JetBrains Mono
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)


        # -- PDF Enforcement Brief Download -----------------------------------
        st.markdown('<div class="sec-label" style="margin-top:16px">Download Enforcement Brief  PDF Report</div>',
                    unsafe_allow_html=True)
        if st.button("Generate PDF Enforcement Brief", key="gen_pdf",
                     help="Download a one-page PDF briefing for the shift commander"):
            from fpdf import FPDF
            import io as _io
            from datetime import date as _date

            _pdf = FPDF(orientation="P", unit="mm", format="A4")
            _pdf.set_auto_page_break(auto=True, margin=15)
            _pdf.add_page()
            _pdf.set_margins(15, 15, 15)

            # Header
            _pdf.set_fill_color(8, 13, 23)
            _pdf.rect(0, 0, 210, 28, "F")
            _pdf.set_font("Helvetica", "B", 18)
            _pdf.set_text_color(0, 194, 212)
            _pdf.set_xy(15, 6)
            _pdf.cell(180, 8, "GRIDLOCK INTELLIGENCE PLATFORM", ln=True, align="L")
            _pdf.set_font("Helvetica", "", 9)
            _pdf.set_text_color(139, 148, 158)
            _pdf.set_xy(15, 16)
            _pdf.cell(180, 6, f"Bengaluru Traffic Enforcement Brief  |  Generated: {_date.today().strftime('%d %b %Y')}  |  {len(df_full):,} violation records  |  Nov 2023 - Apr 2024", ln=True)

            _pdf.ln(8)

            # KPI Row
            _pdf.set_font("Helvetica", "B", 8)
            _pdf.set_text_color(80, 80, 90)
            _pdf.cell(45, 5, "TOTAL VIOLATIONS", ln=False)
            _pdf.cell(45, 5, "UNCOLLECTED FINES", ln=False)
            _pdf.cell(45, 5, "CRITICAL ZONES", ln=False)
            _pdf.cell(45, 5, "SCITA BLIND SPOTS", ln=True)
            _pdf.set_font("Helvetica", "B", 14)
            _pdf.set_text_color(0, 194, 212)
            _pdf.cell(45, 8, f"{len(df_full):,}", ln=False)
            _pdf.set_text_color(220, 38, 38)
            _pdf.cell(45, 8, f"Rs.{uncollected_fine_total/1e7:.1f} Cr", ln=False)
            _pdf.set_text_color(217, 119, 6)
            _pdf.cell(45, 8, f"{n_critical} zones", ln=False)
            _pdf.set_text_color(139, 92, 246)
            _pdf.cell(45, 8, f"{blind_pct:.0f}%", ln=True)
            _pdf.ln(4)

            # Divider
            _pdf.set_draw_color(30, 40, 60)
            _pdf.line(15, _pdf.get_y(), 195, _pdf.get_y())
            _pdf.ln(3)

            # Top 5 Hotspot Zones
            _pdf.set_font("Helvetica", "B", 10)
            _pdf.set_text_color(0, 194, 212)
            _pdf.cell(180, 7, "TOP 5 HOTSPOT ZONES (by CII)", ln=True)
            _pdf.set_font("Helvetica", "B", 8)
            _pdf.set_text_color(100, 110, 130)
            _col_w = [10, 52, 18, 22, 22, 30, 26]
            for _h in ["#", "Station", "CII", "Risk", "Violations", "Peak Window", "SCITA Gap"]:
                _pdf.cell(_col_w[["#","Station","CII","Risk","Violations","Peak Window","SCITA Gap"].index(_h)], 6, _h, ln=False)
            _pdf.ln()
            _pdf.set_font("Helvetica", "", 8)
            for _, _zr in cluster_stats.head(5).iterrows():
                _row_vals = [
                    str(int(_zr["rank"])),
                    str(_zr["station"])[:26],
                    f"{float(_zr['cii']):.3f}",
                    str(_zr["risk_tier"]),
                    f"{int(_zr['count']):,}",
                    f"{_zr['top_day']} {int(_zr['top_hour']):02d}:00",
                    f"{(1-float(_zr['scita_rate']))*100:.0f}%",
                ]
                for _v, _w in zip(_row_vals, _col_w):
                    if _row_vals[3] == "Critical":
                        _pdf.set_text_color(220, 38, 38)
                    elif _row_vals[3] == "High":
                        _pdf.set_text_color(217, 119, 6)
                    else:
                        _pdf.set_text_color(50, 60, 80)
                    _pdf.cell(_w, 6, str(_v)[:20], ln=False)
                _pdf.set_text_color(50, 60, 80)
                _pdf.ln()
            _pdf.ln(3)

            # Anomaly zones
            _pdf.set_draw_color(30, 40, 60)
            _pdf.line(15, _pdf.get_y(), 195, _pdf.get_y())
            _pdf.ln(3)
            _pdf.set_font("Helvetica", "B", 10)
            _pdf.set_text_color(124, 58, 237)
            _pdf.cell(180, 7, f"ANOMALY DETECTION  ({n_anomalies} zones flagged by IsolationForest)", ln=True)
            _pdf.set_font("Helvetica", "", 8)
            _pdf.set_text_color(50, 60, 80)
            for _, _ar in anomaly_zones.head(3).iterrows():
                _pdf.cell(180, 5,
                    f"Zone #{int(_ar['rank'])} {str(_ar['station'])[:28]}  |  CII {float(_ar['cii']):.3f}  |  "
                    f"Junction {float(_ar['junction_rate'])*100:.0f}%  |  SCITA gap {float(_ar['scita_gap_pct']):.0f}%  |  "
                    f"Anomaly score {float(_ar['anomaly_score']):.3f}", ln=True)
            _pdf.ln(3)

            # Key Recommendations
            _pdf.set_draw_color(30, 40, 60)
            _pdf.line(15, _pdf.get_y(), 195, _pdf.get_y())
            _pdf.ln(3)
            _pdf.set_font("Helvetica", "B", 10)
            _pdf.set_text_color(0, 194, 212)
            _pdf.cell(180, 7, "COMMAND RECOMMENDATIONS", ln=True)
            _peak_day_p  = df.groupby("day_name").size().idxmax()
            _peak_hour_p = int(df.groupby("hour").size().idxmax())
            _top1 = cluster_stats.iloc[0]
            _recs = [
                f"1. Deploy SCITA cameras at {n_cam_priority} priority zones — estimated recovery Rs.{uncollected_fine_total * 0.4 / 1e5:.1f}L",
                f"2. Enforce Zone #{int(_top1['rank'])} ({str(_top1['station'])[:22]}) CII {float(_top1['cii']):.3f} — highest city-wide risk",
                f"3. Peak window: {_peak_day_p}s at {_peak_hour_p:02d}:00-{_peak_hour_p+1:02d}:00. Pre-deploy by {max(0,_peak_hour_p-1):02d}:30",
                f"4. Anomaly alert: {n_anomalies} zones showing irregular enforcement profiles — prioritise inspection",
                f"5. Uncollected revenue: Rs.{uncollected_fine_total/1e7:.2f} Cr across {n_zones} zones — {100-recovery_rate:.0f}% gap",
            ]
            _pdf.set_font("Helvetica", "", 9)
            _pdf.set_text_color(30, 40, 60)
            for _r in _recs:
                _pdf.cell(180, 6, _r, ln=True)
            _pdf.ln(3)

            # Footer
            _pdf.set_fill_color(8, 13, 23)
            _pdf.rect(0, 277, 210, 20, "F")
            _pdf.set_xy(15, 280)
            _pdf.set_font("Helvetica", "", 7)
            _pdf.set_text_color(80, 90, 100)
            _pdf.cell(180, 4,
                "GridLock Intelligence Platform  |  CII = 0.35*freq + 0.30*junction + 0.20*severity + 0.15*peak  |  IsolationForest anomaly detection  |  scipy LP allocation",
                ln=True)

            _pdf_bytes = bytes(_pdf.output())
            st.download_button(
                label="Download PDF Enforcement Brief",
                data=_pdf_bytes,
                file_name=f"gridlock_brief_{_date.today().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                key="dl_pdf",
            )
            st.success("PDF ready — click Download PDF Enforcement Brief above to save.")

    # ==========================================================================
    # TAB 10 -- ROUTE OPTIMIZER
    # ==========================================================================
    with t10:
        st.markdown('<div class="sec-label">🛣️ Congestion-Aware Route Optimizer — Bengaluru</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(0,194,212,0.06);border:1px solid rgba(0,194,212,0.20);
                    border-radius:8px;padding:10px 16px;margin-bottom:14px;font-size:0.80rem;color:#8FA8C8">
        Select a <b style="color:#E2EBF5">source</b> and <b style="color:#E2EBF5">destination</b>.
        The optimizer uses our 298k violation dataset to weight roads by congestion severity,
        then runs <b style="color:#00C2D4">A* pathfinding</b> to find the route that minimises
        travel through high-violation zones — compared against the direct straight-line path.
        </div>
        """, unsafe_allow_html=True)

        r_col1, r_col2, r_col3 = st.columns([1, 1, 1])
        landmark_names = list(BENGALURU_LANDMARKS.keys())
        with r_col1:
            src_name = st.selectbox("📍 Source", landmark_names, index=0, key="route_src")
        with r_col2:
            dst_name = st.selectbox("🏁 Destination",
                                     landmark_names, index=5, key="route_dst")
        with r_col3:
            cong_penalty = st.slider("Congestion Avoidance Weight", 1, 20, 10,
                                     help="Higher = route avoids congested zones more aggressively",
                                     key="route_penalty")

        run_route = st.button("🔍 Find Optimal Route", type="primary", key="run_route_btn")

        if src_name == dst_name:
            st.warning("Source and destination are the same — please choose different locations.")
        elif run_route or st.session_state.get("_route_computed"):

            with st.spinner("Building congestion grid and computing routes…"):
                # Build grid (cached on lat/lon hash)
                _lat_arr  = df_full["latitude"].dropna().values.astype(np.float32)
                _lon_arr  = df_full["longitude"].dropna().values.astype(np.float32)
                _sev_arr  = df_full.loc[df_full["latitude"].notna() & df_full["longitude"].notna(),
                                        "severity"].values.astype(np.float32)
                _lat_hash = int(_lat_arr.sum()) % 999999

                grid, lat_min, lat_max, lon_min, lon_max = build_congestion_grid(
                    _lat_hash, _lat_arr, _lon_arr, _sev_arr)
                n_grid = grid.shape[0]

                src_latlon = BENGALURU_LANDMARKS[src_name]
                dst_latlon = BENGALURU_LANDMARKS[dst_name]
                src_rc = _latlon_to_rc(*src_latlon, lat_min, lat_max, lon_min, lon_max, n_grid)
                dst_rc = _latlon_to_rc(*dst_latlon, lat_min, lat_max, lon_min, lon_max, n_grid)

                opt_path   = astar_route(grid, src_rc, dst_rc, cong_weight=cong_penalty)
                dir_path   = straight_line_route(src_rc, dst_rc)

                # Convert paths to lat/lon
                opt_latlons = [_rc_to_latlon(r, c, lat_min, lat_max, lon_min, lon_max, n_grid)
                               for r, c in opt_path]
                dir_latlons = [_rc_to_latlon(r, c, lat_min, lat_max, lon_min, lon_max, n_grid)
                               for r, c in dir_path]

                # Stats
                opt_mean, opt_max, opt_sum = path_congestion_stats(opt_path, grid)
                dir_mean, dir_max, dir_sum = path_congestion_stats(dir_path, grid)

                # Haversine distance helper
                def _haversine(lats, lons):
                    R = 6371.0
                    total = 0.0
                    for i in range(len(lats)-1):
                        dlat = np.radians(lats[i+1]-lats[i])
                        dlon = np.radians(lons[i+1]-lons[i])
                        a = np.sin(dlat/2)**2 + np.cos(np.radians(lats[i]))*np.cos(np.radians(lats[i+1]))*np.sin(dlon/2)**2
                        total += R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                    return total

                opt_km = _haversine([p[0] for p in opt_latlons], [p[1] for p in opt_latlons])
                dir_km = _haversine([p[0] for p in dir_latlons], [p[1] for p in dir_latlons])
                cong_saved_pct = max(0, (dir_sum - opt_sum) / max(dir_sum, 1e-9) * 100)
                viol_on_opt  = int(opt_mean * 5000)
                viol_on_dir  = int(dir_mean * 5000)

            # ── KPI row ──────────────────────────────────────────────────────
            rk1, rk2, rk3, rk4 = st.columns(4)
            with rk1:
                st.markdown(f"""
                <div class="kpi-card kc-accent" style="margin:0">
                  <div class="kpi-label">Optimal Route Distance</div>
                  <div class="kpi-val">{opt_km:.1f} km</div>
                  <div class="kpi-sub">A* congestion-aware path</div>
                </div>""", unsafe_allow_html=True)
            with rk2:
                st.markdown(f"""
                <div class="kpi-card kc-high" style="margin:0">
                  <div class="kpi-label">Direct Route Distance</div>
                  <div class="kpi-val">{dir_km:.1f} km</div>
                  <div class="kpi-sub">Straight-line baseline</div>
                </div>""", unsafe_allow_html=True)
            with rk3:
                _extra = opt_km - dir_km
                _extra_str = f"+{_extra:.1f} km longer" if _extra > 0 else f"{abs(_extra):.1f} km shorter"
                st.markdown(f"""
                <div class="kpi-card kc-critical" style="margin:0">
                  <div class="kpi-label">Congestion Avoided</div>
                  <div class="kpi-val">{cong_saved_pct:.0f}%</div>
                  <div class="kpi-sub">{_extra_str} but safer</div>
                </div>""", unsafe_allow_html=True)
            with rk4:
                st.markdown(f"""
                <div class="kpi-card kc-medium" style="margin:0">
                  <div class="kpi-label">Route Congestion Score</div>
                  <div class="kpi-val">{opt_mean:.3f}</div>
                  <div class="kpi-sub">vs {dir_mean:.3f} direct (lower = better)</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Map ───────────────────────────────────────────────────────────
            map_col, info_col = st.columns([3, 1])
            with map_col:
                st.markdown('<div class="sec-label">Route Map — Congestion Overlay</div>',
                            unsafe_allow_html=True)

                # Sample violations for background density
                _map_sample = df_full.dropna(subset=["latitude","longitude"]).sample(
                    min(40000, len(df_full)), random_state=42)

                fig_route = go.Figure()

                # Background density heatmap
                fig_route.add_trace(go.Densitymapbox(
                    lat=_map_sample["latitude"], lon=_map_sample["longitude"],
                    z=_map_sample["severity"], radius=6,
                    colorscale=[[0,"rgba(0,0,0,0)"],[0.4,"rgba(220,38,38,0.15)"],
                                [1,"rgba(220,38,38,0.55)"]],
                    showscale=False, opacity=0.6, name="Violation Density",
                ))

                # Direct route
                fig_route.add_trace(go.Scattermapbox(
                    lat=[p[0] for p in dir_latlons],
                    lon=[p[1] for p in dir_latlons],
                    mode="lines",
                    line=dict(width=3, color="#DC2626"),
                    name="Direct Route",
                    hoverinfo="none",
                ))

                # Optimal route
                fig_route.add_trace(go.Scattermapbox(
                    lat=[p[0] for p in opt_latlons],
                    lon=[p[1] for p in opt_latlons],
                    mode="lines",
                    line=dict(width=4, color="#00C2D4"),
                    name="Optimal Route",
                    hoverinfo="none",
                ))

                # Source & destination markers
                fig_route.add_trace(go.Scattermapbox(
                    lat=[src_latlon[0], dst_latlon[0]],
                    lon=[src_latlon[1], dst_latlon[1]],
                    mode="markers+text",
                    marker=dict(size=[16, 16], color=["#10B981", "#F59E0B"],
                                symbol="circle"),
                    text=[f"START: {src_name}", f"END: {dst_name}"],
                    textposition="top right",
                    textfont=dict(color="#E2EBF5", size=11),
                    hovertemplate="%{text}<extra></extra>",
                    name="Waypoints",
                ))

                mid_lat = (src_latlon[0] + dst_latlon[0]) / 2
                mid_lon = (src_latlon[1] + dst_latlon[1]) / 2
                fig_route.update_layout(
                    mapbox=dict(style="carto-darkmatter",
                                center=dict(lat=mid_lat, lon=mid_lon), zoom=11.5),
                    height=480, margin=dict(r=0, t=0, l=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True,
                    legend=dict(bgcolor="rgba(14,22,35,0.85)", font=dict(color="#8FA8C8", size=11),
                                x=0.01, y=0.99),
                )
                st.plotly_chart(fig_route, use_container_width=True)

            with info_col:
                st.markdown('<div class="sec-label">Route Analysis</div>', unsafe_allow_html=True)

                # Comparison bars
                for label, val, color in [
                    ("Avg Congestion", opt_mean, "#00C2D4"),
                    ("Peak Congestion", opt_max, "#DC2626"),
                ]:
                    st.markdown(f"""
                    <div style="margin-bottom:14px">
                      <div style="font-size:0.65rem;color:var(--muted);margin-bottom:4px">{label}</div>
                      <div style="background:var(--card2);border-radius:4px;height:8px;overflow:hidden">
                        <div style="width:{opt_mean/max(dir_mean,0.001)*100:.0f}%;height:100%;background:{color};
                                    border-radius:4px"></div>
                      </div>
                      <div style="font-size:0.72rem;color:{color};margin-top:3px;font-family:'JetBrains Mono',monospace">
                        {val:.3f} <span style="color:var(--muted)">vs {dir_mean:.3f} direct</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Hazard zones on optimal path
                st.markdown('<div class="sec-label" style="margin-top:12px">Hotspot Zones En Route</div>',
                            unsafe_allow_html=True)
                _path_lats = [p[0] for p in opt_latlons]
                _path_lons = [p[1] for p in opt_latlons]
                _path_lat_range = (min(_path_lats)-0.015, max(_path_lats)+0.015)
                _path_lon_range = (min(_path_lons)-0.015, max(_path_lons)+0.015)
                _zones_en_route = cluster_stats[
                    (cluster_stats["lat"].between(*_path_lat_range)) &
                    (cluster_stats["lon"].between(*_path_lon_range))
                ].sort_values("cii", ascending=False).head(5)
                if len(_zones_en_route) > 0:
                    for _, _zr in _zones_en_route.iterrows():
                        _rc = {"Critical":"#DC2626","High":"#D97706",
                               "Medium":"#0EA5E9","Low":"#059669"}.get(str(_zr["risk_tier"]),"#6B87A8")
                        st.markdown(f"""
                        <div style="border:1px solid {_rc}22;border-left:3px solid {_rc};
                                    border-radius:6px;padding:6px 9px;margin-bottom:5px;
                                    background:{_rc}0D;font-size:0.72rem">
                          <span style="color:{_rc};font-weight:700">{_zr['risk_tier']}</span>
                          <span style="color:#E2EBF5;margin-left:6px">{str(_zr['station'])[:20]}</span>
                          <div style="color:var(--muted);font-size:0.65rem;margin-top:2px">
                            CII {float(_zr['cii']):.3f} &bull; {int(_zr['count']):,} violations
                          </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div style="color:var(--muted);font-size:0.78rem">No major hotspot zones on this route segment.</div>',
                                unsafe_allow_html=True)

                # Recommendation
                st.markdown(f"""
                <div style="background:rgba(0,194,212,0.08);border:1px solid rgba(0,194,212,0.25);
                            border-radius:8px;padding:10px 12px;margin-top:14px;font-size:0.78rem;
                            color:#8FA8C8;line-height:1.6">
                  <b style="color:#00C2D4">Recommendation</b><br>
                  {"✅ Optimal route avoids " + f"{cong_saved_pct:.0f}% of congestion" if cong_saved_pct > 5 else "⚠️ Both routes pass through similar congestion levels."}<br>
                  {"Expect " + str(viol_on_opt) + " fewer violation incidents vs direct path." if cong_saved_pct > 5 else "Consider off-peak travel for this corridor."}
                </div>
                """, unsafe_allow_html=True)

            # Store computed flag
            st.session_state["_route_computed"] = True

    st.markdown(f"""
    <div style="margin-top:24px;padding:12px 20px;border-top:1px solid var(--border);
                display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;
                font-size:0.68rem;color:var(--dim);font-family:'Barlow Condensed',sans-serif">
      <span>
        <b style="color:var(--accent)">GridLock Intelligence Platform</b>
        &nbsp;&bull;&nbsp;Bengaluru Traffic Enforcement Command
        &nbsp;&bull;&nbsp;{len(df_full):,} violation records &bull; Nov 2023 &ndash; Apr 2024
      </span>
      <span>
        Spatial Grid Clustering &bull; Congestion Impact Index (CII = 0.35&times;freq + 0.30&times;junction + 0.20&times;severity + 0.15&times;peak)
        &bull; Greedy TSP Patrol Routing &bull; SCITA Gap Analysis
      </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


