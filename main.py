# Careers@Gov Job Matcher — Streamlit app (Python/Streamlit)
# -----------------------------------------------------------
# Features
# 1) Scrape & clean job postings from Careers@Gov (Selenium, pagination)
# 2) Upload a CV (PDF/DOCX/TXT), extract skills & experience with GPT-4o-mini
# 3) Create sentence embeddings with OpenAI embeddings (text-embedding-3-small)
# 4) Cosine similarity matching
# 5) Streamlit UI to browse top matches & skill overlaps
#
# IMPORTANT:
# - Set environment variable OPENAI_API_KEY before running.
# - Scraping public websites can be subject to Terms of Use/robots.txt. Use responsibly.
# - If scraping is blocked or flaky, use the "Use sample dataset" toggle in the sidebar.
#
# Quick start
# ----------
# pip install -U streamlit selenium webdriver-manager beautifulsoup4 requests pandas numpy openai pdfplumber docx2txt python-dotenv
# streamlit run app.py

from __future__ import annotations
import os
import time
import io
import re
import json
import math
import tempfile
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import streamlit as st
# --- Local vector store (Chroma) ---
# --- Local vector store (Chroma) ---
try:
    import chromadb as chroma_mod  # don't shadow with the same name
    CHROMA_OK = True
except Exception:
    chroma_mod = None
    CHROMA_OK = False

# Settings is optional (older Chroma API); don't fail if missing
try:
    from chromadb.config import Settings as ChromaSettings  # optional
except Exception:
    ChromaSettings = None

# --- Optional file parsers ---
try:
    import pdfplumber
except Exception:
    pdfplumber = None
try:
    import docx2txt
except Exception:
    docx2txt = None

# --- CV file reader (define BEFORE any UI that calls it) ---
def read_cv_file(uploaded) -> str:
    """Read and normalize text from an uploaded CV file (PDF, DOCX, or TXT)."""
    import io, tempfile, re

    if not hasattr(uploaded, "read"):
        raise ValueError("Invalid file-like object passed to read_cv_file().")

    filename = (getattr(uploaded, "name", "") or "").lower()
    content = uploaded.read()

    def _clean(t: str) -> str:
        return re.sub(r"\s+", " ", (t or "")).strip()

    # PDF
    if filename.endswith(".pdf"):
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed. Install with: pip install pdfplumber")
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return _clean("\n".join(pages))

    # DOCX
    if filename.endswith(".docx"):
        if docx2txt is None:
            raise RuntimeError("docx2txt is not installed. Install with: pip install docx2txt")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(content)
            tmp.flush()
            return _clean(docx2txt.process(tmp.name))

    # TXT / fallback
    try:
        return _clean(content.decode("utf-8", errors="ignore"))
    except Exception:
        return _clean(content.decode("latin-1", errors="ignore"))

# --- Selenium for dynamic pages ---
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- OpenAI client ---
from openai import OpenAI

def _job_kwargs(**pairs):
    """Filter to only the fields that exist on the Job dataclass."""
    fields = set(getattr(Job, "__dataclass_fields__", {}).keys())
    return {k: v for k, v in pairs.items() if k in fields}

# Accept both `texts=` and legacy `input=` kwargs
@st.cache_data(show_spinner=False)
def embed_texts(texts: list[str] | None = None,
                model: str = "text-embedding-3-small",
                **kwargs) -> list[list[float]]:
    # Allow callers to pass input=... (alias for texts)
    if texts is None:
        texts = kwargs.get("input") or kwargs.get("texts") or []
    if isinstance(texts, str):
        texts = [texts]
    # use your compat helper to support both openai v1 and v0
    return _create_embeddings_compat(list(texts), model)

# ---- OpenAI embeddings compat (supports both v1 and legacy v0 SDKs) ----
import openai as openai_module  # legacy fallback

# Version-aware OpenAI embeddings (works on v1+ and legacy 0.x)
import openai as openai_module
OPENAI_PY_VERSION = getattr(openai_module, "__version__", "0")

def _create_embeddings_compat(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    if not texts:
        return []
    if client is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY.")

    if not OPENAI_PY_VERSION.startswith("0."):  # v1+
        resp = client.embeddings.create(model=model, input=texts)  # OpenAI v1
        return [d.embedding for d in resp.data]

    # legacy v0.x
    openai_module.api_key = OPENAI_API_KEY
    resp = openai_module.Embedding.create(model=model, input=texts)
    return [d["embedding"] for d in resp["data"]]

@st.cache_data(show_spinner=False)
def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    return _create_embeddings_compat(texts, model)

# -----------------------------
# Chroma (local, file-based) RAG for CVs
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")  # change if you want another folder
CHROMA_COLLECTION = "cv_store"

# --- Embedding wrapper (works even if embed_texts isn't defined yet) ---
def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    return _create_embeddings_compat(texts, model)
    # Prefer your existing embed_texts if present
    try:
        return embed_texts(texts, model=model)  # uses your cached helper if defined
    except NameError:
        pass  # fall back to a local minimal implementation

    # Fallback: direct OpenAI call
    if client is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY.")
    vectors: List[List[float]] = []
    BATCH = 200
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = embed_texts(batch, model=model)
        vectors.extend([d.embedding for d in resp.data])
    return vectors

CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")  # keep your existing path

def _get_chroma_client():
    if not CHROMA_OK or chroma_mod is None:
        st.error("Chroma is not available. Install with: pip install chromadb")
        return None
    # Prefer modern API (0.5.x)
    try:
        return chroma_mod.PersistentClient(path=CHROMA_DIR)
    except Exception:
        # Older API fallback (0.3.x) using Settings
        if ChromaSettings is None:
            st.error("Your Chroma version lacks PersistentClient and Settings. "
                     "Either upgrade chromadb>=0.5 or pin to a compatible 0.3.x with Settings.")
            return None
        try:
            return chroma_mod.Client(ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=CHROMA_DIR
            ))
        except Exception as e:
            st.error(f"Failed to create Chroma client: {e}")
            return None

def _get_cv_collection(client):
    if client is None:
        return None
    try:
        col = client.get_or_create_collection(name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    except TypeError:
        # Older API doesn't support metadata in get_or_create_collection
        col = client.get_or_create_collection(name=CHROMA_COLLECTION)
    return col

def _chroma_count(col) -> int:
    try:
        return col.count()  # new API
    except Exception:
        try:
            # older API might not have .count()
            res = col.get()
            return len(res.get("ids", []))
        except Exception:
            return 0

# Config
# -----------------------------
st.set_page_config(page_title="Careers@Gov Job Matcher", layout="wide")

from utility import check_password

# Do not continue if check_password is not True.  
if not check_password():  
    st.stop()

# ---- Panel flags (make sure these defaults are set earlier) ----
st.session_state.setdefault("show_about", False)
st.session_state.setdefault("show_methodology", False)

# ---- Helpers ----
def _open(key: str):  st.session_state[key] = True
def _close(key: str): st.session_state[key] = False

# ---- Unique keys (avoid collisions) ----
ABOUT_TOP_KEY         = "about_top_btn_v2"
ABOUT_SIDEBAR_KEY     = "about_sidebar_btn_v2"
ABOUT_CLOSE_KEY       = "about_close_btn_v2"
METH_TOP_KEY          = "methodology_top_btn_v2"
METH_SIDEBAR_KEY      = "methodology_sidebar_btn_v2"
METH_CLOSE_KEY        = "methodology_close_btn_v2"

# ---- Top-of-page buttons ----
c1, c2, _ = st.columns([1.2, 1.8, 6])
with c1:
    st.button("About Us", key=ABOUT_TOP_KEY, use_container_width=True,
              on_click=_open, args=("show_about",))
with c2:
    st.button("Methodology", key=METH_TOP_KEY, use_container_width=True,
              on_click=_open, args=("show_methodology",))

# ---- Panels ----
if st.session_state.get("show_about", False):
    with st.expander("About Us", expanded=True):
        st.markdown(
            "This prototype matches CVs to Careers@Gov postings using LLM-based parsing, "
            "OpenAI embeddings, and cosine similarity. Built for learning and experimentation."
        )
        st.button("Close", key=ABOUT_CLOSE_KEY, on_click=_close, args=("show_about",))

if st.session_state.get("show_methodology", False):
    with st.expander("Methodology", expanded=True):
        st.markdown(
            """
**Pipeline**
1. Ingestion: scrape listings or load sample JSON.  
2. Normalization: map to fields (title, organisation, employment_type).  
3. CV parsing: GPT-4o-mini extracts skills/roles/industries/summary (JSON).  
4. Embeddings: OpenAI `text-embedding-3-small`.  
5. Scoring: cosine similarity (candidate vector vs. job text).  
6. Skill overlaps: substring match of candidate skills in job title+description.  
7. Ranking: sort by similarity (desc), then overlap count.
            """
        )
        st.button("Close", key=METH_CLOSE_KEY, on_click=_close, args=("show_methodology",))

# ---- Sidebar openers (optional; note unique keys) ----
with st.sidebar:
    st.markdown("---")
    st.subheader("Info")
    st.button("About Us", key=ABOUT_SIDEBAR_KEY, use_container_width=True,
              on_click=_open, args=("show_about",))
    st.button("Methodology", key=METH_SIDEBAR_KEY, use_container_width=True,
              on_click=_open, args=("show_methodology",))


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY environment variable is not set. Some features will not work.")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SAMPLE_JOBS_JSON = r"""
{
  "generated_at": "2025-08-15T08:21:00.235343Z",
  "source_file": "job_listings.json",
  "record_count": 2,
  "records": [
    {
      "job_title": "Intern, Policy Research and Stakeholder Engagement (FWFE), WPSD",
      "organization": "Ministry of Manpower",
      "employment_type": "Internship",
      "location": "Singapore",
      "description": "Assist with policy research and stakeholder engagement initiatives; support data analysis and report drafting. Skills: research, writing, Excel/Sheets."
    },
    {
      "job_title": "Assistant Executive (Infrastructure Sustainment)",
      "organization": "MHA - Singapore Prison Service (SPS)",
      "employment_type": "Permanent/Contract",
      "location": "Singapore",
      "description": "Support infrastructure sustainment and facilities operations; coordinate vendors and maintenance schedules. Skills: operations, coordination, MS Office."
    },
    {
      "job_title": "General Education Officer (with teaching qualifications) for Tamil Language",
      "organization": "Ministry of Education",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Sch of Tech for the Arts, Media and Design – Assoc. Lecturer Video Prod, AI Film",
      "organization": "Republic Polytechnic",
      "employment_type": "Contract"
    },
    {
      "job_title": "ACE@RP - Assistant Manager/Deputy Manager",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "LECTURER, AEROSPACE ENGINEERING - SCHOOL OF ENGINEERING",
      "organization": "Ngee Ann Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-ITCD] MANAGER / DEPUTY / ASST MANAGER,  IT SYSTEMS (NETWORK)",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Lecturer (Human Resource Management) - School of Business",
      "organization": "Singapore Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Assistant Regulatory Manager/Senior Enforcement Inspector, TRB",
      "organization": "Health Sciences Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Research Executive",
      "organization": "Temasek Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-ITCD] SNR MGR / MANAGER / DEPUTY MANAGER, CLOUD INFRA AUTOMATION",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Lecturer/Senior Lecturer (Banking & Finance) - School of Business Management",
      "organization": "Nanyang Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-RAOM] EXECUTIVE / ENGINEER, ASSET ENGINEERING (POWER & SERVICES)",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Asst Mgr / Mgr, IP&D (Facilities Management) - 12 Months Contract",
      "organization": "National Arts Council",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Executive / Senior Executive - Licensing",
      "organization": "National Parks Board",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Assistant Manager/Manager (Environmental Hygiene Compliance Branch)",
      "organization": "National Environment Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Officer, Asset Mgt & Funds (Funds team)",
      "organization": "Monetary Authority of Singapore",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Lead/Senior Economist, Macroprudential Surveillance Department",
      "organization": "Monetary Authority of Singapore",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Assistant Manager to Senior Manager Analytics & Insights 2",
      "organization": "Singapore Tourism Board",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Director of Policy & Projects",
      "organization": "Civil Aviation Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior/ Investigator (Etomidate E-Vaporisers), Health Products Regulation Group",
      "organization": "Health Sciences Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-NXG] MANAGER/DEPUTY/ASSISTANT MANAGER, COMMUNICATIONS (SYSTEMS)",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Assistant Director to Deputy Director, Human Resource Development",
      "organization": "Singapore Tourism Board",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager (Outreach - Special Project), NE CDC",
      "organization": "People's Association",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Assistant Executive - Contract",
      "organization": "National Environment Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Manager / Manager (Services Demand & Capacity - Planning)",
      "organization": "Ministry of Health",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Manager/ Senior Manager (Corporate Planning)",
      "organization": "VITAL",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager/ Assistant Director (2 years contract)",
      "organization": "Singapore Food Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Engineering Mgr/Asst Engineering Mgr (Integrated Waste Mgmt Facility) - Store",
      "organization": "National Environment Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager/Senior Executive, Service Policy and Leadership, ServiceSG, PSD",
      "organization": "Public Service Division",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior Manager / Manager (Services Demand & Capacity - Care Transformation)",
      "organization": "Ministry of Health",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager, Corporate Communications & Engagement Department",
      "organization": "Accounting and Corporate Regulatory Authority",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior Project Manager (New Estates) - 2 Yr Contract",
      "organization": "JTC Corporation",
      "employment_type": "Contract"
    },
    {
      "job_title": "Intern, Business Transformation, WPD",
      "organization": "Ministry of Manpower",
      "employment_type": "Internship"
    },
    {
      "job_title": "Senior / Court Family Specialist (Family Justice Courts)",
      "organization": "Supreme Court",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Manager / Senior Manager, COE Hub",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior Executive, Strategic Planning and Policy",
      "organization": "Intellectual Property Office of Singapore",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Marine Manager (Marine Safety & Port Operations) (2-years contract)",
      "organization": "Maritime and Port Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Temporary Assistant Manager/ Manager (HR Ops) Management of Pfiles",
      "organization": "Ministry of Digital Development and Information",
      "employment_type": "Contract"
    },
    {
      "job_title": "Intern, Strategic Planning (AI tools), SPTD",
      "organization": "Ministry of Manpower",
      "employment_type": "Internship"
    },
    {
      "job_title": "Watch Manager (Vessel Traffic Management)) (2-years contract)",
      "organization": "Maritime and Port Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Executive/Manager (Data Analytics) (1-Year Contract)",
      "organization": "Ministry of National Development",
      "employment_type": "Contract"
    },
    {
      "job_title": "Assistant Manager, Events & Programming",
      "organization": "Sentosa Development Corporation",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Manager / Manager (Health Analytics)",
      "organization": "Ministry of Health",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Intern, Compliance Strategy & Analytics, WPD",
      "organization": "Ministry of Manpower",
      "employment_type": "Internship"
    },
    {
      "job_title": "LECTURER, JOINT PROGRAMME - SCHOOL OF BUSINESS & ACCOUNTANCY",
      "organization": "Ngee Ann Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "LECTURER, MECHATRONICS & ROBOTICS - SCHOOL OF ENGINEERING",
      "organization": "Ngee Ann Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Assistant Manager / Senior Executive (SATCC or Seletar)",
      "organization": "Civil Aviation Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager / Marketing",
      "organization": "National Parks Board",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Manager/Assistant Dir (Space Management)",
      "organization": "Maritime and Port Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Lecturer/Senior Lecturer (Food & Beverage Business) - School of Applied Science",
      "organization": "Nanyang Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Lecturer - School of Electrical and Electronic Engineering",
      "organization": "Singapore Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Deputy Director (Seafarers Policy, Development & Welfare)",
      "organization": "Maritime and Port Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Manager or Manager / Communications (Media) (Contract)",
      "organization": "National Parks Board",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Deputy/Asst. Director, Investment Risk & Performance Mgmt (Internal Fund Mgmt)",
      "organization": "Monetary Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Communications & Engagement Manager (Partnerships & Outreach)",
      "organization": "Housing and Development Board",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Assistant Director, Records Management",
      "organization": "National Environment Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager / Senior Manager - Centre for Animal Rehabilitation (Contract)",
      "organization": "National Parks Board",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Lecturer / Senior Lecturer (Diploma in Psychology Studies)",
      "organization": "Temasek Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager/Senior Manager, Knowledge Services",
      "organization": "Economic Development Board",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager/Senior Manager, Planning (Temporary Contract - 12 months) [SPD]",
      "organization": "Workforce Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "Intern, RPA Enhancement and Development, WPD",
      "organization": "Ministry of Manpower",
      "employment_type": "Internship"
    },
    {
      "job_title": "Senior Accountant / Accountant",
      "organization": "Singapore Labour Foundation",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager / Senior Manager, Industry Development (IDD2)",
      "organization": "SkillsFuture Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-AM] PRINCIPAL / SENIOR EXECUTIVE, ENFORCEMENT, INVESTIGATIONS & INSPECTIONS",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-RCID] SENIOR / EXECUTIVE / ENGINEER, COMMUTER INFRASTRUCTURE DEVELOPMENT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Commercial Properties Manager (Planning)",
      "organization": "Housing and Development Board",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Desk Head 1",
      "organization": "MHA - Singapore Civil Defence Force (SCDF)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager/Senior Manager, Strategy (Temporary Contract - 12 months) [SPD]",
      "organization": "Workforce Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "Deputy Director, Workplace Investigations (2-year Contract)",
      "organization": "Monetary Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Engr Executive/Asst/Engineering Manager (Integrated Waste Mgmt Facility)",
      "organization": "National Environment Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Intern, Policy Research and Stakeholder Engagement (Senior Employment), WPSD",
      "organization": "Ministry of Manpower",
      "employment_type": "Internship"
    },
    {
      "job_title": "Economist, Strategic Planning and Policy",
      "organization": "Intellectual Property Office of Singapore",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "(Senior) Development Partner",
      "organization": "Enterprise Singapore",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-ITCD] IT TECH MANAGER",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Manager/Manager (Human Performance & Wellbeing)",
      "organization": "Civil Aviation Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager/Senior Manager(Tax Investigation) - Investigation and Forensics Division",
      "organization": "Inland Revenue Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Executive, Design Management",
      "organization": "Sentosa Development Corporation",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior Manager/Assistant Director (AML/CFT/CPF), CPAD",
      "organization": "Accounting and Corporate Regulatory Authority",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior Manager / Assistant Director, Workplace Experience",
      "organization": "Ministry of Digital Development and Information",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior / Technical Executive (Hydrographic Survey) (2yr contract)",
      "organization": "Maritime and Port Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Product Manager/ Product Manager (Contract)",
      "organization": "Monetary Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Assistant Director/Senior Assistant Director, FPPD, ISPD",
      "organization": "Ministry of Manpower",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Lead Engineer/Engineer, Engineering, xCode",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Assistant Manager / Manager / Senior Manager, Family Policy (FCD/FEC)",
      "organization": "Ministry Of Social And Family Development",
      "employment_type": "Contract"
    },
    {
      "job_title": "[LTA-RI&E] SENIOR / EXECUTIVE / PROJECT ENGINEER, CIVIL",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Lead Associate/ Senior Associate/ Associate, Continuous Professional Development",
      "organization": "Ministry of Finance",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Lead Engineer, Agile Development, ICPMC",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Legal/ Prosecuting Counsel (1-year Temp)",
      "organization": "Health Sciences Authority",
      "employment_type": "Contract"
    },
    {
      "job_title": "Deputy Director / Head, Manufacturing, Trade & Connectivity",
      "organization": "National Research Foundation",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior Analyst (Defence Capabilities)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Finance Manager (2-year contract)",
      "organization": "Sentosa Development Corporation",
      "employment_type": "Contract"
    },
    {
      "job_title": "Head (Digital Transformation - Procurement), GPFO",
      "organization": "Ministry of Finance",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager (Strategic Communications Technology)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Lead Associate /Senior Associate /Associate, Workforce Planning and Development",
      "organization": "Ministry of Finance",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Lead Data Scientist, Alliance for Digital Transformation (ADX)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager (NS Management) - Gombak",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Associate (Board of Review & Legal Unit)",
      "organization": "Ministry of Finance",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior/Manager, Research & Development (Behavioural Science & Insights)",
      "organization": "Health Promotion Board",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Assistant Director (Strategic Communications Technology)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Systems Analyst (A)",
      "organization": "Housing and Development Board",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior Manager/Manager, Frontline Services, CXD",
      "organization": "Ministry of Manpower",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager/ Senior Manager, School Campus (Quantity Surveying)",
      "organization": "Ministry of Education",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager, Internal Audit (Temp Contract - 12 months) [IAU]",
      "organization": "Workforce Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "HR Manager (Business Partner)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Assistant Manager/Manager (Planning & Organisation Development), ECDA/CDS",
      "organization": "Ministry Of Social And Family Development",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior Associate/Associate (Policies and Practices)",
      "organization": "Ministry of Finance",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Manager (Eco-System & Partnerships)",
      "organization": "MINDEF",
      "employment_type": "Contract"
    },
    {
      "job_title": "Executive, Service Excellence",
      "organization": "Supreme Court",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior Manager/Manager, Media Communications 4, CED",
      "organization": "Ministry of Manpower",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Engineer (IFED / LRD-IFE) - Contract",
      "organization": "Housing and Development Board",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager, CSVD - EPU (Temp Contract - 12 months) [CSVD]",
      "organization": "Workforce Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager, Security",
      "organization": "Supreme Court",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Community Engagement Manager",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Principal Product Manager (Consulting)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Associate/Associate (Economic & Fiscal Analysis)",
      "organization": "Ministry of Finance",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Head, Product Management, xCode",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Lead Engineer/Engineer, Strategic Planning & Coordination, xCode",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Manager (Admin) - Overseas",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager, Customer Operations (Services Sector), WPD",
      "organization": "Ministry of Manpower",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior / Assistant Technical Director, Sustainability Reporting Office",
      "organization": "Accounting and Corporate Regulatory Authority",
      "employment_type": "Contract"
    },
    {
      "job_title": "[LTA-T&ID] PRINCIPAL / SENIOR EXECUTIVE, AUTONOMOUS VEHICLE PROGRAMME OFFICE",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager/Senior Manager, Industry Engagement (Retail) [TLD]",
      "organization": "Workforce Singapore",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager, Industry Engagement (Food Services/Food Manufacturing) [TLD]",
      "organization": "Workforce Singapore",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Deputy/ Assistant Director, Investment Schemes and Disclosures",
      "organization": "Monetary Authority of Singapore",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Engineer (LRD-LR) - Contract",
      "organization": "Housing and Development Board",
      "employment_type": "Contract"
    },
    {
      "job_title": "Intern, Records Management, OSHD",
      "organization": "Ministry of Manpower",
      "employment_type": "Internship"
    },
    {
      "job_title": "Senior Executive, Finance (Accounts Payable)",
      "organization": "Info-communications Media Development Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager, Research and Statistics Unit",
      "organization": "Info-communications Media Development Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior/Lead Economist, Economic Surveillance (International Economy)(Contract)",
      "organization": "Monetary Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-VS] EXECUTIVE, FOREIGN VEHICLE PERMITS (OPERATIONS)",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-RAOM] SENIOR / MANAGER, ASSET MGMT (RENEWAL & PLANNING)",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior/ Data Scientist, Digital Capabilities & Innovation",
      "organization": "Government Technology Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Software Engineer, Digital Economy Products (DEP) - TradeNet",
      "organization": "Government Technology Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-CRL] SENIOR / EXECUTIVE / PROJECT ENGINEER, CIVIL",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Technical Executive (Building)",
      "organization": "Housing and Development Board",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "NLB Academy Intern",
      "organization": "National Library Board",
      "employment_type": "Internship"
    },
    {
      "job_title": "[LTA-FIN] MANAGER / DEPUTY MANAGER, FINANCIAL PLANNING",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Data Engineer, GovTech Anti Scam Products (GASP)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager/Senior Manager (Risk Management), ECDA/TFN",
      "organization": "Ministry Of Social And Family Development",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager/Executive Manager, International Affairs Branch",
      "organization": "National Environment Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager (Training & Learning Development) - Pulau Tekong",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Principal Engineer, Engineering, xCode",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Executive (6-month Temporary Position)",
      "organization": "Temasek Polytechnic",
      "employment_type": "Casual"
    },
    {
      "job_title": "Senior Statistical Specialist (Data Collection), MRSD",
      "organization": "Ministry of Manpower",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Lead Associate/ Associate (Digital Transformation - Procurement)",
      "organization": "Ministry of Finance",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Assistant Director (International Engagements)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager / Facilities (Contract)",
      "organization": "National Parks Board",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Associate/Associate (Strategic Planning and Development)",
      "organization": "Ministry of Finance",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Manager/ Senior Manager/ Lead Manager, School Campus (Facilities Management)",
      "organization": "Ministry of Education",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior Executive/ Manager / Senior Manager, New Campus",
      "organization": "Ministry of Education",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager/Senior Manager, Industry Development (Advanced Manufacturing) [MCD]",
      "organization": "Workforce Singapore",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager/Senior Manager (Business Partnership)",
      "organization": "Civil Aviation Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Mechanical / Electrical Engineer (Contract)",
      "organization": "Housing and Development Board",
      "employment_type": "Contract"
    },
    {
      "job_title": "Platform Infrastructure Engineer, Singpass",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager/Senior Manager (Ops Planning), ECDA/TFN",
      "organization": "Ministry Of Social And Family Development",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager / Senior Manager - Investigation",
      "organization": "National Parks Board",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Principal Manager/Senior Manager / Manager, Ops Policy, WPD",
      "organization": "Ministry of Manpower",
      "employment_type": "Contract"
    },
    {
      "job_title": "Director/ Deputy Director, Technology Business Management",
      "organization": "Monetary Authority of Singapore",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Assistant Manager (Social Media Management) - School of Engineering",
      "organization": "Nanyang Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Assistant Archivist/ Archivist (Archives Services)",
      "organization": "National Library Board",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Lead Engineer/Engineer, Standards & Governance, xCode",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "ICT Infrastructure Engineer (OpenStack) (Contract)",
      "organization": "Monetary Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "CDC Ambassador, SE CDC",
      "organization": "People's Association",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager (Workplace Experience)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Executive (Engagement & Recognition)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Manager, Data Translation & Senior Research Analyst",
      "organization": "Health Promotion Board",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager (Training Policy)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Accountant (2-year contract)",
      "organization": "Singapore Labour Foundation",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Cybersecurity Consultant (Project Management & Governance), CIO",
      "organization": "Cyber Security Agency of Singapore",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Temp Asst Mngr/Snr Exec (Strategic Planning & Programme Integration - Parents)",
      "organization": "Sport Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager/Senior Manager, Resource and Performance Management, SPTD",
      "organization": "Ministry of Manpower",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior Data Analyst/ Principal Data Analyst, Co-Lab, SPTD",
      "organization": "Ministry of Manpower",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Regulatory Specialist, Medical Devices",
      "organization": "Health Sciences Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Temporary Senior Executive / Executive (Enterprise Student Systems)",
      "organization": "Temasek Polytechnic",
      "employment_type": "Contract"
    },
    {
      "job_title": "Software Engineer, Secured Infrastructure Programme Office",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Lead/ Senior Associate (Talent Development Programme)",
      "organization": "Ministry of Finance",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Penetration Test & Vulnerability Assessment Specialist",
      "organization": "Government Technology Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Deputy Director, Organisational Development (CDA)",
      "organization": "Communicable Diseases Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "M/SM, Copywriter/Content Development Specialist (Contract) [POD]",
      "organization": "Workforce Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Data Scientist (Computer Vision), AI Practice",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager (Professional Standards and Training), ECDA/PDS",
      "organization": "Ministry Of Social And Family Development",
      "employment_type": "Contract"
    },
    {
      "job_title": "Principal/ Lead Product Manager, Product Strategy Office",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager/Assistant Manager (Policy)",
      "organization": "Ministry Of Social And Family Development",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "[NSCS] Head (International Relations)",
      "organization": "National Security Coordination Secretariat",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Manager (Performance and Talent Management)",
      "organization": "Military Security Department",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager (Manpower Plans)",
      "organization": "Military Security Department",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager / Senior Manager (Exhibitions & Project Management)",
      "organization": "National Heritage Board",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager (Employer Branding & Campus Outreach)",
      "organization": "Military Security Department",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager (Career Development)",
      "organization": "Military Security Department",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-TRO] EXECUTIVE/ ENGINEER, INNOVATION & ANALYTICS",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] MANAGER, ENFORCEMENT TRAINING",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior / Manager (Health Info Protection Office)",
      "organization": "Ministry of Health",
      "employment_type": "Contract"
    },
    {
      "job_title": "[LTA-TRO] EXECUTIVE/ ENGINEER, TRAFFIC & STREET LIGHTING",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager (Learning & Development Partner)",
      "organization": "Military Security Department",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Deputy Manager, Estates & Development (Mechanical & Electrical)",
      "organization": "Singapore Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[Temp] Ground Facilitators, KidsSTOP (6 Months)",
      "organization": "Science Centre Board",
      "employment_type": "Casual"
    },
    {
      "job_title": "Manager, TeamSG Brand, Content & Events",
      "organization": "Sport Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "[NSCS] Head (Policy 3)",
      "organization": "National Security Coordination Secretariat",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior Design Manager/ Assistant Director, Architecture & Design",
      "organization": "Sentosa Development Corporation",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager / Senior Manager - Customer Experience & Membership",
      "organization": "Sport Singapore",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "SO International Partnerships",
      "organization": "MHA - Singapore Civil Defence Force (SCDF)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Executive/Assistant Manager (Syariah Court)",
      "organization": "Ministry of Culture, Community and Youth",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "M/SM, Digital Marketing Specialist (Temp) [POD]",
      "organization": "Workforce Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior Manager/Manager, Content Editor (Temp) [POD]",
      "organization": "Workforce Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "[LTA-TRO] MANAGER, ENFORCEMENT ENGAGEMENT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager (Service Delivery)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager (Budget)",
      "organization": "Communicable Diseases Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager / Assistant Manager, Finance Services",
      "organization": "Communicable Diseases Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Executive / Senior Executive (SDO) - School of Applied Science",
      "organization": "Nanyang Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Engineering Manager, Digital Economy Products (DEP)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager (Remuneration)",
      "organization": "Military Security Department",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Executive/Assistant Director (Transformation Office), RFPD",
      "organization": "Ministry of Sustainability and the Environment",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Cybersecurity and AI Software Engineer",
      "organization": "Government Technology Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] ASST/ DEPUTY/ MANAGER, TECHNOLOGY, OPERATIONS POLICY & STRATEGY",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-RLE&M] SNR / ASSISTANT PROJECT ENGINEER, RAIL SERVICES",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] ASST MANAGER, VIOLATION MGMT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Executive/Executive (Development Management (NPGNR)) (Contract)",
      "organization": "National Parks Board",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] ASSISTANT ENGINEER, ROAD FACILITY CONSTRUCTION",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] DEPUTY/ ASST MANAGER, ACTIVE MOBILITY ENFORCEMENT PARTNERSHIP",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Product Manager (GreenGov), Digital Economy Products (DEP)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Lead / Senior Software Engineer (Applications Consultant), Consulting Practice",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-TRO] DEPUTY/ ASSISTANT MANAGER, TACTICAL ENFORCEMENT OPS",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-RLE&M] SENIOR / EXECUTIVE / PROJECT ENGINEER, POWER SUPPLY",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "AI Programme & Engagement Manager (National AI Group)",
      "organization": "Government Technology Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] ASSISTANT MANAGER, ACTIVE MOBILITY ENFORCEMENT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Product Manager (TradeNet), Digital Economy Products (DEP)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Product Manager (SSG), Digital Economy Products (DEP)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-TRO] ENGINEER, KPE/MCE MANAGEMENT",
      "organization": "Land Transport Authority",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-TRO] EXECUTIVE ENGINEER / ENGINEER, TRAFFIC ANALYSIS & PROJECTS",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Dy Manager/Manager (Social Work)",
      "organization": "Nanyang Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "(Senior) Investigation Officer, Investigation",
      "organization": "Enterprise Singapore",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-TRO] SENIOR/ EXECUTIVE/ ENGINEER, ROAD FACILITY CONSTRUCTION",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] EXECUTIVE/ ENGINEER, PATH MANAGEMENT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] ASST MANAGER, APPEALS OPS & SETTLEMENT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] DEPUTY/ ASST MANAGER, APPEALS",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Assistant Director – Planning & Capability Development, Active Health",
      "organization": "Sport Singapore",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Team Lead – Policies & Safety Management, Sport Safety & Training Academy",
      "organization": "Sport Singapore",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "[LTA-CC]  SENIOR CONTENT PRODUCER / CONTENT PRODUCER",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Data Engineer, Digital Government Blueprint (DGB 2.0)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-TRO] ASSISTANT MANAGER, ENFORCEMENT ENGAGEMENT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager, Business Service Excellence (Temp Contract - 9 months) [CGD]",
      "organization": "Workforce Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "Principal Software Engineer (Partner Solutions), Consulting Practice",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-TRO] DATA ENGINEER, ROAD DATA MANAGEMENT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] EXECUTIVE/ ENGINEER, ASSET STEWARD & MGMT (TUNNELS)",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Assistant Director – Programme Design & Development, Active Health",
      "organization": "Sport Singapore",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Assistant Manager/ Senior Executive, Admissions",
      "organization": "Singapore Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager, Data Analysis and Review (Temporary Contract) [RPD]",
      "organization": "Workforce Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior Associate/Assistant Director (Learning & Development)",
      "organization": "Accountant-General's Department",
      "employment_type": "Contract"
    },
    {
      "job_title": "Quality Engineer (Test Automation), Singpass",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "M/SM, Digital Partnership Operations & Integration (Temp) [POD]",
      "organization": "Workforce Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "M/SM, Digital Partnerships Development (Temp) [POD]",
      "organization": "Workforce Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "Software Engineer, Singpass (Authentication)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Executive/ Manager, Home Team Emerging Skills Development",
      "organization": "MHA - Home Team Academy (HTA)",
      "employment_type": "Contract"
    },
    {
      "job_title": "Manager / Assistant Manager (Special Projects), NYC [1-year Contract]",
      "organization": "National Youth Council",
      "employment_type": "Contract"
    },
    {
      "job_title": "Temporary Executive / Assistant Manager (Workplace Experience) - 1 year contract",
      "organization": "Ministry of Digital Development and Information",
      "employment_type": "Contract"
    },
    {
      "job_title": "[LTA-TRO] EXEC/ ENGINEER, ROADWORK PLANNING, ROAD WORKS REGULATION & LICENSING",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior / Health Policy Analyst (Regulatory, Policy & Legislation)",
      "organization": "Ministry of Health",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior Executive / Assistant Manager – Customer Experience & Membership",
      "organization": "Sport Singapore",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior Manager / Manager, Financial Reporting",
      "organization": "Communicable Diseases Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Software Engineer (MCCY Discovery)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA TRO] ASSISTANT MANAGER, INVESTIGATIONS",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Software Engineer (Singpass)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-TRO] ASSISTANT/ DEPUTY MANAGER, ROAD & PEDESTRIAN SAFETY",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] ENGINEER, COMMUTER FACILITIES MANAGEMENT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Deputy Director, Corporate Communications (Marcoms)",
      "organization": "Communicable Diseases Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-TRO] ASST MANAGER, WARRANT ENFORCEMENT, VIOLATION MGMT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] EXECUTIVE/ ENGINEER/ DATA ANALYST, ROAD & PEDESTRIAN SAFETY",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-TRO] EXECUTIVE/ ENGINEER, 1 TRAFFIC SCHEMES DESIGN & DEVELOPMENT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Engineering Manager, Singpass",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Strategic Planning Specialist",
      "organization": "Central Provident Fund Board",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior / Health Policy Analyst (Finance Partnerships & Governance)- Partnerships",
      "organization": "Ministry of Health",
      "employment_type": "Internship"
    },
    {
      "job_title": "Conservation Intern (Heritage & Archival Studies)",
      "organization": "Urban Redevelopment Authority",
      "employment_type": "Internship"
    },
    {
      "job_title": "Senior Associate/ Associate (Fiscal Policy)",
      "organization": "Ministry of Finance",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior / Manager (Budget)",
      "organization": "Ministry of Health",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Manager/Senior Manager (Marcom), 3PN (11-months contract)",
      "organization": "PUB, The National Water Agency",
      "employment_type": "Contract"
    },
    {
      "job_title": "Senior Assistant Director/Assistant Director (Knowledge Management)",
      "organization": "Attorney-General's Chambers",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Business Auditor",
      "organization": "Central Provident Fund Board",
      "employment_type": "Contract"
    },
    {
      "job_title": "DevOps & ML Engineer, AI Platform",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Manager/Manager, Occupational Safety & Health Policy (OSHP), WPSD",
      "organization": "Ministry of Manpower",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Engineer / Lead Engineer, Product management, AI Products",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Engineer / Lead Engineer - Embodied AI Research & Development, RAUS COE",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Executive (Protocol - Navy)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Engineer / Lead Engineer, AI Engineering, AI Products",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Technical Executive (Building) - Lift Engineering",
      "organization": "Housing and Development Board",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior Assistant Director/Assistant Director (Procurement Policy)",
      "organization": "Ministry of National Development",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Assistant / Senior Assistant Registrar, Legal Services Regulatory Authority",
      "organization": "Ministry of Law",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Manager (Data Strategy)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Associate/ Associate (Performance & Evaluation)",
      "organization": "Ministry of Finance",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior Associate/ Associate (Grants Governance)",
      "organization": "Ministry of Finance",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Lead Engineer/Engineer, Sensor Systems, Robotics Automation and Unmanned Systems",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Engineer, Workplace Safety, Health and Security, STRMO",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Technology Lead – Transformation Project Office",
      "organization": "Sport Singapore",
      "employment_type": "Contract"
    },
    {
      "job_title": "Self-Employed Collections Executive",
      "organization": "Central Provident Fund Board",
      "employment_type": "Contract"
    },
    {
      "job_title": "Head Manpower - SPF/ Tanglin Div",
      "organization": "MHA - Singapore Police Force (SPF)",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Senior Manager, Media Communicatons 2, CED",
      "organization": "Ministry of Manpower",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Knowledge & Resources Executive (Contract)",
      "organization": "Housing and Development Board",
      "employment_type": "Contract"
    },
    {
      "job_title": "Assistant Director (New Media Engagement)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager/Assistant Manager (Allocation and Systems Department) - Contract",
      "organization": "National Environment Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Engineer / Lead Engineer, Sensors and AI Solutioning, S&S CoE",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager (Scholarship Publicity)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Digital Forensics Engineer, DIF CoE",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager, Corporate Communications (1-year Temp)",
      "organization": "Health Sciences Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Engineer / Lead Engineer, Applied Vision AI R&D, S&S CoE",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Manager/Senior Manager, Retirement Systems, ISPD",
      "organization": "Ministry of Manpower",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Lead Engineer - AI Infrastructure, HTxAI",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Assistant Director / Senior Manager / Manager (Content Regulation)",
      "organization": "Ministry of Digital Development and Information",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Executive Officer (Contract) - SPF/OPS",
      "organization": "MHA - Singapore Police Force (SPF)",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Sr/Asst/Engineer",
      "organization": "PUB, The National Water Agency",
      "employment_type": "Permanent/Contract"
    },
    {
      "job_title": "Lead Engineer, DevOps, Marine Systems, Robotics Automation and Unmanned Systems",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Engineer / Lead Engineer - Software Engineering, AI Products (Full Stack)",
      "organization": "Home Team Science and Technology Agency (HTX)",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Physical Planning Intern",
      "organization": "Urban Redevelopment Authority",
      "employment_type": "Internship"
    },
    {
      "job_title": "Manager (Accident Prevention)",
      "organization": "MINDEF",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Lead Product Managers, NDTC (National Development and Transport Cluster)",
      "organization": "Government Technology Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Lead/Senior Product Managers, CCYC (Culture, Community and Youth Cluster)",
      "organization": "Government Technology Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Hospitality - Lecturer (Hotel & Leisure Management – Food & Beverage)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Data Scientist/ Senior Data Scientist, Singpass",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Cybersecurity Policy Developer",
      "organization": "Government Technology Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Principal Product Manager (ACRA)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "School of Engineering - Associate Lecturer (Unmanned Aircraft Short Courses)",
      "organization": "Republic Polytechnic",
      "employment_type": "Contract"
    },
    {
      "job_title": "School of Infocomm - Lecturer (Cybersecurity & Digital Forensics)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Product Manager (Digital Conveyancing)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Senior Asst Manager, Rewards & Performance / HR Department, NYP",
      "organization": "Nanyang Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Office of International Relations – Manager/Dpy Manager/Asst Manager/Snr Exec",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Hospitality - Lecturer/Senior Lecturer (Events & Project Management)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Hospitality - Lecturer/Senior Lecturer (CET)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager, Industry Development (Office for Space Technology and Industry)",
      "organization": "Economic Development Board",
      "employment_type": "Permanent"
    },
    {
      "job_title": "School of Engineering - Associate Lecturer (Aerospace Engineering)",
      "organization": "Republic Polytechnic",
      "employment_type": "Contract"
    },
    {
      "job_title": "School of Sports and Health - Associate Lecturer (Outdoor Education)",
      "organization": "Republic Polytechnic",
      "employment_type": "Contract"
    },
    {
      "job_title": "Office of Student Support - Manager/Senior Manager (School Counsellor)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Market Director, Network & Partnership Development (SGN)",
      "organization": "Economic Development Board",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Lead Infra Ops & Support Specialist (Contract)",
      "organization": "Monetary Authority of Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "[LTA-IDE] SENIOR / EXECUTIVE ARCHITECT",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Sports and Health-Lecturer/Senior Lecturer (Sport & Exercise Science)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Infocomm - Lecturer (Artificial Intelligence/Machine Learning)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "COI-SCM - Manager/Senior Manager",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Software Engineer, Government Productivity Engineering (QuickBuy)",
      "organization": "Government Technology Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Infocomm - Lecturer (Software Development / Full Stack Development)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Infocomm - Lecturer (FinTech)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Centre for Foundational Studies - Lecturer (Foundational Studies)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Manager (IT Strategy and Planning)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-TRO] ACTIVE MOBILITY ENFORCEMENT OFFICER",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Infocomm - Lecturer/Senior Lecturer (CET and Skills Future)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager, Strategy & Policy (Office of Space Technology & Industry)",
      "organization": "Economic Development Board",
      "employment_type": "Permanent"
    },
    {
      "job_title": "[LTA-HR] MANAGER, HUMAN RESOURCE (TALENT ACQUISITION)",
      "organization": "Land Transport Authority",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Executive Manager/ Assistant Director (Finance Services Department) – Contract",
      "organization": "National Environment Agency",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Director - Digital Solutions & Services",
      "organization": "Nanyang Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Assistant Director, Data Governance",
      "organization": "National Library Board",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Applied Science - Lecturer/Senior Lecturer (Protein Science)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Hospitality - Senior Lecturer / Lecturer (Hotel & Leisure Mgt)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Hospitality - Lecturer/Senior Lecturer",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Infocomm - Lecturer (Cloud Native Infrastructure)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Product Manager, Digital Resiliency Engineering (DRE)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "School of Business – Associate Lecturer (Accounting or Taxation or Audit)",
      "organization": "Republic Polytechnic",
      "employment_type": "Contract"
    },
    {
      "job_title": "School of Business - Lecturer (Business)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Business - Lecturer (Human Resource Management with Psychology)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Office of Student Support - Senior Manager (Mentoring and Data Management)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Applied Science-Lecturer/Senior Lecturer, Applied Chemistry(Org. Syn.)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Manager, Data Analytics and Systems Development [EPD]",
      "organization": "Workforce Singapore",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Hospitality - Lecturer (Hotel & Leisure Management – Front Office)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior Manager, Digital Industry Singapore",
      "organization": "Economic Development Board",
      "employment_type": "Permanent"
    },
    {
      "job_title": "Data Scientist, Responsible AI Team (AI Practice)",
      "organization": "Government Technology Agency",
      "employment_type": "Permanent"
    },
    {
      "job_title": "School of Engineering - Lecturer (Common Engineering Programme)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Infocomm - Lecturer (Common Infocomm Programme)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "M to SM, Financial Planning & Analysis (MTI Budget Management & Reporting)",
      "organization": "Singapore Tourism Board",
      "employment_type": "Permanent"
    },
    {
      "job_title": "School of Sports and Health - CET Associate Lecturer",
      "organization": "Republic Polytechnic",
      "employment_type": "Contract"
    },
    {
      "job_title": "School of Infocomm - Assistant Director Capability and Industry",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Engineering - Associate Lecturer (Electrical & Electronic Engineering)",
      "organization": "Republic Polytechnic",
      "employment_type": "Contract"
    },
    {
      "job_title": "School of Applied Science - Senior/Research Fellow (Aquaculture)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Sports and Health -CET Associate Lecturer (Sports & Exercise Sciences)",
      "organization": "Republic Polytechnic",
      "employment_type": "Contract"
    },
    {
      "job_title": "Office of the Registrar - Assistant Manager/Deputy Manager (Admissions)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "School of Engineering - Lecturer (Aerospace Engineering)",
      "organization": "Republic Polytechnic",
      "employment_type": "Fixed Terms"
    },
    {
      "job_title": "Senior / Health Policy Analyst (Finance Partnerships & Governance) - Governance",
      "organization": "Ministry of Health",
      "employment_type": "Contract"
    }

  ]
}
"""
try:
    SAMPLE_JOBS = [json.loads(SAMPLE_JOBS_JSON)]
except Exception as e:
    st.error(f"Sample JSON parse failed: {e}")
    SAMPLE_JOBS = None

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Job:
    title: str
    organisation: Optional[str]
    employment_type: Optional[str]
    description: str
    url: str = ""   # keep but optional

@dataclass
class CandidateProfile:
    skills: List[str]
    roles: List[str]
    industries: List[str]
    summary: str

# -----------------------------
# Utilities
# -----------------------------

def clean_text(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return t


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@st.cache_data(show_spinner=False)
def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    return _create_embeddings_compat(texts, model)
    """Embeds a list of strings and returns list of vectors."""
    if client is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY.")
    # Chunk to avoid large payloads
    vectors: List[List[float]] = []
    BATCH = 1000  # text-embedding endpoint supports large batches; stay conservative
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = embed_texts(batch, model=model)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def _chrome_driver(headless: bool = True) -> webdriver.Chrome:
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver


def _extract_job_links_from_list_page(driver: webdriver.Chrome, url: str, wait: int = 10) -> List[str]:
    driver.get(url)
    try:
        WebDriverWait(driver, wait).until(
            EC.presence_of_element_located((By.TAG_NAME, "a"))
        )
    except Exception:
        pass
    time.sleep(1.0)
    links = [a.get_attribute("href") for a in driver.find_elements(By.CSS_SELECTOR, "a[href*='/jobs/hrp/']")]
    # Deduplicate and keep only proper job detail URLs
    out = []
    for href in links:
        if href and "/jobs/hrp/" in href:
            out.append(href.split("#")[0])
    return sorted(set(out))


def _guess_meta_from_text(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    org = None
    emp = None
    loc = None
    # Heuristics: look for common labels
    m = re.search(r"(?:Agency|Organisation|Organization)\s*:?\s*([A-Za-z0-9&(),./'\- ]{3,100})", text, flags=re.I)
    if m:
        org = clean_text(m.group(1))
    m = re.search(r"(?:Employment\s*Type|Job\s*Type|Work\s*Type)\s*:?\s*([A-Za-z/\- ]{3,60})", text, flags=re.I)
    if m:
        emp = clean_text(m.group(1))
    m = re.search(r"(?:Location|Based\s*in)\s*:?\s*([A-Za-z0-9,./'\- ]{3,80})", text, flags=re.I)
    if m:
        loc = clean_text(m.group(1))
    return org, emp, loc


def _parse_job_detail_html(html: str, url: str) -> Job:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.find(["h1", "h2"]) 
    title_txt = clean_text(title.get_text(strip=True)) if title else "(Untitled)"

    # Prefer main content blocks if present
    main = soup.find("main") or soup.find("article") or soup.find("div", attrs={"role": "main"})
    text = clean_text((main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)))
    # Trim extremely long pages
    text = text[:20000]

    org, emp, loc = _guess_meta_from_text(text)
    return Job(title=title_txt, organisation=org, employment_type=emp, description=text)


# --- Helpers to support JSON-based sample datasets ---

def _slugify(text: Optional[str]) -> str:
    if not text:
        return "job"
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "job"


def _normalize_sample_dataset(sample: Any) -> List[Job]:
    """Normalize user-provided SAMPLE_JOBS structures into a list of Job objects.

    Supports:
    - dict with key 'records': { records: [ { job_title, organization/organisation, employment_type, description?? } ] }
    - list of such dicts (possibly wrapped with metadata)
    - list of job-like dicts directly
    - (falls through) list of URLs handled elsewhere
    """
    jobs: List[Job] = []

def _normalize_sample_dataset(sample: Any) -> List[Job]:
    jobs: List[Job] = []

    def add_from_rec(rec: Dict[str, Any]):
        if not isinstance(rec, dict):
            return
        title = rec.get("job_title") or rec.get("title") or "(Untitled)"
        org = rec.get("organization") or rec.get("organisation")
        emp = rec.get("employment_type") or rec.get("job_type") or rec.get("type")
        loc = rec.get("location")
        desc = rec.get("description") or rec.get("job_description") or ""
        url = rec.get("url") or f"sample://{_slugify(title)}"

        jobs.append(
            Job(**_job_kwargs(
                title=clean_text(title),
                organisation=org,
                employment_type=emp,
                location=loc,          # ignored if Job has no 'location'
                description=clean_text(desc),
                url=url                # ignored if Job has no 'url'
            ))
        )

    if isinstance(sample, dict):
        recs = sample.get("records")
        if isinstance(recs, list):
            for r in recs:
                add_from_rec(r)
    elif isinstance(sample, list):
        for item in sample:
            if isinstance(item, dict):
                if isinstance(item.get("records"), list):
                    for r in item["records"]:
                        add_from_rec(r)
                else:
                    add_from_rec(item)
            # strings (URLs) are handled elsewhere
    return jobs


def _fetch_job_detail(driver: webdriver.Chrome, url: str, wait: int = 10) -> Optional[Job]:
    try:
        driver.get(url)
        WebDriverWait(driver, wait).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(0.5)
        html = driver.page_source
        return _parse_job_detail_html(html, url)
    except Exception as e:
        return None

def _job_kwargs(**pairs):
    """Filter kwargs to only the fields that actually exist on Job."""
    fields = set(getattr(Job, "__dataclass_fields__", {}).keys())
    return {k: v for k, v in pairs.items() if k in fields}

def _make_job_from_summary(title: str,
                           org: Optional[str],
                           emp: Optional[str],
                           href: Optional[str]) -> Job:
    return Job(**_job_kwargs(
        title=clean_text(title),
        organisation=org,
        employment_type=emp,
        location=None,          # will be ignored if Job has no 'location'
        description="",         # will be ignored if missing
        url=href or ""          # will be ignored if Job has no 'url'
    ))

def _extract_job_summaries_from_list_page(driver: webdriver.Chrome, url: str, wait: int = 10) -> List[Job]:
    """
    Parse a Careers@Gov listing page and return Job objects with:
    title, organisation, employment_type, (optional) url; description left empty.
    """
    driver.get(url)
    try:
        WebDriverWait(driver, wait).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
    except Exception:
        pass
    time.sleep(0.5)

    anchors = driver.find_elements(By.CSS_SELECTOR, "a[href*='/jobs/hrp/']")
    jobs_local: List[Job] = []
    seen = set()

    for a in anchors:
        href = (a.get_attribute("href") or "").split("#")[0]
        title = (a.text or "").strip()
        if not title:
            continue

        key = (title, href)
        if key in seen:
            continue
        seen.add(key)

        # find nearest card/container
        container = None
        for xp in ("ancestor::article[1]", "ancestor::li[1]", "ancestor::div[1]"):
            try:
                container = a.find_element(By.XPATH, xp)
                break
            except Exception:
                continue

        card_text = (container.text if container else "").strip()
        lines = [ln.strip() for ln in card_text.splitlines() if ln.strip()]
        lines_wo_title = [ln for ln in lines if ln != title]
        rest = "\n".join(lines_wo_title)

        # Guess organisation
        org = None
        m = re.search(r"(?:Agency|Organisation|Organization)\s*:?\s*(.+)", rest, flags=re.I)
        if m:
            org = clean_text(m.group(1))
        else:
            for ln in lines_wo_title:
                if not re.search(r"(Closing|Posted|Apply|Location|Employment|Job\s*Type)", ln, flags=re.I):
                    org = clean_text(ln)
                    break

        # Guess employment type
        emp = None
        m = re.search(
            r"(Internship|Permanent|Contract|Temporary|Temp|Fixed\s*Term|Term|Part[- ]time|Full[- ]time)",
            rest, flags=re.I
        )
        if m:
            emp = clean_text(m.group(1))

        jobs_local.append(_make_job_from_summary(title, org, emp, href))

    return jobs_local

@st.cache_data(show_spinner=True)
def scrape_careers_gov(start_page: int, end_page: int, per_page_cap: Optional[int],
                       headless: bool = True, delay: float = 0.3) -> List[Job]:
    """
    Scrape Careers@Gov LISTING pages only and collect job summaries:
    title, organisation, employment_type, (optional) url. No detail fetch here.
    """
    all_jobs: List[Job] = []
    driver = _chrome_driver(headless=headless)
    try:
        for p in range(start_page, end_page + 1):
            list_url = f"https://jobs.careers.gov.sg/?p={p}"
            st.info(f"Scanning page {p}: {list_url}")

            page_jobs = _extract_job_summaries_from_list_page(driver, list_url)
            if not page_jobs:
                st.warning(f"No job cards parsed on page {p}. The site layout may have changed.")

            if per_page_cap:
                page_jobs = page_jobs[:per_page_cap]

            all_jobs.extend(page_jobs)
            time.sleep(delay)

        # de-duplicate across pages by (title, organisation, employment_type)
        dedup: dict[tuple, Job] = {}
        for j in all_jobs:
            key = (j.title, j.organisation or "", j.employment_type or "")
            if key not in dedup:
                dedup[key] = j
        return list(dedup.values())
    finally:
        try:
            driver.quit()
        except Exception:
            pass

def extract_profile_from_cv(cv_text: str) -> CandidateProfile:
    if client is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY.")
    system = (
        "You are an expert career analyst. Extract concise skills as short phrases, "
        "roles (titles), industries, and a 3-4 sentence professional summary from the CV text. "
        "Return strict JSON with keys: skills (string array), roles (string array), industries (string array), summary (string)."
    )
    user = (
        "CV text:\n" + cv_text[:25000]
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    skills = [clean_text(s) for s in data.get("skills", []) if isinstance(s, str)]
    roles = [clean_text(s) for s in data.get("roles", []) if isinstance(s, str)]
    industries = [clean_text(s) for s in data.get("industries", []) if isinstance(s, str)]
    summary = clean_text(data.get("summary", ""))
    return CandidateProfile(skills=skills, roles=roles, industries=industries, summary=summary)

# -----------------------------
# Matching logic
# -----------------------------

def build_job_corpus(jobs: List[Job]) -> List[str]:
    # Use title + description as the semantic text
    return [clean_text(f"{j.title}. {j.description}") for j in jobs]


def find_skill_overlaps(job: Job, candidate_skills: List[str]) -> List[str]:
    text = f"{job.title} {job.description}".lower()
    overlaps = []
    for s in candidate_skills:
        s_norm = s.lower()
        # simple phrase presence check; avoid false positives for very short skills
        if len(s_norm) >= 3 and s_norm in text:
            overlaps.append(s)
    return sorted(set(overlaps))


def score_matches(jobs: List[Job], candidate_profile: CandidateProfile) -> pd.DataFrame:
    if not jobs:
        return pd.DataFrame()

    corpus = build_job_corpus(jobs)
    job_vecs = embed_texts(corpus)  # list of vectors

    # Compose candidate text from summary + skills for embedding
    cand_text = candidate_profile.summary + "\nSkills: " + ", ".join(candidate_profile.skills)
    cand_vec = np.array(embed_texts([cand_text])[0], dtype=np.float32)

    rows = []
    for j, v in zip(jobs, job_vecs):
        v = np.array(v, dtype=np.float32)
        sim = cosine_sim(cand_vec, v)
        overlaps = find_skill_overlaps(j, candidate_profile.skills)
        rows.append({
            "title": j.title,
            "organisation": j.organisation or "",
            "employment_type": j.employment_type or "",
            "similarity": sim,
            "overlap_count": len(overlaps),
            "matched_skills": ", ".join(overlaps),
            "description": j.description[:1500] + ("..." if len(j.description) > 1500 else ""),
        })

    df = pd.DataFrame(rows).sort_values(["similarity", "overlap_count"], ascending=[False, False])
    return df


def explain_match(job: Job, candidate_profile: CandidateProfile) -> str:
    if client is None:
        return "OpenAI not configured."
    prompt = (
        "In 5 bullet points, explain why this candidate may be a fit for the job. "
        "Base your reasoning on explicit evidence from the job text and the candidate skills."
        f"\nJob Title: {job.title}\n"
        f"Job Text (excerpt): {job.description[:1800]}\n"
        f"Candidate Skills: {', '.join(candidate_profile.skills)}\n"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are an expert technical recruiter."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


# -----------------------------
# UI
# -----------------------------

st.title("Careers@Gov — CV ↔ Job Matcher")
st.caption("Scrape public job postings, parse your CV, and see best-fit roles based on semantic similarity and skill overlap.")

with st.sidebar:
    st.header("Data Source")
    use_sample = st.toggle("Use 15 Aug 2025 Careers@Gov scrapped dataset (skip scraping)", value=False)
    st.divider()
    st.subheader("Scraper settings")
    start_page = st.number_input("Start page", 1, 500, 1, step=1)
    end_page = st.number_input("End page", 1, 1000, 3, step=1)
    per_page_cap = st.number_input("Cap job links per page (0 = no cap)", 0, 200, 20, step=5)
    headless = st.toggle("Run Chrome headless", value=True)
    scrape_btn = st.button("Scrape Careers@Gov now", type="primary", use_container_width=True)

    st.markdown("---")

# Collapsible disclaimer
with st.expander("Disclaimer (click to expand)", expanded=False):
    st.markdown(
        """
**IMPORTANT NOTICE:** This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalized advice.
        """
    )

# --- Scraping or sample load ---
jobs: List[Job] = st.session_state.get("jobs", [])

if use_sample:
    if st.session_state.get("jobs_source") != "sample" or not st.session_state.get("jobs"):
        st.info("Loading 15 Aug 2025 scrapped job postings…")
        jobs_tmp: List[Job] = []
        # 1) Try a user-provided SAMPLE_JOBS variable (JSON-like structure)
        sample_var = None
        try:
            sample_var = SAMPLE_JOBS  # type: ignore[name-defined]
        except Exception:
            sample_var = None
        if sample_var is not None:
            normalized = _normalize_sample_dataset(sample_var)
            jobs_tmp.extend(normalized)
            # If the sample was instead a list of URLs, fetch them
            if not jobs_tmp and isinstance(sample_var, list) and all(isinstance(x, str) for x in sample_var):
                for u in sample_var:  # type: ignore[assignment]
                    try:
                        r = requests.get(u, timeout=20)
                        j = _parse_job_detail_html(r.text, u)
                        jobs_tmp.append(j)
                    except Exception:
                        pass
        st.session_state["jobs"] = jobs_tmp
        st.session_state["jobs_source"] = "sample"
    jobs = st.session_state["jobs"]
elif scrape_btn:
    with st.spinner("Scraping listings and job details…"):
        cap = None if per_page_cap == 0 else int(per_page_cap)
        jobs = scrape_careers_gov(int(start_page), int(end_page), cap, headless=headless)
        if not jobs:
            st.warning("No jobs scraped. Try enabling 'Use 15 Aug 2025 Careers@Gov scrapped dataset' or adjust scraper settings.")
        st.session_state["jobs"] = jobs
        st.session_state["jobs_source"] = "scrape"

# Show scraped jobs summary
if st.session_state.get("jobs"):
    jobs = st.session_state["jobs"]
    st.success(f"Collected {len(jobs)} job postings")
    job_preview = pd.DataFrame([
        {
            "title": j.title,
            "organisation": j.organisation or "",
            "employment_type": j.employment_type or "",
        }
        for j in jobs
    ])
    st.dataframe(job_preview, use_container_width=True, hide_index=True)

# --- CV Upload & Extract ---
st.markdown("## 1) Upload your CV")
cv_file = st.file_uploader("Upload a PDF, DOCX, or TXT CV", type=["pdf", "docx", "txt"])

candidate: Optional[CandidateProfile] = None

if cv_file:
    with st.spinner("Reading your CV…"):
        try:
            cv_text = read_cv_file(cv_file)
            st.text_area("Raw CV text (you can edit minor OCR issues if needed)", value=cv_text[:10000], height=200)
        except Exception as e:
            st.error(f"Failed to read CV: {e}")
            cv_text = ""

    if cv_text:
        if client is None:
            st.error("OpenAI not configured. Set OPENAI_API_KEY to extract skills.")
        else:
            with st.spinner("Extracting skills & experience with GPT-4o-mini…"):
                try:
                    candidate = extract_profile_from_cv(cv_text)
                except Exception as e:
                    st.error(f"OpenAI extraction failed: {e}")
                    candidate = None
                else:
                    st.session_state["candidate"] = candidate
                    st.markdown("### Candidate Profile (from CV)")
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown("**Skills**")
                        st.write(candidate.skills)
                    with cols[1]:
                        st.markdown("**Roles**")
                        st.write(candidate.roles)
                    with cols[2]:
                        st.markdown("**Industries**")
                        st.write(candidate.industries)
                    st.markdown("**Summary**")
                    st.write(candidate.summary)


from uuid import uuid4

def index_cv_docs_to_chroma(files: List, label_prefix: str = "cv"):
    chroma_client = _get_chroma_client()
    col = _get_cv_collection(chroma_client)
    if col is None:
        return 0, []

    docs, ids, metas = [], [], []
    for f in files:
        try:
            text = read_cv_file(f)
            if not text.strip():
                continue
            docs.append(text)
            from uuid import uuid4
            ids.append(f"{label_prefix}-{uuid4().hex[:12]}")
            metas.append({"filename": getattr(f, "name", "uploaded_cv")})
        except Exception as e:
            st.warning(f"Failed to read {getattr(f, 'name', 'file')}: {e}")

    if not docs:
        return 0, []

    try:
        vecs = embed_texts(docs)  # <-- no `input=` kwarg
    except Exception as e:
        st.error(f"Embedding failed: {e}")
        return 0, []

    try:
        col.add(documents=docs, metadatas=metas, ids=ids, embeddings=vecs)
        try:
            chroma_client.persist()
        except Exception:
            pass
        return len(ids), ids
    except Exception as e:
        st.error(f"Chroma add failed: {e}")
        return 0, []


def query_cv_store(question: str, top_k: int = 5):
    chroma_client = _get_chroma_client()
    col = _get_cv_collection(chroma_client)
    if col is None:
        return None

    # If empty, short-circuit to a consistent empty shape
    try:
        if _chroma_count(col) == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    except Exception:
        pass

    # Embed the query – ensure q_vec exists before use
    try:
        q_vecs = embed_texts([question])
        if not q_vecs or not q_vecs[0]:
            st.error("Embedding query failed (empty vector).")
            return None
        q_vec = q_vecs[0]
    except Exception as e:
        st.error(f"Embedding query failed: {e}")
        return None

    # Chroma query (new/old APIs both accept query_embeddings)
    try:
        return col.query(
            query_embeddings=[q_vec],
            n_results=int(top_k),
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        st.error(f"Chroma query failed: {e}")
        return None

def rag_answer_over_cvs(question: str, retrieved, model: str = "gpt-4o-mini") -> str:
    """
    Simple RAG synthesis: feed retrieved CV snippets to GPT with the user's question.
    """
    if client is None:
        return "OpenAI not configured. Set OPENAI_API_KEY."

    docs = retrieved.get("documents", [[]])[0]
    metas = retrieved.get("metadatas", [[]])[0]
    ctx_blocks = []
    for i, (d, m) in enumerate(zip(docs, metas), 1):
        name = (m or {}).get("filename", f"doc_{i}")
        snippet = (d or "")[:1200]
        ctx_blocks.append(f"[{i}] {name}\n{snippet}\n")

    system = (
        "You are a helpful assistant. Answer the user's question ONLY using the provided CV excerpts. "
        "If the information is not present, say you couldn't find it. Be concise."
    )
    user = f"Question:\n{question}\n\nContext:\n" + "\n".join(ctx_blocks)

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Failed to generate answer: {e}"

# --- CV file reader (define BEFORE any UI that calls it) ---
def read_cv_file(uploaded) -> str:
    """Read and normalize text from an uploaded CV file (PDF, DOCX, or TXT)."""
    import io, tempfile  # local import is fine
    filename = (uploaded.name or "").lower()
    content = uploaded.read()

    def _clean(t: str) -> str:
        import re
        return re.sub(r"\s+", " ", t or "").strip()

    # PDF
    if filename.endswith(".pdf"):
        if pdfplumber is None:
            raise RuntimeError("pdfplumber is not installed. Install with: pip install pdfplumber")
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return _clean("\n".join(pages))

    # DOCX
    if filename.endswith(".docx"):
        if docx2txt is None:
            raise RuntimeError("docx2txt is not installed. Install with: pip install docx2txt")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(content)
            tmp.flush()
            return _clean(docx2txt.process(tmp.name))

    # TXT / fallback
    try:
        return _clean(content.decode("utf-8", errors="ignore"))
    except Exception:
        return _clean(content.decode("latin-1", errors="ignore"))

# -----------------------------
# Local RAG over Uploaded CVs (Chroma)
# -----------------------------
st.markdown("## 1b) CV Knowledge Base (Local, File-based RAG)")
with st.expander("Build & Query Local CV Store (Chroma)", expanded=False):
    if not CHROMA_OK or chroma_mod is None:
      st.warning("Chroma is not installed. Run: `pip install chromadb`")
    else:
        chroma_client = _get_chroma_client()
        col = _get_cv_collection(chroma_client)
        current_count = _chroma_count(col) if col else 0
        st.caption(f"📦 Stored CV documents: **{current_count}**  (folder: `{CHROMA_DIR}`)")

        uploaded_cvs = st.file_uploader(
            "Upload multiple CV files (PDF/DOCX/TXT) to index into the local store",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="rag_cv_uploader"
        )

        col_i, col_q = st.columns([1, 2])
        with col_i:
            if st.button("Add to Local Store", use_container_width=True, key="btn_add_cvs"):
                if not uploaded_cvs:
                    st.info("Upload at least one CV file first.")
                else:
                    with st.spinner("Indexing CVs into local store…"):
                        added_count, ids = index_cv_docs_to_chroma(uploaded_cvs)
                        if added_count:
                            st.success(f"Indexed {added_count} document(s) into the local store.")
                        else:
                            st.warning("No documents were indexed.")
                # refresh count
                col = _get_cv_collection(_get_chroma_client())
                st.caption(f"📦 Stored CV documents: **{_chroma_count(col)}**")

        with col_q:
            # --- Search the local CV store ---
            question = st.text_input(
                "Ask a question about the stored CVs (e.g., 'Who has React and NLP experience?')",
                key="rag_query"
            )
            topk = st.slider("How many results to retrieve?", 1, 10, 5, key="rag_topk")
            if st.button("Search CV Store", use_container_width=True, key="btn_query_cvs"):
                if not question.strip():
                    st.info("Enter a question first.")
                else:
                    with st.spinner("Retrieving similar CV snippets…"):
                        retrieved = query_cv_store(question, top_k=topk)
                        if not retrieved or not retrieved.get("documents"):
                            st.warning("No results found in local store.")
                            st.session_state["rag_results"] = None
                            st.session_state["rag_question"] = None
                        else:
                            # show hits
                            docs = retrieved["documents"][0]
                            metas = retrieved["metadatas"][0]
                            dists = retrieved.get("distances", [[None]*len(docs)])[0]
                            df_hits = pd.DataFrame([
                                {
                                    "filename": (metas[i] or {}).get("filename", f"doc_{i+1}"),
                                    "distance": (dists[i] if dists and len(dists) > i else None),
                                    "preview": (docs[i] or "")[:240] + ("…" if docs[i] and len(docs[i]) > 240 else "")
                                } for i in range(len(docs))
                            ])
                            st.subheader("Nearest CV Snippets")
                            st.dataframe(df_hits, use_container_width=True, hide_index=True)

                            # PERSIST for next rerun (so the Summarize button can use them)
                            st.session_state["rag_results"] = retrieved
                            st.session_state["rag_question"] = question

             # --- Summarize with GPT (works across reruns) ---
            saved_results = st.session_state.get("rag_results")
            saved_question = st.session_state.get("rag_question")
            if saved_results and saved_question:
                if st.button("Summarize with GPT (RAG)", key="btn_rag_summarize_v2"):
                    if client is None:
                        st.error("OpenAI not configured. Set OPENAI_API_KEY.")
                    else:
                        with st.spinner("Composing answer…"):
                            answer = rag_answer_over_cvs(saved_question, saved_results)
                            st.markdown("### Answer")
                            st.write(answer)

        # Utility: clear local store
        if st.button("Clear Local CV Store", key="btn_clear_chroma"):
            try:
                # safest way: delete collection
                c = _get_cv_collection(_get_chroma_client())
                if c:
                    try:
                        _get_chroma_client().delete_collection(CHROMA_COLLECTION)  # new API
                    except Exception:
                        # fallback: recreate empty collection by deleting and re-creating dir (nuclear option)
                        import shutil
                        if os.path.isdir(CHROMA_DIR):
                            shutil.rmtree(CHROMA_DIR, ignore_errors=True)
                st.success("Local CV store cleared.")
            except Exception as e:
                st.error(f"Failed to clear store: {e}")

# --- Matching ---
st.markdown("## 2) Find Matches")

def _normalize_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure expected columns exist so display never KeyErrors."""
    expected_defaults = {
        "organisation": "",
        "employment_type": "",
        "location": "",
        "matched_skills": "",
        "overlap_count": 0,
        "url": "",
    }
    for col, default in expected_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df

# 1) Compute & store results when clicked
if st.button("Compute job matches", type="primary", key="compute_matches_btn"):
    jobs_current = st.session_state.get("jobs", [])
    cand_current = st.session_state.get("candidate")
    if not jobs_current:
        st.error("No jobs loaded yet. Scrape or use the sample dataset first.")
    elif cand_current is None:
        st.error("No candidate profile yet. Upload a CV and extract skills.")
    else:
        with st.spinner("Embedding and scoring…"):
            try:
                results_df = score_matches(jobs_current, cand_current)
                if results_df.empty:
                    st.warning("No results to show.")
                    st.session_state["results_df"] = None
                else:
                    st.session_state["results_df"] = _normalize_results_df(results_df)
                    st.success("Job matches computed. Adjust the slider to view more/less.")
            except Exception as e:
                st.session_state["results_df"] = None
                st.error(f"Scoring failed: {e}")

# 2) Render slider + table on every rerun
results_df = st.session_state.get("results_df")
if isinstance(results_df, pd.DataFrame) and not results_df.empty:
    max_k = min(50, len(results_df))
    default_k = min(10, max_k)
    top_k = st.slider("Top job matches", 1, max_k, default_k, key="top_k_slider")

    # Show only columns that exist (avoids KeyError if an older df lacks 'url')
    desired_cols = ["title", "organisation", "employment_type", "similarity",
                    "overlap_count", "matched_skills", "url"]
    display_cols = [c for c in desired_cols if c in results_df.columns]

    st.subheader("Top Matches")
    st.dataframe(results_df.head(top_k)[display_cols],
                 use_container_width=True, hide_index=True)

    st.markdown("### Inspect a match")
    row_idx = st.number_input(
        "Row index to inspect (0-based)",
        min_value=0,
        max_value=max(0, min(len(results_df) - 1, 1000)),
        value=0,
        key="inspect_row_idx"
    )
    row = results_df.iloc[int(row_idx)]

    st.markdown(f"**{row.get('title','(Untitled)')}** — {row.get('organisation','')}")
    st.markdown(f"Similarity: `{row.get('similarity', 0):.4f}` | Skills matched: `{row.get('overlap_count', 0)}`")

    url_val = row.get("url", "")
    if url_val:
        st.markdown(f"URL: {url_val}")

    # Find the job object for explanation (prefer URL, fall back to title)
    jobs_current = st.session_state.get("jobs", [])
    job_for_exp = None
    if url_val:
        job_for_exp = next((j for j in jobs_current if j.url == url_val), None)
    if not job_for_exp:
        job_for_exp = next((j for j in jobs_current if j.title == row.get("title")), None)

    cand_current = st.session_state.get("candidate")
    if job_for_exp and cand_current:
        if st.button("Explain this match (GPT-4o-mini)", key="explain_btn"):
            with st.spinner("Generating explanation…"):
                try:
                    exp = explain_match(job_for_exp, cand_current)
                    st.write(exp)
                except Exception as e:
                    st.error(f"Explanation failed: {e}")

st.markdown("---")
st.caption(
    "This demo uses OpenAI text-embedding-3-small for vectors (cosine similarity) and GPT-4o-mini for CV parsing/explanations. "
    "Use responsibly and respect website terms of use when scraping.")
