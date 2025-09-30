# main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, date

# ML libs
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="NLP Phase Evaluator", layout="wide")
st.title("📰 Politifact NLP Phase Evaluator")

# -----------------------
# Helper: Extract date robustly
# -----------------------
def extract_date_from_text(text):
    if not text or not isinstance(text, str):
        return pd.NaT
    m = re.search(r'([A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{4})', text)
    if not m:
        m = re.search(r'([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})', text)
    if m:
        try:
            return pd.to_datetime(m.group(1), errors='coerce')
        except:
            pass
    m2 = re.search(r'(\d{4}-\d{2}-\d{2})', text)
    if m2:
        return pd.to_datetime(m2.group(1), errors='coerce')
    return pd.to_datetime(text, errors='coerce')

# -----------------------
# Scraper: cached
# -----------------------
@st.cache_data(ttl=24*3600)
def scrape_politifact_date_range(start_dt: date, end_dt: date, max_pages: int = 10, sleep_sec: float = 0.8):
    base_url = "https://www.politifact.com/factchecks/list/?page="
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0"}
    all_rows = []
    start_ts = pd.to_datetime(start_dt).normalize()
    end_ts = pd.to_datetime(end_dt).normalize()

    for page in range(1, max_pages+1):
        url = base_url + str(page)
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, "html.parser")
        except:
            break

        cards = soup.select("div.m-statement") or soup.select("li.o-listicle__item") or soup.select("article")
        if not cards:
            break

        page_dates = []
        for card in cards:
            statement = None
            quote = card.select_one(".m-statement__quote") or card.select_one("p") or card.select_one("div.quote")
            if quote:
                statement = quote.get_text(separator=" ", strip=True)
            speaker = None
            sp = card.select_one(".m-statement__name") or card.select_one(".statement__name") or card.select_one("a")
            if sp:
                speaker = sp.get_text(strip=True)
            rating = None
            img = card.select_one(".m-statement__meter img")
            if img and img.has_attr("alt"):
                rating = img["alt"].strip()
            else:
                meter = card.select_one(".m-statement__meter") or card.select_one(".meter")
                if meter:
                    rating = meter.get_text(strip=True)
            date_text = None
            footer = card.select_one(".m-statement__footer") or card.select_one("footer") or card.select_one(".statement__footer")
            if footer:
                date_text = footer.get_text(separator=" ", strip=True)
            if not date_text:
                t = card.find("time")
                if t:
                    date_text = t.get_text(strip=True)
            parsed_date = extract_date_from_text(date_text)
            if parsed_date is pd.NaT or pd.isna(parsed_date):
                extra = card.get_text(" ", strip=True)
                parsed_date = extract_date_from_text(extra)
            if parsed_date is pd.NaT or pd.isna(parsed_date):
                continue
            parsed_date = pd.to_datetime(parsed_date).normalize()
            page_dates.append(parsed_date)
            if start_ts <= parsed_date <= end_ts:
                all_rows.append({
                    "statement": statement or "",
                    "speaker": speaker or "",
                    "rating": rating or "",
                    "date": parsed_date.date().isoformat()
                })
        if page_dates:
            newest_on_page = max([d for d in page_dates if pd.notna(d)])
            if newest_on_page < start_ts:
                break
        time.sleep(sleep_sec)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# -----------------------
# Initialize session_state
# -----------------------
if "data" not in st.session_state:
    st.session_state.data = None

# -----------------------
# UI Layout
# -----------------------
col1, col2, col3 = st.columns([1,2,2])

# LEFT: Data Sourcing
with col1:
    st.subheader("⚙️ Data Sourcing")
    source_choice = st.radio("Source", ["Upload CSV", "Scrape PolitiFact (by date range)"])

    if source_choice == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            st.session_state.data = df
            st.success("CSV uploaded.")
            st.dataframe(df.head())

    else:
        today = pd.Timestamp.today().date()
        default_start = today.replace(year=today.year-1)
        start_date = st.date_input("From", default_start)
        end_date = st.date_input("To", today)
        max_pages = st.slider("Max pages (~20 statements/page)", 1, 50, 5)

        if st.button("Scrape PolitiFact"):
            if start_date > end_date:
                st.error("Start date must be before end date.")
            else:
                with st.spinner("Scraping PolitiFact..."):
                    df_scraped = scrape_politifact_date_range(start_date, end_date, max_pages=max_pages)
                if df_scraped.empty:
                    st.warning("No statements found.")
                else:
                    st.session_state.data = df_scraped
                    st.success(f"Scraped {len(df_scraped)} statements from {start_date} to {end_date}.")
                    st.dataframe(df_scraped.head())

    # Filters
    if st.session_state.data is not None and not st.session_state.data.empty:
        data = st.session_state.data
        st.markdown("**Optional Filters**")
        if "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            min_date = data["date"].min().date()
            max_date = data["date"].max().date()
            fd, td = st.date_input("Filter date range", [min_date, max_date])
            data = data[(data["date"] >= pd.to_datetime(fd)) & (data["date"] <= pd.to_datetime(td))]
        if "speaker" in data.columns:
            speakers = sorted(data["speaker"].fillna("").unique().tolist())
            sel_speaker = st.selectbox("Filter by speaker", options=["All"] + speakers)
            if sel_speaker != "All":
                data = data[data["speaker"] == sel_speaker]
        st.session_state.data = data
        st.write(f"Using {len(data)} statements for analysis.")
        st.dataframe(data.head())

    st.markdown("**Analysis Configuration**")
    nlp_phase = st.selectbox("Choose NLP Phase", ["Lexical", "Syntactic", "Semantic", "Pragmatic", "Discourse"])

# CENTER: Benchmarking & Run Analysis
with col2:
    st.subheader("📊 Benchmarking Results")
    results_df = None

    if st.button("Run Analysis"):
        data = st.session_state.data
        if data is None or data.empty:
            st.warning("No data available. Upload CSV or scrape first.")
        else:
            data.columns = data.columns.str.strip().str.lower()
            label_candidates = ["label", "rating", "target", "class"]
            y_col = None
            for c in label_candidates:
                if c in data.columns:
                    y_col = c
                    break
            if "statement" not in data.columns or y_col is None:
                st.error("Data must have 'statement' and a label column.")
            else:
                X = data["statement"].astype(str)
                y = data[y_col].astype(str)

                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

                vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1,2))
                X_vect = vectorizer.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_vect, y_encoded, test_size=0.2, random_state=42,
                    stratify=y_encoded if len(np.unique(y_encoded))>1 else None
                )

                models = {
                    "Naive Bayes": MultinomialNB(),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "SVM": SVC(kernel="linear"),
                    "KNN": KNeighborsClassifier()
                }

                results = []
                for name, model in models.items():
                    start_t = time.time()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    end_t = time.time()

                    acc = accuracy_score(y_test, y_pred) * 100
                    f1 = f1_score(y_test, y_pred, average="weighted") * 100
                    duration = end_t - start_t

                    results.append({
                        "Model": name,
                        "Accuracy (%)": round(acc,2),
                        "F1-Score (%)": round(f1,2),
                        "Time (s)": round(duration,3)
                    })

                results_df = pd.DataFrame(results)
                st.markdown("**Model Performance Table**")
                st.dataframe(results_df, use_container_width=True)
                metric = st.selectbox("Compare Models on Metric", ["Accuracy (%)", "F1-Score (%)", "Time (s)"])
                st.bar_chart(results_df.set_index("Model")[metric])

# RIGHT: Trade-Off Plot
with col3:
    st.subheader("📈 Performance Trade-Off")
    if results_df is not None and not results_df.empty:
        fig, ax = plt.subplots()
        ax.scatter(results_df["Time (s)"], results_df["Accuracy (%)"], s=100)
        for i, row in results_df.iterrows():
            ax.annotate(row["Model"], (row["Time (s)"] + 0.02, row["Accuracy (%)"] + 0.02))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Trade-Off: Time vs Accuracy")
        st.pyplot(fig)
    else:
        st.info("Run analysis to see trade-off plot.")

# Footer Notes
st.markdown("---")
st.markdown(
    "**Notes:**\n- Scraping uses public PolitiFact pages; a small pause avoids server overload.\n"
    "- If scraping returns no rows, increase max pages or widen the date range.\n"
    "- Session state ensures scraped or uploaded data persists across button clicks."
)

