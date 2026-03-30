"""
app.py
======
Streamlit dashboard for the Multi-Source Feedback Intelligence System.

Launch with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, timedelta
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from analysis.sentiment import SentimentAnalyser
from analysis.trend_analysis import TrendAnalyser
from analysis.issue_prioritizer import IssuePrioritizer
from fetchers.google_play import GooglePlayFetcher
from fetchers.app_store import AppStoreFetcher
from fetchers.csv_loader import CSVLoader
from reporting.pdf_generator import PDFReportGenerator
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    page_icon=config.DASHBOARD_PAGE_ICON,
    layout="wide",
)


# ===========================================================================
# Helper — strip timezone safely
# ===========================================================================

def _strip_tz(series: pd.Series) -> pd.Series:
    """
    Convert a datetime Series to tz-naive (no timezone).
    Works whether the series has a timezone or not.
    """
    series = pd.to_datetime(series, errors="coerce")
    if hasattr(series.dt, "tz") and series.dt.tz is not None:
        series = series.dt.tz_convert(None)
    return series.dt.normalize()


# ===========================================================================
# Cached data loading & analysis helpers
# ===========================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_google_play(app_id: str, count: int) -> pd.DataFrame:
    """Fetch and cache Google Play reviews."""
    fetcher = GooglePlayFetcher(app_id=app_id, count=count)
    return fetcher.fetch()


@st.cache_data(ttl=3600, show_spinner=False)
def load_app_store(app_id: str, country: str, count: int) -> pd.DataFrame:
    """Fetch and cache App Store reviews."""
    fetcher = AppStoreFetcher(app_id=app_id, country=country, count=count)
    return fetcher.fetch()


@st.cache_data(ttl=3600, show_spinner=False)
def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Run (cached) sentiment analysis on a DataFrame."""
    return SentimentAnalyser.analyse_dataframe(df)


# ===========================================================================
# Main app
# ===========================================================================

def main() -> None:
    """Entry point for the Streamlit app."""
    st.title(f"{config.DASHBOARD_PAGE_ICON} {config.DASHBOARD_TITLE}")
    st.caption("Unified review intelligence across Google Play, App Store, and CSV surveys.")

    # -----------------------------------------------------------------------
    # Sidebar — data sources & filters
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Data Sources")

        # Google Play
        st.subheader("Google Play")
        gp_enabled = st.checkbox("Enable Google Play", value=False)
        gp_app_id = st.text_input("Package Name", "com.spotify.music",
                                   disabled=not gp_enabled)
        gp_count = st.slider("Review Count", 50, 500, 150, 50,
                              disabled=not gp_enabled)

        st.divider()

        # App Store
        st.subheader("Apple App Store")
        as_enabled = st.checkbox("Enable App Store", value=False)
        as_app_id = st.text_input("App ID (numeric)", "324684580",
                                   disabled=not as_enabled)
        as_country = st.selectbox("Country", ["us", "gb", "au", "ca", "in"],
                                   disabled=not as_enabled)
        as_count = st.slider("Review Count ", 50, 500, 150, 50,
                              disabled=not as_enabled)

        st.divider()

        # CSV
        st.subheader("CSV Survey Upload")
        csv_file = st.file_uploader("Upload CSV", type=["csv"])

        st.divider()

        # Filters
        st.header("🔍 Filters")
        date_range = st.date_input(
            "Date Range",
            value=(
                date.today() - timedelta(days=365),
                date.today(),
            ),
        )
        source_filter = st.multiselect(
            "Source",
            options=["google_play", "app_store", "csv"],
            default=["google_play", "app_store", "csv"],
        )
        sentiment_filter = st.multiselect(
            "Sentiment",
            options=["POSITIVE", "NEGATIVE"],
            default=["POSITIVE", "NEGATIVE"],
        )

        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.session_state.clear()
            st.success("Cache cleared!")

        fetch_btn = st.button("🔄 Fetch & Analyse", type="primary")

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------
    if "reviews" not in st.session_state:
        st.session_state["reviews"] = pd.DataFrame()
    if "trend_df" not in st.session_state:
        st.session_state["trend_df"] = pd.DataFrame()
    if "issues_df" not in st.session_state:
        st.session_state["issues_df"] = pd.DataFrame()
    if "alerts" not in st.session_state:
        st.session_state["alerts"] = []

    # -----------------------------------------------------------------------
    # Data fetching
    # -----------------------------------------------------------------------
    if fetch_btn:
        frames: List[pd.DataFrame] = []
        errors: List[str] = []

        with st.spinner("Fetching reviews…"):
            if gp_enabled and gp_app_id:
                try:
                    df_gp = load_google_play(gp_app_id, gp_count)
                    frames.append(df_gp)
                    st.success(f"✅ Google Play: {len(df_gp)} reviews")
                except Exception as exc:
                    errors.append(f"Google Play: {exc}")

            if as_enabled and as_app_id:
                try:
                    df_as = load_app_store(as_app_id, as_country, as_count)
                    frames.append(df_as)
                    st.success(f"✅ App Store: {len(df_as)} reviews")
                except Exception as exc:
                    errors.append(f"App Store: {exc}")

            if csv_file is not None:
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".csv", delete=False
                    ) as tmp:
                        tmp.write(csv_file.read())
                        tmp_path = tmp.name
                    loader = CSVLoader(tmp_path)
                    df_csv = loader.load()
                    os.remove(tmp_path)
                    frames.append(df_csv)
                    st.success(f"✅ CSV: {len(df_csv)} reviews")
                except Exception as exc:
                    errors.append(f"CSV: {exc}")

        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")

        if not frames:
            st.warning("No data loaded. Please enable at least one source.")
            return

        combined = pd.concat(frames, ignore_index=True)

        # Normalise dates immediately after fetch
        combined["date"] = _strip_tz(combined["date"])

        # Sentiment
        with st.spinner("Running sentiment analysis…"):
            try:
                combined = run_sentiment(combined)
            except Exception as exc:
                st.error(f"Sentiment analysis failed: {exc}")
                return

        # Trend
        try:
            analyser = TrendAnalyser()
            trend_df, alerts = analyser.compute(combined)
            st.session_state["trend_df"] = trend_df
            st.session_state["alerts"] = alerts
        except Exception as exc:
            st.warning(f"Trend analysis skipped: {exc}")

        # Issues
        try:
            prioritizer = IssuePrioritizer()
            issues_df = prioritizer.prioritize(combined)
            st.session_state["issues_df"] = issues_df
        except Exception as exc:
            st.warning(f"Issue prioritisation skipped: {exc}")

        st.session_state["reviews"] = combined
        st.rerun()

    # -----------------------------------------------------------------------
    # Dashboard — only render when data is available
    # -----------------------------------------------------------------------
    reviews: pd.DataFrame = st.session_state["reviews"]
    trend_df: pd.DataFrame = st.session_state["trend_df"]
    issues_df: pd.DataFrame = st.session_state["issues_df"]
    alerts: list = st.session_state["alerts"]

    if reviews.empty:
        st.info("👈 Configure your data sources in the sidebar and click **Fetch & Analyse**.")
        _render_sample_hint()
        return

    # -----------------------------------------------------------------------
    # Apply filters
    # -----------------------------------------------------------------------
    filtered = reviews.copy()

    # Ensure date column is always tz-naive before filtering
    filtered["date"] = _strip_tz(filtered["date"])

    # ---- DEBUG INFO (remove after confirming fix) ----
    with st.expander("🛠️ Debug Info — actual review dates", expanded=False):
        st.write("Total reviews before filter:", len(filtered))
        st.write("Date dtype:", str(filtered["date"].dtype))
        st.write("Earliest date:", str(filtered["date"].min()))
        st.write("Latest date:",   str(filtered["date"].max()))
        st.write("Sample dates:",  filtered["date"].head(10).tolist())
    # ---- END DEBUG ----

    # Date filter
    if len(date_range) == 2:
        start = pd.Timestamp(date_range[0]).normalize()
        end   = pd.Timestamp(date_range[1]).normalize() + pd.Timedelta(days=1)
        filtered = filtered[
            (filtered["date"] >= start) & (filtered["date"] < end)
        ]
        logger.info(
            "Date filter: %d reviews between %s and %s",
            len(filtered), start.date(), end.date(),
        )

    # Source filter
    if source_filter:
        filtered = filtered[filtered["source"].isin(source_filter)]

    # Sentiment filter
    if sentiment_filter and "sentiment_label" in filtered.columns:
        filtered = filtered[filtered["sentiment_label"].isin(sentiment_filter)]

    if filtered.empty:
        st.warning("No reviews match the selected filters.")
        st.info(
            f"💡 Tip: The reviews fetched are dated between "
            f"**{reviews['date'].min().date()}** and "
            f"**{reviews['date'].max().date()}**. "
            f"Adjust your date range filter to match."
        )
        return

    # -----------------------------------------------------------------------
    # Alerts banner
    # -----------------------------------------------------------------------
    if alerts:
        for alert in alerts:
            st.error(alert["message"])

    # -----------------------------------------------------------------------
    # KPI Cards
    # -----------------------------------------------------------------------
    total       = len(filtered)
    pos         = int((filtered["sentiment_label"] == "POSITIVE").sum()) \
                  if "sentiment_label" in filtered.columns else 0
    neg         = total - pos
    avg_rating  = float(filtered["rating"].mean()) \
                  if "rating" in filtered.columns else 0.0
    avg_sentiment = float(filtered["sentiment_score"].mean()) \
                    if "sentiment_score" in filtered.columns else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("📝 Total Reviews",   f"{total:,}")
    k2.metric("😊 Positive",        f"{pos:,}",
              f"{100 * pos / total:.1f}%" if total else "")
    k3.metric("😞 Negative",        f"{neg:,}",
              f"{100 * neg / total:.1f}%" if total else "")
    k4.metric("⭐ Avg Rating",      f"{avg_rating:.2f}/5")
    k5.metric("🎯 Avg Sentiment",   f"{avg_sentiment:+.3f}")

    st.divider()

    # -----------------------------------------------------------------------
    # Trend chart
    # -----------------------------------------------------------------------
    if not trend_df.empty:
        st.subheader("📈 Sentiment Trend")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend_df["date"],
            y=trend_df["avg_sentiment"],
            name="Daily Avg",
            line=dict(color="#2980B9", width=1.5),
            opacity=0.7,
        ))
        fig_trend.add_trace(go.Scatter(
            x=trend_df["date"],
            y=trend_df["rolling_avg_sentiment"],
            name=f"{config.ROLLING_WINDOW_DAYS}-Day Rolling Avg",
            line=dict(color="#E74C3C", width=2.5),
        ))
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            legend=dict(orientation="h"),
            height=350,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # -----------------------------------------------------------------------
    # Pie chart + Top Issues
    # -----------------------------------------------------------------------
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🥧 Sentiment Distribution")
        if "sentiment_label" in filtered.columns:
            dist = filtered["sentiment_label"].value_counts().reset_index()
            dist.columns = ["sentiment", "count"]
            fig_pie = px.pie(
                dist,
                names="sentiment",
                values="count",
                color="sentiment",
                color_discrete_map={
                    "POSITIVE": "#27AE60",
                    "NEGATIVE": "#E74C3C",
                },
                hole=0.4,
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("🔥 Top Issues")
        if not issues_df.empty:
            display_issues = issues_df[
                ["rank", "keyword", "frequency",
                 "avg_neg_strength", "priority_score"]
            ].copy()
            display_issues["keyword"] = display_issues["keyword"].str.title()
            st.dataframe(
                display_issues,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "rank": st.column_config.NumberColumn(
                        "Rank", width="small"
                    ),
                    "keyword": st.column_config.TextColumn("Keyword"),
                    "frequency": st.column_config.NumberColumn("Frequency"),
                    "avg_neg_strength": st.column_config.NumberColumn(
                        "Neg Strength", format="%.3f"
                    ),
                    "priority_score": st.column_config.ProgressColumn(
                        "Priority Score",
                        min_value=0,
                        max_value=float(issues_df["priority_score"].max()),
                        format="%.3f",
                    ),
                },
            )
        else:
            st.info("No issues identified in the selected reviews.")

    st.divider()

    # -----------------------------------------------------------------------
    # Raw data table
    # -----------------------------------------------------------------------
    with st.expander("📋 Raw Reviews", expanded=False):
        st.dataframe(filtered, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # PDF Download
    # -----------------------------------------------------------------------
    st.subheader("📄 Download Report")
    if st.button("Generate PDF Report"):
        with st.spinner("Building PDF…"):
            try:
                date_from = (
                    date_range[0]
                    if len(date_range) == 2
                    else date.today() - timedelta(days=365)
                )
                date_to = (
                    date_range[1]
                    if len(date_range) == 2
                    else date.today()
                )
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as tmp:
                    output_path = tmp.name

                generator = PDFReportGenerator(
                    reviews_df=filtered,
                    trend_df=trend_df,
                    issues_df=issues_df,
                    alerts=alerts,
                    date_from=date_from,
                    date_to=date_to,
                    output_path=output_path,
                )
                generator.generate()

                with open(output_path, "rb") as f:
                    pdf_bytes = f.read()
                os.remove(output_path)

                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"feedback_report_{date.today()}.pdf",
                    mime="application/pdf",
                )
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")
                logger.exception("PDF generation error")


def _render_sample_hint() -> None:
    """Show a brief usage hint when no data has been loaded yet."""
    st.markdown(
        """
        ### Quick Start
        1. **Google Play** — enter a package name like `com.spotify.music`
        2. **App Store** — enter a numeric ID like `324684580`
        3. **CSV** — upload a file with a `review_text` column
        4. Click **Fetch & Analyse**
        """
    )


if __name__ == "__main__":
    main()