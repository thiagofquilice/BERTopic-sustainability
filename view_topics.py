#!/usr/bin/env python3
"""Interactive viewer for yearly BERTopic trends.

Run ``python view_topics.py --csv path/to/topics_over_year.csv`` to launch a
Streamlit app showing how topics evolve over calendar years.
"""
import argparse
import pandas as pd
import plotly.express as px
try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - runtime check
    print("Streamlit is required. Install it with 'pip install streamlit'.")
    raise SystemExit(1)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load the CSV and parse timestamps."""
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "Topic", "Frequency"])
    return df


def run_app(csv: str, title: str) -> None:
    """Render the Streamlit interface."""
    if title:
        st.set_page_config(page_title=title)
        st.title(title)

    df = load_data(csv)
    df = df.sort_values("Timestamp")
    all_topics = sorted(df["Topic"].unique())
    freq_sum = df.groupby("Topic")["Frequency"].sum()

    def update_selection() -> None:
        topn = int(st.session_state.top_n)
        st.session_state.topics = freq_sum.nlargest(topn).index.tolist()

    if "topics" not in st.session_state:
        st.session_state.topics = all_topics[:10]
    st.number_input(
        "Top-N by total frequency",
        min_value=1,
        max_value=len(all_topics),
        value=10,
        step=1,
        key="top_n",
        on_change=update_selection,
    )
    st.multiselect(
        "Select topics",
        options=all_topics,
        key="topics",
    )

    view = df[df["Topic"].isin(st.session_state.topics)]
    fig = px.line(
        view,
        x=view["Timestamp"].dt.year,
        y="Frequency",
        color="Topic",
        markers=True,
    )
    fig.update_layout(xaxis_title="Year")
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument(
        "--title",
        default="Topic evolution over years",
        help="Custom page title",
    )
    args = ap.parse_args()

    if not st.runtime.scriptrunner.is_running_with_streamlit:
        from streamlit.web import cli as stcli
        import sys

        sys.argv = ["streamlit", "run", sys.argv[0], "--"] + sys.argv[1:]
        sys.exit(stcli.main())
    run_app(args.csv, args.title)


if __name__ == "__main__":
    main()
