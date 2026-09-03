import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import streamlit as st

from src.extractor import AbstractInsightExtractor
from src.pubmed_client import PubMedClient, PubMedFetchError
from src.schemas import PubMedPaper


st.set_page_config(
    page_title="Scientific Literature Trend Finder",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    if "papers" not in st.session_state:
        st.session_state.papers = []
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""


init_session_state()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_available_models(api_key: str, base_url: str) -> list:
    """Dynamically fetch list of available models from provider API using the user's API key."""
    if not api_key:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        models_page = client.models.list()
        model_ids = [m.id for m in models_page.data if m.id]
        return sorted(model_ids)
    except Exception:
        return []


# Sidebar Controls
with st.sidebar:
    st.title("Settings & Engine")

    st.divider()

    st.subheader("Extraction Engine")
    engine_mode = st.radio(
        "Select Provider:",
        ["Offline Heuristic Engine", "Local Ollama Model", "Google Gemini API Key", "Groq API Key", "OpenAI API Key"],

        index=0,
        help="Offline Engine & Ollama work free locally without cloud API keys.",
    )

    api_key = ""
    base_url = None
    model_name = "gemini-1.5-flash"

    if "Ollama" in engine_mode:
        api_key = "ollama"
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        dynamic_models = fetch_available_models(api_key, base_url)
        default_ollama = ["llama3.2", "qwen2.5", "llama3", "mistral"]
        model_options = dynamic_models if dynamic_models else default_ollama
        model_name = st.selectbox("Select Ollama Model:", options=model_options, help="Make sure Ollama is running locally on port 11434 or via Docker Compose")

    elif "Gemini" in engine_mode:
        api_key = st.text_input("Gemini API Key:", type="password", help="Free key from aistudio.google.com")
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

        if api_key.strip():
            dynamic_models = fetch_available_models(api_key.strip(), base_url)
            model_options = dynamic_models if dynamic_models else ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        else:
            model_options = ["Enter API Key to load models..."]
        model_name = st.selectbox("Select Model:", options=model_options)

    elif "Groq" in engine_mode:
        api_key = st.text_input("Groq API Key:", type="password", help="Free key from console.groq.com")
        base_url = "https://api.groq.com/openai/v1"

        if api_key.strip():
            dynamic_models = fetch_available_models(api_key.strip(), base_url)
            active_models = [m for m in dynamic_models if not any(x in m for x in ["8192", "3b-preview", "1b-preview"])] if dynamic_models else []
            model_options = active_models if active_models else ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
        else:
            model_options = ["Enter API Key to load models..."]
        model_name = st.selectbox("Select Model:", options=model_options)

    elif "OpenAI" in engine_mode:
        api_key = st.text_input("OpenAI API Key:", type="password", help="Key from platform.openai.com")
        base_url = "https://api.openai.com/v1"

        if api_key.strip():
            dynamic_models = fetch_available_models(api_key.strip(), base_url)
            active_models = [m for m in dynamic_models if "gpt" in m] if dynamic_models else []
            model_options = active_models if active_models else ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
        else:
            model_options = ["Enter API Key to load models..."]
        model_name = st.selectbox("Select Model:", options=model_options)





    max_papers = st.slider("Number of recent abstracts to fetch:", min_value=1, max_value=10, value=5)


    st.divider()

    st.subheader("About & How to Use")
    st.markdown(
        "1. Enter a scientific keyword (e.g. *HIV phylogenetics*).\n"
        "2. Click **Find Trends** or a quick topic button.\n"
        "3. Automatically extract 2-bullet summaries, study locations, and sample sizes from top PubMed literature.\n"
        "4. Download structured results to CSV for your research!"
    )


# Main Interface Header
st.title("Scientific Literature Trend Finder")
st.caption("Extract structured insights, sample sizes, and study locations from PubMed literature.")

# Search Input Bar
col1, col2 = st.columns([5, 1])
with col1:
    user_query = st.text_input(
        "Search Keyword:",
        value=st.session_state.current_query,
        placeholder="e.g., HIV phylogenetics, CRISPR gene editing, Malaria epidemiology",
        label_visibility="collapsed",
    )
with col2:
    search_button = st.button("Find Trends", use_container_width=True, type="primary")

# Keyword Chips
st.markdown("**Suggested Quick Topics:**")
chip_cols = st.columns(4)
sample_topics = ["HIV phylogenetics", "CRISPR gene editing", "Cancer immunotherapy", "Long COVID epidemiology"]

for idx, topic in enumerate(sample_topics):
    if chip_cols[idx].button(topic, key=f"chip_{idx}"):
        st.session_state.current_query = topic
        st.rerun()

st.divider()

# Perform Search & Extraction
if search_button or (user_query and st.session_state.current_query != user_query):
    query_to_search = user_query.strip()
    if not query_to_search:
        st.warning("Please enter a scientific search keyword.")
    else:
        st.session_state.current_query = query_to_search
        with st.spinner(f"Querying PubMed API & extracting trends for '{query_to_search}'..."):
            pubmed_client = PubMedClient()
            try:
                raw_papers = pubmed_client.get_top_abstracts(query_to_search, max_results=max_papers)

                if not raw_papers:
                    st.info(f"No recent articles found in PubMed for query: '{query_to_search}'. Try another term.")
                    st.session_state.papers = []
                else:
                    extractor = AbstractInsightExtractor(
                        api_key=api_key if api_key else None,
                        base_url=base_url,
                        model_name=model_name,
                    )

                    processed_papers = [None] * len(raw_papers)
                    progress_bar = st.progress(0)
                    last_err = None

                    def _process_item(item):
                        i, p = item
                        insight = extractor.extract_paper_insight(p)
                        p.insights = insight
                        err = getattr(extractor, "last_error", None)
                        return i, p, err

                    completed_count = 0
                    with ThreadPoolExecutor(max_workers=min(5, len(raw_papers))) as executor:
                        futures = [executor.submit(_process_item, (i, p)) for i, p in enumerate(raw_papers)]
                        for future in as_completed(futures):
                            i, p, err = future.result()
                            processed_papers[i] = p
                            if err:
                                last_err = err
                            completed_count += 1
                            progress_bar.progress(completed_count / len(raw_papers))

                    progress_bar.empty()
                    st.session_state.papers = processed_papers
                    st.session_state.last_api_error = last_err
                    st.success(f"Successfully processed top {len(processed_papers)} PubMed abstracts!")

            except PubMedFetchError as e:
                st.error(f"PubMed API Error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Display Results
if st.session_state.papers:
    papers = st.session_state.papers

    # Fallback Warning Banner
    if engine_mode != "Offline Heuristic Engine":
        used_heuristics = any(p.insights and "Heuristic Engine" in p.insights.extracted_via for p in papers)
        if used_heuristics:
            st.warning(
                "Notice: Switched to Offline Heuristic Engine to display your results uninterrupted!"
            )
            with st.expander("Show API Error Details", expanded=True):
                err_msg = st.session_state.get("last_api_error")
                if err_msg:
                    st.code(err_msg)
                elif not api_key:
                    st.info("No API key was entered in the sidebar. Please paste your API key to test LLM extraction.")
                else:
                    st.info("LLM call encountered a network or quota error.")



    m1, m2, m3 = st.columns(3)
    m1.metric("Papers Analyzed", len(papers))
    locations_found = len(set(p.insights.study_location for p in papers if p.insights))
    m2.metric("Unique Study Locations", locations_found)
    sample_sizes_found = sum(1 for p in papers if p.insights and p.insights.sample_size != "Not specified")
    m3.metric("Sample Sizes Detected", f"{sample_sizes_found} / {len(papers)}")

    st.subheader("Structured Literature Insights")

    export_data = []
    for p in papers:
        export_data.append({
            "PMID": p.pmid,
            "Title": p.title,
            "Journal": p.journal,
            "Pub Year": p.pub_date,
            "Location": p.insights.study_location if p.insights else "",
            "Sample Size": p.insights.sample_size if p.insights else "",
            "Study Type": p.insights.study_type if p.insights else "",
            "Engine": p.insights.extracted_via if p.insights else "",
            "Bullet 1": p.insights.bullet_summary[0] if p.insights and len(p.insights.bullet_summary) > 0 else "",
            "Bullet 2": p.insights.bullet_summary[1] if p.insights and len(p.insights.bullet_summary) > 1 else "",
            "URL": p.pubmed_url,
        })
    df_export = pd.DataFrame(export_data)

    st.download_button(
        "Export Results to CSV",
        data=df_export.to_csv(index=False),
        file_name=f"pubmed_trends_{st.session_state.current_query.replace(' ', '_')}.csv",
        mime="text/csv",
    )

    st.write("")

    # Render Paper Cards inside distinct bordered container boxes
    for idx, paper in enumerate(papers):
        insight = paper.insights

        with st.container(border=True):
            st.markdown(f"### {idx + 1}. [{paper.title}]({paper.pubmed_url})")

            authors_str = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
            st.caption(f"Authors: **{authors_str}** | Journal: *{paper.journal}* ({paper.pub_date}) | PMID: `{paper.pmid}`")

            if insight:
                st.markdown(
                    f"**Location:** `{insight.study_location}` | **Sample Size:** `{insight.sample_size}` | **Type:** `{insight.study_type}` | **Engine:** `{insight.extracted_via}`"
                )

                st.markdown("**Key Summary Bullets:**")
                for b in insight.bullet_summary:
                    st.markdown(f"- {b}")

            with st.expander("View Full Abstract"):
                st.write(paper.abstract)


