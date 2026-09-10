# Scientific Literature Trend Finder


## Quick Start (How to Run)

### Option 1: Run with Docker Compose (Recommended)

Start the app and local Ollama model together:

```bash
docker compose up --build
```

Access the Streamlit interface at http://localhost:8501.

### Option 2: Run Locally with Python

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Streamlit:
```bash
python3 -m streamlit run app.py
```

Access the application at http://localhost:8501.

---

## How to Use

1. Enter a scientific research keyword (e.g., "HIV phylogenetics") in the search bar or click a suggested topic button.
2. Choose an extraction engine in the sidebar:
   - **Offline Heuristic Engine:** Free, instant extraction without API keys.
   - **Local Ollama Model:** Free offline LLM inference (`llama3.2`).
   - **Google Gemini / Groq / OpenAI:** Cloud LLM extraction using your API key.
3. Click **Find Trends** to view structured 2-bullet summaries, locations, sample sizes, and study types.
4. Click **Export Results to CSV** to download the structured table.

---

## Use Case Diagram

```mermaid
graph TD
    User["USER"] -->|"1. Search keyword"| SearchUI["Streamlit App (app.py)"]
    User -->|"2. Select engine"| EngineSelect["Engine Configuration"]
    
    SearchUI -->|"3. Query top 5 articles"| PubMedAPI["PubMed Client (NCBI E-utilities)"]
    PubMedAPI -->|"4. Return abstract XML"| AbstractData["Raw Abstracts & Metadata"]
    
    AbstractData -->|"5. Pass abstract text"| ExtractionEngine["Hybrid Insight Extractor"]
    EngineSelect -->|"Configures mode"| ExtractionEngine
    
    ExtractionEngine -->|"Mode A: Pydantic Structured JSON"| CloudLLM["Cloud LLM / Ollama"]
    ExtractionEngine -->|"Mode B: Fallback Regex & NLP"| HeuristicEngine["Offline Heuristic Engine"]
    
    CloudLLM -->|"Extract insights"| PydanticSchema["Pydantic Schema (PaperInsight)"]
    HeuristicEngine -->|"Extract insights"| PydanticSchema
    
    PydanticSchema -->|"6. Render metrics & paper cards"| Dashboard["Literature Dashboard"]
    Dashboard -->|"7. Export CSV"| CSVFile["pubmed_trends.csv"]
```


