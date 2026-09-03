"""Unit tests for AbstractInsightExtractor (Heuristic NLP engine and LLM mode)."""

import json
from unittest.mock import MagicMock
from src.extractor import AbstractInsightExtractor
from src.schemas import PubMedPaper


def test_heuristic_sample_size_extraction():
    """Test regex pattern matching for sample sizes."""
    text1 = "We recruited a cohort of 450 patients with HIV-1 in South Africa."
    text2 = "A total of n = 1,250 subjects were enrolled in the trial."
    text3 = "No sample numbers were mentioned in this theoretical paper."

    assert AbstractInsightExtractor._extract_sample_size(text1) == "Cohort of 450"
    assert AbstractInsightExtractor._extract_sample_size(text2) == "N = 1,250"
    assert AbstractInsightExtractor._extract_sample_size(text3) == "Not specified"


def test_heuristic_location_extraction():
    """Test geographic country/region extraction."""
    text1 = "Clinical study conducted in Uganda and Kenya."
    text2 = "Survey conducted across the United States."
    text3 = "Abstract with no explicit geographic location."

    assert AbstractInsightExtractor._extract_location(text1) == "Uganda, Kenya"
    assert AbstractInsightExtractor._extract_location(text2) == "United States"
    assert AbstractInsightExtractor._extract_location(text3) == "Not specified / Global"


def test_heuristic_study_type_classification():
    """Test keyword-based study type classification."""
    text1 = "Phylogenetic analysis of viral clade diversity."
    text2 = "Randomized double-blind clinical trial of novel antiviral drug."
    text3 = "Systematic review of epidemiological data."

    assert AbstractInsightExtractor._classify_study_type(text1) == "Phylogenetic Analysis"
    assert AbstractInsightExtractor._classify_study_type(text2) == "Clinical Trial"
    assert AbstractInsightExtractor._classify_study_type(text3) == "Systematic Review / Meta-Analysis"


def test_extract_with_heuristics_full_paper():
    """Test full heuristic extraction pipeline for a PubMedPaper object."""
    paper = PubMedPaper(
        pmid="99999",
        title="Genomic Surveillance of Malaria Parasites in Tanzania",
        abstract=(
            "Objective: To analyze genetic markers of drug resistance. "
            "We collected samples from n = 320 patients across Tanzania. "
            "Results: The analysis showed a high frequency of Pfk13 mutations associated with resistance."
        ),
    )

    insight = AbstractInsightExtractor.extract_with_heuristics(paper)

    assert len(insight.bullet_summary) == 2
    assert "Genomic Surveillance" in insight.bullet_summary[0] or "Objective" in insight.bullet_summary[0]
    assert "Tanzania" in insight.study_location
    assert "N = 320" in insight.sample_size or "320" in insight.sample_size
    assert insight.extracted_via == "Heuristic Engine"


def test_llm_extraction_mock(mocker):
    """Test LLM extraction mode with mocked OpenAI API completion response."""
    mock_openai = mocker.patch("src.extractor.OpenAI")
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    mock_json_response = json.dumps({
        "bullet_summary": [
            "Objective: Investigate HIV transmission clusters.",
            "Main Finding: Identified 12 major transmission clusters.",
        ],
        "study_location": "Uganda",
        "sample_size": "n = 500 patients",
        "study_type": "Phylogenetic Analysis",
    })

    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content=mock_json_response))]
    mock_client.chat.completions.create.return_value = mock_completion

    extractor = AbstractInsightExtractor(api_key="test_key", model_name="gpt-4o-mini")
    paper = PubMedPaper(
        pmid="88888",
        title="HIV Transmission Clusters in Kampala",
        abstract="Detailed abstract text describing HIV transmission clusters in Uganda with 500 patients.",
    )

    insight = extractor.extract_paper_insight(paper)

    assert insight.study_location == "Uganda"
    assert insight.sample_size == "n = 500 patients"
    assert insight.study_type == "Phylogenetic Analysis"
    assert "LLM Engine (gpt-4o-mini)" in insight.extracted_via
