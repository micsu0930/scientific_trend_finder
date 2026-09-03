"""Unit tests for Pydantic data schemas (PaperInsight and PubMedPaper)."""

import pytest
from src.schemas import PaperInsight, PubMedPaper


def test_paper_insight_validation():
    """Test valid creation of PaperInsight schema."""
    insight = PaperInsight(
        bullet_summary=["Objective: Test objective", "Main Finding: Test finding"],
        study_location="Uganda",
        sample_size="n = 150 patients",
        study_type="Clinical Trial",
        extracted_via="LLM Engine (llama3.2)",
    )
    assert len(insight.bullet_summary) == 2
    assert insight.study_location == "Uganda"
    assert insight.sample_size == "n = 150 patients"
    assert insight.study_type == "Clinical Trial"
    assert "LLM Engine" in insight.extracted_via


def test_paper_insight_defaults():
    """Test default field values in PaperInsight."""
    insight = PaperInsight(
        bullet_summary=["Bullet 1"],
        study_location="Global",
    )
    assert insight.sample_size == "Not specified"
    assert insight.study_type == "Observational study"
    assert insight.extracted_via == "Heuristic Engine"


def test_pubmed_paper_url_property():
    """Test default pubmed_url property generation."""
    paper = PubMedPaper(
        pmid="12345678",
        title="Test Article Title",
        abstract="Test abstract content.",
    )
    assert paper.pubmed_url == "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    assert paper.journal == "PubMed Journal"
    assert paper.pub_date == "Recent"


def test_pubmed_paper_custom_url():
    """Test custom URL setting on PubMedPaper."""
    paper = PubMedPaper(
        pmid="87654321",
        title="Custom URL Article",
        abstract="Abstract content",
        url="https://doi.org/10.1016/j.test.2025",
    )
    assert paper.pubmed_url == "https://doi.org/10.1016/j.test.2025"
