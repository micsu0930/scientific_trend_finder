"""Data Models and Contracts for PubMed Literature Trend Finder.

This module defines Pydantic models used to enforce structured data contracts across
the application. It ensures type validation and predictable fields when receiving data
from NCBI PubMed API or LLM/Heuristic extraction engines.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PaperInsight(BaseModel):
    """Structured insights extracted from a scientific paper abstract.

    Attributes:
        bullet_summary (List[str]): 2 bullet points summarizing core objective and main finding.
        study_location (str): Country or geographic region where study took place.
        sample_size (Optional[str]): Participant count, cohort size, or subject count if mentioned.
        study_type (Optional[str]): Categorized scientific study design (e.g. 'Phylogenetic Analysis').
        extracted_via (str): Extraction engine metadata identifier ('LLM Engine' or 'Heuristic Engine').
    """

    bullet_summary: List[str] = Field(
        ...,
        description="Exactly 2 bullet points summarizing the core objective and main finding.",
        min_length=1,
        max_length=3,
    )

    study_location: str = Field(
        ...,
        description="Location or geographic region of the study.",
    )

    sample_size: Optional[str] = Field(
        default="Not specified",
        description="Sample size or cohort count if mentioned (e.g., 'n = 450 patients').",
    )

    study_type: Optional[str] = Field(
        default="Observational study",
        description="Scientific study category (e.g., 'Phylogenetic Analysis', 'Clinical Trial').",
    )

    extracted_via: str = Field(
        default="Heuristic Engine",
        description="Engine used for extraction ('LLM Engine' or 'Heuristic Engine').",
    )


class PubMedPaper(BaseModel):
    """Container for scientific paper metadata and raw abstract text.

    Attributes:
        pmid (str): NCBI PubMed Unique Identifier.
        title (str): Full article title.
        authors (List[str]): List of author names.
        journal (str): Journal name where article was published.
        pub_date (str): Publication year or date string.
        abstract (str): Raw text of the paper abstract.
        url (Optional[str]): Direct link to PubMed article page.
        insights (Optional[PaperInsight]): Processed structured insights.
    """

    pmid: str = Field(..., description="PubMed Unique Identifier (PMID)")
    title: str = Field(..., description="Article Title")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    journal: str = Field(default="PubMed Journal", description="Journal Name")
    pub_date: str = Field(default="Recent", description="Publication Year/Date")
    abstract: str = Field(..., description="Abstract text of the paper")
    url: Optional[str] = Field(default=None, description="Direct URL to PubMed article")
    insights: Optional[PaperInsight] = Field(
        default=None, description="Extracted LLM or Heuristic insights"
    )

    @property
    def pubmed_url(self) -> str:
        """Return direct PubMed URL using pmid if custom URL is not set."""
        return self.url or f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
