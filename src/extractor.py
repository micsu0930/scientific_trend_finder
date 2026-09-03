"""Hybrid Insight Extractor module.

Combines LLM Structured JSON Extraction (OpenAI/Groq/Gemini/Ollama) with a 100% Free
Offline Heuristic NLP Engine (Regex pattern matching + sentence extraction) to guarantee
uninterrupted operation even without API keys or internet connection.
"""

import json
import logging
import re
from typing import List, Optional
from openai import OpenAI

from src.schemas import PaperInsight, PubMedPaper

logger = logging.getLogger(__name__)

# List of common countries & geographic regions for heuristic location matching
COUNTRIES_AND_REGIONS = [
    "Uganda", "South Africa", "Kenya", "Nigeria", "Ethiopia", "Tanzania", "Malawi", "Zambia", "Zimbabwe",
    "United States", "USA", "US", "Canada", "United Kingdom", "UK", "Germany", "France", "Italy", "Spain",
    "China", "India", "Japan", "Brazil", "Australia", "Sub-Saharan Africa", "Southeast Asia", "Europe",
    "North America", "Latin America", "Global", "Worldwide"
]

# Keywords for study design classification
STUDY_TYPE_KEYWORDS = {
    "Phylogenetic Analysis": ["phylogenetics", "phylogenetic", "tree", "lineage", "evolutionary", "clade", "genomic"],
    "Clinical Trial": ["clinical trial", "randomized", "rct", "placebo", "double-blind"],
    "Cohort Study": ["cohort", "prospective", "retrospective", "longitudinal"],
    "Cross-Sectional Study": ["cross-sectional", "survey", "prevalence study"],
    "Systematic Review / Meta-Analysis": ["systematic review", "meta-analysis", "prisma"],
    "Experimental Study": ["in vitro", "in vivo", "mouse model", "assay", "cell line"],
}


class AbstractInsightExtractor:
    """Hybrid insight extractor supporting LLM APIs and Offline Heuristic NLP."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: str = "gemini-1.5-flash",
    ):
        """Initialize the extractor.

        Args:
            api_key (Optional[str]): Provider API key.
            base_url (Optional[str]): Provider API base URL (e.g. OpenAI/Groq/Gemini/Ollama).
            model_name (str): LLM model identifier.
        """
        self.api_key = api_key
        self.base_url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.model_name = model_name
        self.last_error: Optional[str] = None

        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        else:
            self.client = None

    def extract_paper_insight(self, paper: PubMedPaper) -> PaperInsight:
        """Extract structured insight. Uses LLM API if configured; falls back to Heuristics.

        Args:
            paper (PubMedPaper): Input paper containing abstract text.

        Returns:
            PaperInsight: Validated Pydantic insight object.
        """
        self.last_error = None
        if self.client and paper.abstract and len(paper.abstract.strip()) > 30:
            try:
                return self._extract_with_llm(paper)
            except Exception as e:
                self.last_error = str(e)
                logger.warning(f"LLM extraction failed for PMID {paper.pmid} ({e}). Falling back to Heuristics.")
                return self.extract_with_heuristics(paper)

        return self.extract_with_heuristics(paper)

    def _extract_with_llm(self, paper: PubMedPaper) -> PaperInsight:
        """Extract insight using OpenAI/Gemini/Groq/Ollama API with Pydantic JSON enforcement.

        Args:
            paper (PubMedPaper): Input paper metadata.

        Returns:
            PaperInsight: Structured insight validated via Pydantic.
        """
        system_prompt = (
            "You are a scientific research assistant summarizing medical literature for PhD students.\n"
            "Extract structured metadata from the abstract into JSON matching this format:\n"
            "{\n"
            '  "bullet_summary": ["Bullet 1: objective/methods", "Bullet 2: main finding/conclusion"],\n'
            '  "study_location": "Country or region where study took place, or \'Not specified\'",\n'
            '  "sample_size": "Sample size if mentioned (e.g. \'n = 450 patients\'), or \'Not specified\'",\n'
            '  "study_type": "Study type (e.g. \'Phylogenetic Analysis\', \'Clinical Trial\')"\n'
            "}"
        )

        user_prompt = f"TITLE: {paper.title}\nABSTRACT: {paper.abstract}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        raw_json = response.choices[0].message.content
        data = json.loads(raw_json)
        insight = PaperInsight.model_validate(data)
        insight.extracted_via = f"LLM Engine ({self.model_name})"
        return insight

    @staticmethod
    def extract_with_heuristics(paper: PubMedPaper) -> PaperInsight:
        """Offline Heuristic NLP Extractor using pattern matching and rule-based parsing.

        Args:
            paper (PubMedPaper): Input paper object.

        Returns:
            PaperInsight: Extracted metadata insight using Heuristic engine.
        """
        abstract = paper.abstract or ""
        title = paper.title or ""
        full_text = f"{title} {abstract}"

        # 1. Extract Sample Size via Regex
        sample_size = AbstractInsightExtractor._extract_sample_size(full_text)

        # 2. Extract Geographic Location
        location = AbstractInsightExtractor._extract_location(full_text)

        # 3. Classify Study Category
        study_type = AbstractInsightExtractor._classify_study_type(full_text)

        # 4. Generate 2-Bullet Summary
        bullets = AbstractInsightExtractor._generate_2_bullet_summary(title, abstract)

        return PaperInsight(
            bullet_summary=bullets,
            study_location=location,
            sample_size=sample_size,
            study_type=study_type,
            extracted_via="Heuristic Engine",
        )

    @staticmethod
    def _extract_sample_size(text: str) -> str:
        """Match common sample size regex patterns in academic text (e.g., 'n = 450')."""
        patterns = [
            r"(?i)\bn\s*=\s*(\d[\d,]*\b)",
            r"(?i)\bcohort of\s*(\d[\d,]*)\b",
            r"(?i)\bincluded\s*(\d[\d,]*)\b",
            r"(?i)\b(\d[\d,]{1,})\s*(patients|individuals|participants|subjects|cases|samples|genomes|sequences)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                matched_str = match.group(0).strip()
                if len(matched_str) < 40 and not re.search(r"(?i)HIV-1|SARS-CoV-2", matched_str):
                    return matched_str.capitalize()
        return "Not specified"

    @staticmethod
    def _extract_location(text: str) -> str:
        """Find country or geographic region mentioned in text."""
        found = []
        for loc in COUNTRIES_AND_REGIONS:
            pattern = r"\b" + re.escape(loc) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                if loc in ["US", "USA"]:
                    loc = "United States"
                elif loc == "UK":
                    loc = "United Kingdom"
                if loc not in found:
                    found.append(loc)
        if found:
            return ", ".join(found[:2])
        return "Not specified / Global"

    @staticmethod
    def _classify_study_type(text: str) -> str:
        """Classify scientific study design based on domain keywords."""
        text_lower = text.lower()
        for stype, keywords in STUDY_TYPE_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return stype
        return "Observational / Analytical Study"

    @staticmethod
    def _generate_2_bullet_summary(title: str, abstract: str) -> List[str]:
        """Generate a 2-bullet summary by extracting objective and conclusion sentences."""
        if not abstract or abstract.startswith("No abstract"):
            return [
                f"Objective: Analysis of {title[:80]}...",
                "Main Finding: Abstract text not available in PubMed metadata.",
            ]

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.strip()) > 15]
        if not sentences:
            return [f"Focus: {title[:90]}", "Outcome: Scientific literature analysis."]

        # Bullet 1: Lead / Objective sentence
        bullet_1 = f"Objective: {sentences[0]}"

        # Bullet 2: Conclusion sentence containing key result indicator words
        bullet_2_text = sentences[-1]
        for s in reversed(sentences):
            if any(k in s.lower() for k in ["showed", "demonstrated", "found", "conclude", "results", "associated"]):
                bullet_2_text = s
                break

        bullet_2 = f"Main Finding: {bullet_2_text}"

        if len(bullet_1) > 160:
            bullet_1 = bullet_1[:157] + "..."
        if len(bullet_2) > 160:
            bullet_2 = bullet_2[:157] + "..."
        return [bullet_1, bullet_2]