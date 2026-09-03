"""PubMed API Client for searching and retrieving scientific literature.

This module handles communication with NCBI Entrez E-utilities APIs:
1. `esearch.fcgi`: Search PubMed database by keyword query and return top PMIDs.
2. `efetch.fcgi`: Retrieve XML article records for PMIDs and parse metadata & abstracts.
"""

import logging
import xml.etree.ElementTree as ET
from typing import List, Optional
import requests

from src.schemas import PubMedPaper

logger = logging.getLogger(__name__)

# NCBI E-utilities endpoint URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
REQUEST_TIMEOUT = 12  # Seconds before timing out HTTP network requests


class PubMedFetchError(Exception):
    """Custom exception raised when querying NCBI PubMed API fails."""
    pass


class PubMedClient:
    """Client for querying the PubMed literature database via NCBI E-utilities.

    Attributes:
        api_key (Optional[str]): NCBI API key for higher rate limit limits (10 req/s vs 3 req/s).
        session (requests.Session): Reusable HTTP session with custom User-Agent.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize PubMed client.

        Args:
            api_key (str, optional): NCBI API key for increased rate limits.
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "ScientificLiteratureTrendFinder/1.0 (academic-research-tool)"}
        )

    def search_pmids(self, query: str, max_results: int = 5) -> List[str]:
        """Search PubMed for a keyword query and return top PMIDs sorted by publication date.

        Args:
            query (str): Scientific keyword query (e.g. 'HIV phylogenetics').
            max_results (int): Maximum number of article IDs to fetch (default: 5).

        Returns:
            List[str]: List of PubMed Unique Identifiers (PMIDs).

        Raises:
            PubMedFetchError: If network request or API call fails.
        """
        if not query or not query.strip():
            return []

        params = {
            "db": "pubmed",
            "term": query.strip(),
            "retmode": "json",
            "retmax": max_results,
            "sort": "pub_date",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = self.session.get(ESEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"PubMed search for '{query}' returned PMIDs: {id_list}")
            return id_list
        except Exception as e:
            logger.error(f"Error executing PubMed esearch for term '{query}': {e}")
            raise PubMedFetchError(f"Failed to query PubMed API: {e}") from e

    def fetch_papers_by_ids(self, pmid_list: List[str]) -> List[PubMedPaper]:
        """Fetch article metadata and XML records for a list of PMIDs.

        Args:
            pmid_list (List[str]): List of PubMed IDs to fetch.

        Returns:
            List[PubMedPaper]: Parsed structured paper metadata objects.

        Raises:
            PubMedFetchError: If network request or XML fetch fails.
        """
        if not pmid_list:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmid_list),
            "retmode": "xml",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = self.session.get(EFETCH_URL, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return self._parse_pubmed_xml(response.content)
        except Exception as e:
            logger.error(f"Error fetching PubMed XML records: {e}")
            raise PubMedFetchError(f"Failed to fetch paper details from PubMed: {e}") from e

    def get_top_abstracts(self, query: str, max_results: int = 5) -> List[PubMedPaper]:
        """Search query and return top parsed PubMedPaper objects.

        Args:
            query (str): Scientific search keyword.
            max_results (int): Top N abstracts to fetch (default: 5).

        Returns:
            List[PubMedPaper]: List of papers containing titles, authors, and abstract text.
        """
        pmids = self.search_pmids(query, max_results=max_results)
        if not pmids:
            return []
        return self.fetch_papers_by_ids(pmids)

    def _parse_pubmed_xml(self, xml_content: bytes) -> List[PubMedPaper]:
        """Parse NCBI PubMed XML response into structured PubMedPaper data objects.

        Args:
            xml_content (bytes): Raw bytes of PubMed efetch XML response.

        Returns:
            List[PubMedPaper]: List of parsed paper objects.
        """
        papers = []

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as pe:
            logger.error(f"XML parse error: {pe}")
            return []

        for article_tag in root.findall(".//PubmedArticle"):
            # 1. PMID
            pmid_elem = article_tag.find(".//MedlineCitation/PMID")
            pmid = pmid_elem.text if pmid_elem is not None else "Unknown"

            # 2. Article Title
            title_elem = article_tag.find(".//Article/ArticleTitle")
            title = "".join(title_elem.itertext()).strip() if title_elem is not None else "Untitled Article"

            # 3. Abstract Text (Handles structured sections e.g., OBJECTIVE, METHODS, RESULTS)
            abstract_elems = article_tag.findall(".//Article/Abstract/AbstractText")
            if abstract_elems:
                abstract_chunks = []
                for ab in abstract_elems:
                    label = ab.attrib.get("Label", "")
                    text = "".join(ab.itertext()).strip()
                    if label and text:
                        abstract_chunks.append(f"{label}: {text}")
                    elif text:
                        abstract_chunks.append(text)
                abstract = "\n\n".join(abstract_chunks)
            else:
                abstract = "No abstract available in PubMed record."

            # 4. Authors List
            author_list = []
            for author_elem in article_tag.findall(".//Article/AuthorList/Author"):
                last_name = author_elem.findtext("LastName", "")
                fore_name = author_elem.findtext("ForeName", "")
                initials = author_elem.findtext("Initials", "")
                name_str = f"{last_name} {initials or fore_name}".strip()
                if name_str:
                    author_list.append(name_str)

            if not author_list:
                author_list = ["Author information not available"]

            # 5. Journal Title & Publication Date
            journal_elem = article_tag.find(".//Article/Journal/Title")
            journal = (
                journal_elem.text.strip()
                if journal_elem is not None and journal_elem.text
                else "PubMed Indexed Journal"
            )

            year_elem = article_tag.find(".//Article/Journal/JournalIssue/PubDate/Year")
            pub_date = year_elem.text if year_elem is not None else "Recent"

            paper = PubMedPaper(
                pmid=pmid,
                title=title,
                authors=author_list,
                journal=journal,
                pub_date=pub_date,
                abstract=abstract,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
            papers.append(paper)

        return papers