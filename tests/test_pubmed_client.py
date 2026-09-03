"""Unit tests for PubMed API client (esearch and efetch XML parser)."""

import pytest
from unittest.mock import MagicMock
from src.pubmed_client import PubMedClient, PubMedFetchError

# Sample PubMed XML response for testing efetch XML parsing
SAMPLE_PUBMED_XML = b"""<?xml version="1.0"?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticleSet, 1st January 2019//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_190101.dtd">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>38123456</PMID>
      <Article>
        <ArticleTitle>Phylogenetic Analysis of HIV-1 Subtypes in Uganda</ArticleTitle>
        <Journal>
          <Title>Journal of Virology</Title>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
        </Journal>
        <Abstract>
          <AbstractText Label="OBJECTIVE">To map viral lineage transmission networks in Kampala.</AbstractText>
          <AbstractText Label="RESULTS">The study included n = 450 patients in Uganda and demonstrated high diversity.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Mukasa</LastName>
            <ForeName>David</ForeName>
            <Initials>D</Initials>
          </Author>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>Jane</ForeName>
            <Initials>J</Initials>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_search_pmids_success(mocker):
    """Test successful PMID search query returning a list of PMIDs."""
    mock_get = mocker.patch("requests.Session.get")
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "esearchresult": {"idlist": ["38123456", "38123457"]}
    }
    mock_get.return_value = mock_response

    client = PubMedClient()
    pmids = client.search_pmids("HIV phylogenetics", max_results=2)
    assert pmids == ["38123456", "38123457"]


def test_search_pmids_empty_query():
    """Test empty query handling."""
    client = PubMedClient()
    assert client.search_pmids("") == []
    assert client.search_pmids("   ") == []


def test_parse_pubmed_xml():
    """Test parsing raw PubMed XML content into PubMedPaper objects."""
    client = PubMedClient()
    papers = client._parse_pubmed_xml(SAMPLE_PUBMED_XML)

    assert len(papers) == 1
    paper = papers[0]
    assert paper.pmid == "38123456"
    assert paper.title == "Phylogenetic Analysis of HIV-1 Subtypes in Uganda"
    assert paper.journal == "Journal of Virology"
    assert paper.pub_date == "2024"
    assert "OBJECTIVE: To map viral lineage transmission networks in Kampala." in paper.abstract
    assert "Mukasa D" in paper.authors
    assert paper.pubmed_url == "https://pubmed.ncbi.nlm.nih.gov/38123456/"


def test_fetch_papers_network_failure(mocker):
    """Test exception handling when NCBI API request fails."""
    mocker.patch("requests.Session.get", side_effect=Exception("Network Connection Timeout"))
    client = PubMedClient()

    with pytest.raises(PubMedFetchError) as exc_info:
        client.fetch_papers_by_ids(["38123456"])
    assert "Failed to fetch paper details" in str(exc_info.value)
