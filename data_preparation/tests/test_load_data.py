from data_preparation.load_data import OmegaDownloader, OmegaDownloadError
import xml.etree.ElementTree as ET
from unittest.mock import patch
import pytest
import requests


def test_init_creates_output_dir(tmp_path):
    out = tmp_path / "data"
    OmegaDownloader(output_dir=str(out))
    assert out.exists()


@patch("requests.get")
def test_fetch_batch_success(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.text = "<collection></collection>"

    d = OmegaDownloader("out")
    result = d.fetch_batch(0, 100)

    assert "<collection>" in result


@patch("requests.get")
def test_fetch_batch_http_error(mock_get):
    mock_get.side_effect = requests.RequestException("boom")

    d = OmegaDownloader("out")

    with pytest.raises(OmegaDownloadError):
        d.fetch_batch(0, 100)


def test_parse_batch_valid_xml():
    xml = """
    <collection xmlns:ns2="http://ii.pw.edu.pl/lib">
        <ns2:article><ns2:id>A1</ns2:id></ns2:article>
        <ns2:article><ns2:id>A2</ns2:id></ns2:article>
    </collection>
    """
    d = OmegaDownloader("out")
    articles = d.parse_batch(xml)

    assert len(articles) == 2


def test_parse_batch_no_collection():
    d = OmegaDownloader("out")

    with pytest.raises(OmegaDownloadError):
        d.parse_batch("<xml></xml>")


def test_parse_batch_no_articles():
    xml = '<collection xmlns:ns2="http://ii.pw.edu.pl/lib"></collection>'
    d = OmegaDownloader("out")

    with pytest.raises(OmegaDownloadError):
        d.parse_batch(xml)


def test_parse_batch_invalid_xml():
    d = OmegaDownloader("out")

    with pytest.raises(OmegaDownloadError):
        d.parse_batch("<collection>")


def test_save_article_with_id(tmp_path):
    xml = """
    <ns2:article xmlns:ns2="http://ii.pw.edu.pl/lib">
        <ns2:id>ABC123</ns2:id>
    </ns2:article>
    """
    article = ET.fromstring(xml)
    d = OmegaDownloader(str(tmp_path))

    d.save_article(article, 0)

    assert (tmp_path / "ABC123.xml").exists()


def test_save_article_without_id(tmp_path):
    xml = '<ns2:article xmlns:ns2="http://ii.pw.edu.pl/lib"></ns2:article>'
    article = ET.fromstring(xml)
    d = OmegaDownloader(str(tmp_path))

    d.save_article(article, 5)

    assert (tmp_path / "noid_5.xml").exists()


@patch.object(OmegaDownloader, "fetch_batch")
def test_download_saves_articles(mock_fetch, tmp_path):
    mock_fetch.return_value = """
    <collection xmlns:ns2="http://ii.pw.edu.pl/lib">
        <ns2:article><ns2:id>A1</ns2:id></ns2:article>
        <ns2:article><ns2:id>A2</ns2:id></ns2:article>
    </collection>
    """

    d = OmegaDownloader(
        output_dir=str(tmp_path),
        batch_size=2,
        limit=2
    )

    total = d.download()

    assert total == 2
    assert (tmp_path / "A1.xml").exists()
    assert (tmp_path / "A2.xml").exists()


@patch.object(OmegaDownloader, "fetch_batch")
def test_download_respects_limit(mock_fetch, tmp_path):
    mock_fetch.return_value = """
    <collection xmlns:ns2="http://ii.pw.edu.pl/lib">
        <ns2:article><ns2:id>A1</ns2:id></ns2:article>
        <ns2:article><ns2:id>A2</ns2:id></ns2:article>
    </collection>
    """

    d = OmegaDownloader(
        output_dir=str(tmp_path),
        batch_size=2,
        limit=1
    )

    total = d.download()

    assert total == 1


@patch.object(OmegaDownloader, "download")
def test_handle_download_domain_error(mock_download, capsys):
    mock_download.side_effect = OmegaDownloadError("fail")

    d = OmegaDownloader("out")
    d.handle_download()

    captured = capsys.readouterr()
    assert "[ERROR]" in captured.out


@patch.object(OmegaDownloader, "download")
def test_handle_download_unexpected_error(mock_download, capsys):
    mock_download.side_effect = RuntimeError("boom")

    d = OmegaDownloader("out")
    d.handle_download()

    captured = capsys.readouterr()
    assert "Unexpected Error" in captured.out

