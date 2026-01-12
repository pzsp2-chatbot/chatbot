from data_preparation.xml_to_json_batch import (ArticleMetadata,
                                                JSONOutputHandler,
                                                ArticleParser)
import json


def test_article_metadata_to_dict_keys():
    meta = ArticleMetadata(
        article_id="A1",
        title="T",
        created="C",
        modified="M",
        language="en",
        doi=None,
        url=None,
        authors=[],
        abstract=None,
        keywords=None,
    )

    data = meta.to_dict()
    expected_keys = {
        "id", "title", "created", "modified",
        "language", "doi", "url",
        "authors", "abstract", "keywords"
    }

    assert set(data.keys()) == expected_keys


def test_article_parser_uses_doi_as_url(tmp_path):
    xml = """
    <article>
        <doi>10.1000/test</doi>
        <url>http://example.com</url>
    </article>
    """
    path = tmp_path / "test.xml"
    path.write_text(xml)

    data = ArticleParser(str(path)).parse().to_dict()
    assert data["url"].startswith("https://doi.org/")


def test_article_parser_uses_raw_url_if_no_doi(tmp_path):
    xml = "<article><url>http://example.com</url></article>"
    path = tmp_path / "test.xml"
    path.write_text(xml)

    data = ArticleParser(str(path)).parse().to_dict()
    assert data["url"] == "http://example.com"


def test_append_to_jsonl(tmp_path):
    data = {"id": "A1"}
    path = tmp_path / "out.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        JSONOutputHandler.append_to_jsonl(data, f)

    lines = path.read_text().splitlines()
    assert json.loads(lines[0])["id"] == "A1"
