import xml.etree.ElementTree as ET
from data_preparation.xml_to_json_batch import XMLParser, ContentExtractor


def create_extractor(xml: str, namespace=None):
    """
    Helper: creates ContentExtractor from XML string
    """
    import tempfile
    from pathlib import Path

    if namespace is None:
        namespace = {"ns": "http://ii.pw.edu.pl/lib"}

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
    tmp.write(xml.encode("utf-8"))
    tmp.close()

    parser = XMLParser(tmp.name, namespace)
    return ContentExtractor(parser)


# ---------- LANGUAGE ----------

def test_get_language_en():
    xml = """
    <root xmlns:ns="http://ii.pw.edu.pl/lib">
        <ns:language>
            <code>en</code>
        </ns:language>
    </root>
    """
    extractor = create_extractor(xml)
    assert extractor.get_language() == "en"


def test_get_language_pl():
    xml = """
    <root xmlns:ns="http://ii.pw.edu.pl/lib">
        <ns:language>
            <code>pl</code>
        </ns:language>
    </root>
    """
    extractor = create_extractor(xml)
    assert extractor.get_language() == "pl"


def test_get_language_missing_returns_none():
    xml = "<root></root>"
    extractor = create_extractor(xml, namespace={})
    assert extractor.get_language() is None
    

def test_get_abstract_english():
    xml = """
    <root xmlns:ns="http://ii.pw.edu.pl/lib">
        <ns:language><code>en</code></ns:language>
        <abstractEN>English abstract</abstractEN>
    </root>
    """
    extractor = create_extractor(xml)
    assert extractor.get_abstract() == "English abstract"


def test_get_abstract_polish():
    xml = """
    <root xmlns:ns="http://ii.pw.edu.pl/lib">
        <ns:language><code>pl</code></ns:language>
        <abstractPL>Polski abstrakt</abstractPL>
    </root>
    """
    extractor = create_extractor(xml)
    assert extractor.get_abstract() == "Polski abstrakt"


def test_get_abstract_fallback_to_en():
    xml = "<root><abstractEN>Fallback EN</abstractEN></root>"
    extractor = create_extractor(xml, namespace={})
    assert extractor.get_abstract() == "Fallback EN"


def test_get_abstract_missing_returns_none():
    xml = "<root></root>"
    extractor = create_extractor(xml, namespace={})
    assert extractor.get_abstract() is None


def test_get_keywords_english():
    xml = """
    <root xmlns:ns="http://ii.pw.edu.pl/lib">
        <ns:language><code>en</code></ns:language>
        <keywordsEN>AI, NLP</keywordsEN>
    </root>
    """
    extractor = create_extractor(xml)
    assert extractor.get_keywords() == "AI, NLP"


def test_get_keywords_polish():
    xml = """
    <root xmlns:ns="http://ii.pw.edu.pl/lib">
        <ns:language><code>pl</code></ns:language>
        <keywordsPL>sztuczna inteligencja</keywordsPL>
    </root>
    """
    extractor = create_extractor(xml)
    assert extractor.get_keywords() == "sztuczna inteligencja"


def test_get_keywords_fallback_to_en():
    xml = "<root><keywordsEN>fallback</keywordsEN></root>"
    extractor = create_extractor(xml, namespace={})
    assert extractor.get_keywords() == "fallback"


def test_get_keywords_missing_returns_none():
    xml = "<root></root>"
    extractor = create_extractor(xml, namespace={})
    assert extractor.get_keywords() is None
