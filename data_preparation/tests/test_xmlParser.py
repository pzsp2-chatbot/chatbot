from data_preparation.xml_to_json_batch import XMLParser
import json


def test_find_element_returns_none_if_missing(tmp_path):
    xml = "<root></root>"
    path = tmp_path / "test.xml"
    path.write_text(xml)

    parser = XMLParser(str(path), {})
    assert parser.find_element("missing") is None


def test_find_all_elements_multiple(tmp_path):
    xml = "<root><a /><a /><a /></root>"
    path = tmp_path / "test.xml"
    path.write_text(xml)

    parser = XMLParser(str(path), {})
    elements = parser.find_all_elements("a")

    assert len(elements) == 3


def test_get_element_text_empty_element(tmp_path):
    xml = "<root><a></a></root>"
    path = tmp_path / "test.xml"
    path.write_text(xml)

    parser = XMLParser(str(path), {})
    assert parser.get_element_text("a") is None


def test_find_element_with_namespace(tmp_path):
    xml = '<root xmlns:ns="http://example.com/ns"><ns:child>value</ns:child></root>'
    path = tmp_path / "test.xml"
    path.write_text(xml)

    parser = XMLParser(str(path), {"ns": "http://example.com/ns"})
    element = parser.find_element("ns:child")

    assert element is not None
    assert element.text == "value"


def test_get_element_text_with_namespace(tmp_path):
    xml = (
        '<root xmlns:ns="http://example.com/ns"><ns:child>text value</ns:child></root>'
    )
    path = tmp_path / "test.xml"
    path.write_text(xml)

    parser = XMLParser(str(path), {"ns": "http://example.com/ns"})
    text = parser.get_element_text("ns:child")

    assert text == "text value"
