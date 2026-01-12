from data_preparation.structure import write_structure
import xml.etree.ElementTree as ET
from io import StringIO

def test_write_structure(tmp_path):
    test_xml = """
    <root>
        <child1 attr="value1">
            <subchild>Text</subchild>
        </child1>
        <child2 />
    </root>
    """

    xml_path = tmp_path / "test.xml"
    xml_path.write_text(test_xml.strip(), encoding="utf-8")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    output_path = tmp_path / "structure.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        write_structure(root, f)

    content = output_path.read_text(encoding="utf-8")

    expected_output = (
        "<root>\n"
        "  <child1 attr=\"value1\">\n"
        "    <subchild>\n"
        "    </subchild>\n"
        "  </child1>\n"
        "  <child2>\n"
        "  </child2>\n"
        "</root>\n"
    )

    assert content == expected_output



def test_write_structure_single_element():
    elem = ET.Element("root")

    buffer = StringIO()
    write_structure(elem, buffer)

    result = buffer.getvalue()

    assert result == "<root>\n</root>\n"


def test_write_structure_nested_elements():
    root = ET.Element("root")
    child = ET.SubElement(root, "child")
    ET.SubElement(child, "grandchild")

    buffer = StringIO()
    write_structure(root, buffer)

    result = buffer.getvalue().splitlines()

    assert result[0] == "<root>"
    assert result[1] == "  <child>"
    assert result[2] == "    <grandchild>"
    assert result[3] == "    </grandchild>"
    assert result[4] == "  </child>"
    assert result[5] == "</root>"

def test_write_structure_with_attributes():
    elem = ET.Element("article", attrib={"type": "article", "lang": "en"})

    buffer = StringIO()
    write_structure(elem, buffer)

    output = buffer.getvalue()

    assert '<article type="article" lang="en">' in output


def test_indentation_levels():
    root = ET.Element("a")
    b = ET.SubElement(root, "b")
    ET.SubElement(b, "c")

    buffer = StringIO()
    write_structure(root, buffer)

    lines = buffer.getvalue().splitlines()

    assert lines[0].startswith("<a>")
    assert lines[1].startswith("  <b>")
    assert lines[2].startswith("    <c>")


def test_write_structure_with_namespace():
    ns = "http://ii.pw.edu.pl/lib"
    elem = ET.Element(f"{{{ns}}}article")

    buffer = StringIO()
    write_structure(elem, buffer)

    output = buffer.getvalue()

    assert "article" in output
    assert ns in output
