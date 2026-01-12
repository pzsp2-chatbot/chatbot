from data_preparation.xml_to_json_batch import XMLParser, AuthorProcessor


def test_author_without_surname_is_skipped(tmp_path):
    xml = """
    <root xmlns:ns="http://ii.pw.edu.pl/lib">
        <ns:author>
            <name>Jan</name>
        </ns:author>
    </root>
    """
    path = tmp_path / "test.xml"
    path.write_text(xml)

    parser = XMLParser(str(path), {"ns": "http://ii.pw.edu.pl/lib"})
    authors = AuthorProcessor(parser).extract_authors()

    assert authors == []


def test_author_without_affiliation(tmp_path):
    xml = """
    <root xmlns:ns="http://ii.pw.edu.pl/lib">
        <ns:author>
            <name>Anna</name>
            <surname>Nowak</surname>
        </ns:author>
    </root>
    """
    path = tmp_path / "test.xml"
    path.write_text(xml)

    parser = XMLParser(str(path), {"ns": "http://ii.pw.edu.pl/lib"})
    authors = AuthorProcessor(parser).extract_authors()

    assert authors[0]["affiliation"] is None


def test_author_full_extraction(tmp_path):
    xml = """
    <root xmlns:ns="http://ii.pw.edu.pl/lib">
        <ns:author>
            <name>Piotr</name>
            <surname>Kowalski</surname>
            <externalAuthorAffiliation>
                <fullName>University of Warsaw</fullName>
            </externalAuthorAffiliation>
        </ns:author>
    </root>
    """
    path = tmp_path / "test.xml"
    path.write_text(xml)

    parser = XMLParser(str(path), {"ns": "http://ii.pw.edu.pl/lib"})
    authors = AuthorProcessor(parser).extract_authors()

    assert authors[0]["full_name"] == "Piotr Kowalski"
    assert authors[0]["affiliation"] == "University of Warsaw"


def test_author_with_multiple_affiliations(tmp_path):
    xml = """
    <root xmlns:ns="http://ii.pw.edu.pl/lib">
        <ns:author>
            <name>Maria</name>
            <surname>Wiśniewska</surname>
            <externalAuthorAffiliation>
                <fullName>Institute A</fullName>
            </externalAuthorAffiliation>
            <externalAuthorAffiliation>
                <fullName>Institute B</fullName>
            </externalAuthorAffiliation>
        </ns:author>
    </root>
    """
    path = tmp_path / "test.xml"
    path.write_text(xml)

    parser = XMLParser(str(path), {"ns": "http://ii.pw.edu.pl/lib"})
    authors = AuthorProcessor(parser).extract_authors()

    assert authors[0]["full_name"] == "Maria Wiśniewska"
    assert authors[0]["affiliation"] == "Institute A; Institute B"
