from data_preparation.xml_to_json_batch import DOIProcessor 
                                                
def test_doi_normalize_strips_whitespace():
    doi = " 10.1000/xyz "
    assert DOIProcessor.normalize(doi) == "https://doi.org/10.1000/xyz"


def test_doi_normalize_empty_string():
    assert DOIProcessor.normalize("") is None

def test_doi_normalize_none_input():
    assert DOIProcessor.normalize(None) is None
