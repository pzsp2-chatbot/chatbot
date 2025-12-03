import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional


class XMLParser:
    def __init__(self, xml_file: str, namespace: Dict[str, str]):
        self.xml_file = xml_file
        self.namespace = namespace
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()

    def find_element(self, path: str) -> Optional[ET.Element]:
        return self.root.find(path, self.namespace)

    def find_all_elements(self, path: str) -> List[ET.Element]:
        return self.root.findall(path, self.namespace)

    def get_element_text(self, path: str) -> Optional[str]:
        element = self.find_element(path)
        return element.text.strip() if element is not None and element.text else None


class DOIProcessor:
    @staticmethod
    def normalize(doi: Optional[str]) -> Optional[str]:
        if not doi:
            return None

        cleaned_doi = doi.strip()
        if cleaned_doi.startswith("http"):
            return cleaned_doi
        return f"https://doi.org/{cleaned_doi}"


class AuthorProcessor:
    def __init__(self, parser: XMLParser):
        self.parser = parser

    def extract_authors(self) -> List[Dict[str, Optional[str]]]:
        authors = []
        author_elements = self.parser.find_all_elements(".//ns:author")

        for author_element in author_elements:
            author_data = self._extract_single_author(author_element)
            if author_data["full_name"]:
                authors.append(author_data)

        return authors

    def _extract_single_author(
        self, author_element: ET.Element
    ) -> Dict[str, Optional[str]]:
        name = author_element.findtext("name")
        surname = author_element.findtext("surname")
        full_name = f"{name} {surname}".strip() if name and surname else None

        affiliation = self._extract_affiliation(author_element)

        return {"full_name": full_name, "affiliation": affiliation}

    def _extract_affiliation(self, author_element: ET.Element) -> Optional[str]:
        return author_element.findtext(
            "externalAuthorAffiliation/fullNameEN"
        ) or author_element.findtext("externalAuthorAffiliation/fullName")


class ContentExtractor:
    def __init__(self, parser: XMLParser):
        self.parser = parser

    def get_language(self) -> Optional[str]:
        return self.parser.get_element_text("./ns:language/code")

    def get_abstract(self) -> Optional[str]:
        language = self.get_language()
        abstract_path = self._get_abstract_path(language)
        return self.parser.get_element_text(abstract_path)

    def get_keywords(self) -> Optional[str]:
        language = self.get_language()
        keywords_path = self._get_keywords_path(language)
        return self.parser.get_element_text(keywords_path)

    def _get_abstract_path(self, language: Optional[str]) -> str:
        if language == "en":
            return "./abstractEN"
        elif language == "pl":
            return "./abstractPL"
        else:
            return "./abstractEN"

    def _get_keywords_path(self, language: Optional[str]) -> str:
        if language == "en":
            return "./keywordsEN"
        elif language == "pl":
            return "./keywordsPL"
        else:
            return "./keywordsEN"


class ArticleMetadata:
    def __init__(
        self,
        article_id: Optional[str],
        title: Optional[str],
        created: Optional[str],
        modified: Optional[str],
        language: Optional[str],
        doi: Optional[str],
        url: Optional[str],
        authors: List[Dict[str, Optional[str]]],
        abstract: Optional[str],
        keywords: Optional[str],
    ):
        self.article_id = article_id
        self.title = title
        self.created = created
        self.modified = modified
        self.language = language
        self.doi = doi
        self.url = url
        self.authors = authors
        self.abstract = abstract
        self.keywords = keywords

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.article_id,
            "title": self.title,
            "created": self.created,
            "modified": self.modified,
            "language": self.language,
            "doi": self.doi,
            "url": self.url,
            "authors": self.authors,
            "abstract": self.abstract,
            "keywords": self.keywords,
        }


class ArticleParser:
    NAMESPACE = {"ns": "http://ii.pw.edu.pl/lib"}

    def __init__(self, xml_file: str):
        self.xml_file = xml_file
        self.parser = XMLParser(xml_file, self.NAMESPACE)
        self.doi_processor = DOIProcessor()
        self.author_processor = AuthorProcessor(self.parser)
        self.content_extractor = ContentExtractor(self.parser)

    def parse(self) -> ArticleMetadata:
        basic_metadata = self._extract_basic_metadata()
        authors = self.author_processor.extract_authors()
        abstract = self.content_extractor.get_abstract()
        keywords = self.content_extractor.get_keywords()

        return ArticleMetadata(
            article_id=basic_metadata["id"],
            title=basic_metadata["title"],
            created=basic_metadata["created"],
            modified=basic_metadata["modified"],
            language=basic_metadata["language"],
            doi=basic_metadata["doi"],
            url=basic_metadata["url"],
            authors=authors,
            abstract=abstract,
            keywords=keywords,
        )

    def _extract_basic_metadata(self) -> Dict[str, Optional[str]]:
        article_id = self.parser.get_element_text("./id")
        title = self.parser.get_element_text("./title")
        created = self.parser.get_element_text("./metaData/created")
        modified = self.parser.get_element_text("./metaData/lastModified")
        language = self.parser.get_element_text("./ns:language/code")
        doi = self.parser.get_element_text("./doi")

        doi_url = self.doi_processor.normalize(doi)
        raw_url = self.parser.get_element_text("./url")
        final_url = doi_url or raw_url

        return {
            "id": article_id,
            "title": title,
            "created": created,
            "modified": modified,
            "language": language,
            "doi": doi,
            "url": final_url,
        }


class JSONOutputHandler:
    @staticmethod
    def save_single_article(data: Dict[str, Any], output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    @staticmethod
    def append_to_jsonl(data: Dict[str, Any], jsonl_file) -> None:
        jsonl_file.write(json.dumps(data, ensure_ascii=False) + "\n")


class BatchProcessor:
    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_handler = JSONOutputHandler()
        self._ensure_output_directory()

    def _ensure_output_directory(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

    def process_all(self) -> None:
        jsonl_path = os.path.join(self.output_dir, "articles.jsonl")

        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for filename in self._get_xml_files():
                self._process_single_file(filename, jsonl_file)

        print(f"Processing complete. Output directory: {self.output_dir}")

    def _get_xml_files(self) -> List[str]:
        return [
            filename
            for filename in os.listdir(self.input_dir)
            if filename.endswith(".xml")
        ]

    def _process_single_file(self, filename: str, jsonl_file) -> None:
        try:
            xml_path = os.path.join(self.input_dir, filename)
            article_data = self._parse_article(xml_path)

            self._save_individual_json(article_data, filename)
            self.output_handler.append_to_jsonl(article_data, jsonl_file)

        except Exception as error:
            print(f"Error processing {filename}: {error}")

    def _parse_article(self, xml_path: str) -> Dict[str, Any]:
        parser = ArticleParser(xml_path)
        metadata = parser.parse()
        return metadata.to_dict()

    def _save_individual_json(self, data: Dict[str, Any], filename: str) -> None:
        json_filename = filename.replace(".xml", ".json")
        output_path = os.path.join(self.output_dir, json_filename)
        self.output_handler.save_single_article(data, output_path)


def main():
    INPUT_DIRECTORY = "data/omega_data_nov2024_xml"
    OUTPUT_DIRECTORY = "data/json_clean_nov2024"

    processor = BatchProcessor(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    processor.process_all()


if __name__ == "__main__":
    main()
