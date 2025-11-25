import os
from datetime import datetime, timezone
from typing import List
import requests
import xml.etree.ElementTree as ET


class OmegaDownloadError(Exception):
    """Raised when downloading from Omega REST API fails."""
    pass


class OmegaDownloader:
    BASE_URL = "https://omega-rd.ii.pw.edu.pl/seam/resource/rest/accesspoint/search"
    NS = {"ns2": "http://ii.pw.edu.pl/lib"}

    def __init__(
        self,
        output_dir: str,
        batch_size: int = 100,
        limit: int = 2000,
        start_date: datetime = datetime(2024, 10, 1, tzinfo=timezone.utc)
    ):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.start_date = start_date
        self.limit = limit

        os.makedirs(self.output_dir, exist_ok=True)


    def fetch_batch(self, start: int, end: int) -> str:
        query = f"article/createdDate%3E%27{self.start_date.date()}%27"
        url = f"{self.BASE_URL}/{query}/{start}/{end}?orderBy=createdDate;ascending"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise OmegaDownloadError(f"Failed to fetch batch at offset {offset}: {e}")


    def parse_batch(self, xml_text: str) -> List[ET.Element]:
        if "<collection" not in xml_text:
            raise OmegaDownloadError("No <collection> tag in XML (likely end of data).")

        try:
            root = ET.fromstring(xml_text)
            articles = root.findall(".//ns2:article", self.NS)
            if not articles:
                raise OmegaDownloadError("XML contains <collection> but no <article> entries.")
            return articles
        except ET.ParseError as e:
            raise OmegaDownloadError(f"Failed to parse XML: {e}")


    def save_article(self, article: ET.Element, counter: int) -> None:
        art_id_el = article.find("ns2:id", self.NS)
        article_id = art_id_el.text if art_id_el is not None else f"noid_{counter}"
        filename = os.path.join(self.output_dir, f"{article_id}.xml")

        with open(filename, "wb") as f:
            f.write(ET.tostring(article, encoding="utf-8"))


    def download(self) -> int:
        saved = 0

        while saved < self.limit:
            batch_xml = self.fetch_batch(saved, saved+self.batch_size)
            articles = self.parse_batch(batch_xml)

            for art in articles:
                self.save_article(art, saved)
                saved += 1

        return saved


    def handle_download(self):
        try:
            total = self.download()
            print(f"\nDownloaded {total} articles.")
        except OmegaDownloadError as e:
            print(f"[ERROR] {e}")
        except Exception as e:
            print(f"[Unexpected Error] {e}")



if __name__ == "__main__":
    downloader = OmegaDownloader(
        output_dir="data/omega_data_nov2024_xml",
        batch_size=100,
        limit=500,
        start_date=datetime(2024, 11, 1, tzinfo=timezone.utc),
    )
    downloader.handle_download()
