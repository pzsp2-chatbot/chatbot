import requests
import os
import time
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

BASE_URL = "https://omega-rd.ii.pw.edu.pl/seam/resource/rest/accesspoint/search"
OUTPUT_DIR = "data/omega_data_oct2024_xml"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 100
START_DATE = datetime(2024, 10, 1, tzinfo=timezone.utc)
END_DATE = datetime(2024, 11, 1, tzinfo=timezone.utc)
NS = {"ns2": "http://ii.pw.edu.pl/lib"}

def fetch_batch(offset, limit):
    """Fetch a batch of articles from the REST API."""
    query = f"article/createdDate%3E%27{START_DATE.date()}%27"
    url = f"{BASE_URL}/{query}/{offset}/{limit}?orderBy=createdDate;ascending"

    print("Loading: ", url)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.text

def parse_batch(xml_text):
    """Parse XML text and return list of article elements."""
    root = ET.fromstring(xml_text)
    articles = root.findall(".//ns2:article", NS)
    return articles

def extract_created(article):
    """Extract creation datetime from article's metaData/created as UTC-aware datetime."""
    created_el = article.find(".//metaData/created")
    if created_el is None or not created_el.text:
        return None
    try:
        dt = datetime.fromisoformat(created_el.text.replace("Z", "+00:00"))
        return dt  # already timezone-aware
    except Exception:
        return None

def download():
    offset = 0
    saved = 0

    while offset < BATCH_SIZE:
        try:
            xml_data = fetch_batch(offset, BATCH_SIZE)
        except requests.HTTPError as e:
            print("No data or error: ", e)
            break

        if "<collection" not in xml_data:
            print("End of data reached.")
            break

        articles = parse_batch(xml_data)

        if not articles:
            print("No more articles found.")
            break

        print(f"Batch has {len(articles)} articles")

        for art in articles:
            created = extract_created(art)
            if not created:
                continue

            if not (START_DATE <= created < END_DATE):
                continue

            art_id_el = art.find("ns2:id", NS)
            art_id = art_id_el.text if art_id_el is not None else f"noid_{saved}"

            filename = os.path.join(OUTPUT_DIR, f"{art_id}.xml")
            with open(filename, "wb") as f:
                f.write(ET.tostring(art, encoding="utf-8"))

            print(f"Saved: {filename}")
            saved += 1

        offset += BATCH_SIZE
        time.sleep(0.2)

    print(f"Downloaded {saved} articles.")

if __name__ == "__main__":
    download()
