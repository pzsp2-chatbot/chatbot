import os
import json
import xml.etree.ElementTree as ET

INPUT_DIR = "data/omega_data_oct2024_xml"
OUTPUT_DIR = "data/json_clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NS = {"ns": "http://ii.pw.edu.pl/lib"}

def get_text(elem, path):
    el = elem.find(path, NS)
    return el.text.strip() if el is not None and el.text else None


def normalize_doi(doi):
    """
    Transform DOI to URL.
    """
    if not doi:
        return None
    doi = doi.strip()
    if doi.startswith("http"):
        return doi
    return f"https://doi.org/{doi}"


def parse_article(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # --- METADATA ---
    article_id = get_text(root, "./id")
    title = get_text(root, "./title")
    created = get_text(root, "./metaData/created")
    modified = get_text(root, "./metaData/lastModified")
    language = get_text(root, "./ns:language/code")
    doi = get_text(root, "./doi")

    # normalized DOI → URL
    doi_url = normalize_doi(doi)

    # use doi_url if exists
    # else fallback to <url>
    raw_url = get_text(root, "./url")
    final_url = doi_url if doi_url else raw_url

    # --- AUTHORS ---
    authors = []
    for a in root.findall(".//ns:author", NS):
        name = get_text(a, "./name")
        surname = get_text(a, "./surname")
        full_name = f"{name} {surname}" if name and surname else None

        affiliation = (
            get_text(a, "./externalAuthorAffiliation/fullNameEN")
            or get_text(a, "./externalAuthorAffiliation/fullName")
        )

        authors.append({
            "full_name": full_name,
            "affiliation": affiliation
        })

    # --- ABSTRACTS ---
    possible_pl = [
        ".//abstract_pl",
        ".//streszczenie",
        ".//description_pl",
    ]
    possible_en = [
        ".//abstract_en",
        ".//description",
        ".//streszczenie_en",
    ]

    abstract_pl = None
    for path in possible_pl:
        e = root.find(path)
        if e is not None and e.text:
            abstract_pl = e.text.strip()
            break

    abstract_en = None
    for path in possible_en:
        e = root.find(path)
        if e is not None and e.text:
            abstract_en = e.text.strip()
            break

    # --- FINAL PAYLOAD ---
    return {
        "id": article_id,
        "title": title,
        "created": created,
        "modified": modified,
        "language": language,
        "doi": doi,
        "url": final_url,
        "authors": authors,
        "abstract_pl": abstract_pl,
        "abstract_en": abstract_en
    }


def process_all():
    jsonl_path = os.path.join(OUTPUT_DIR, "articles.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for filename in os.listdir(INPUT_DIR):
            if not filename.endswith(".xml"):
                continue

            xml_path = os.path.join(INPUT_DIR, filename)
            print(f"Processing: {xml_path}")

            data = parse_article(xml_path)

            # save individual json
            out_path = os.path.join(OUTPUT_DIR, filename.replace(".xml", ".json"))
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

            # append to jsonl
            jf.write(json.dumps(data, ensure_ascii=False) + "\n")

    print(f"All articles processed. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_all()
