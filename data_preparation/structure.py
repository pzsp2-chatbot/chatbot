import xml.etree.ElementTree as ET

INPUT_XML = "data/omega_data_oct2024_xml/noid_0.xml"  # path to XML-article
OUTPUT_FILE = "data/article_structure.txt"

def write_structure(elem, file, indent=0):
    ind = "  " * indent
    tag_line = f"{ind}<{elem.tag}"
    
    if elem.attrib:
        attr_str = " ".join(f'{k}="{v}"' for k, v in elem.attrib.items())
        tag_line += f" {attr_str}"
    
    tag_line += ">"
    file.write(tag_line + "\n")
    
    for child in elem:
        write_structure(child, file, indent + 1)
    
    file.write(f"{ind}</{elem.tag}>\n")

def main():
    tree = ET.parse(INPUT_XML)
    root = tree.getroot()
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        write_structure(root, f)

    print(f"Structure of article is written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
