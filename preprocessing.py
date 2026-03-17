## preprocessing.py
## Extract data (text, images, tables) from PDF documents


import os
import box, yaml
import sys, pathlib
import pymupdf, pymupdf4llm, pymupdf.layout
import pytesseract
from PIL import Image
from pprint import pprint

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Open a document
file_path = "data2/Llama2.pdf"
doc = pymupdf.open(file_path)
print(f"document type: {type(doc)}")

# Create an output folder
out_dir = 'data_output'
os.makedirs(out_dir, exist_ok=True) # Create a directory if does not exist

# Get Markdown Layout for the entire PDF:
md = pymupdf4llm.to_markdown(doc)
out_path = os.path.join(out_dir, f"output.md")
with open (out_path, 'w', encoding="utf-8") as file:
                    file.write(md)


def get_metadata(doc):
    """
    Get metadata from the document
    """
    return doc.metadata

def extract_text(doc):
    """
    extract plain text from PDF focuments and save into txt files.
    """
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text()
        out_path = os.path.join(out_dir, f"page_{page_index}.txt")
        with open(out_path, 'w', encoding='utf-8') as file:
            file.write(text)
    
def extract_images(doc):
    """
    extract images from PDF document and save into PNG images.
    """
    out_dir = 'data_output/images'
    os.makedirs(out_dir, exist_ok=True)
    for page_index in range(doc.page_count):
        page = doc[page_index]
        image_list = page.get_images()

        if image_list:
            print(f"Found {len(image_list)} images on page {page_index}")
        else:
            print("No images found on page", page_index)

        for image_index, img in enumerate(image_list, start=1):
            xref = img[0]                       # xref number
            pix = pymupdf.Pixmap(doc, xref)     # make Pixmap from image

            if pix.n - pix.alpha > 3:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            
            out_path = os.path.join(out_dir, f"page_{page_index} - image{image_index}.png")
            pix.save(out_path)
            pix = None                          # free Pixmap resources

def extract_tables(doc):
    """
    extract tables from PDF document and save into Markdown files.
    """
    out_dir = 'data_output/tables'
    os.makedirs(out_dir, exist_ok=True)
    for page_index in range(doc.page_count):
        page = doc[page_index]
        tb = page.find_tables()

        if tb.tables:
            print(f"Found {len(tb.tables)} tables on page {page_index + 1}")
        else:
            print("No tables found on page", page_index)
        for table_index, tab in enumerate(tb.tables, start=1):
            print(f"table {table_index},{tab.page}")
            try:
                markdown_table = tab.to_markdown()
                out_path = os.path.join(out_dir, f"page_{page_index + 1}-table{table_index}.md")
                with open (out_path, 'w', encoding="utf-8") as file:
                    file.write(markdown_table)
            except:
                print(f"Error Converting table {table_index} on {tab.page}")

if __name__ == "__main__":
    # extract_images(doc)
    # extract_tables(doc)
    print(True)