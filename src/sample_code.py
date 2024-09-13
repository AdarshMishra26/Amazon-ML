import pandas as pd
import pytesseract
from PIL import Image
import os
import re
import constants
from utils import download_images
from sanity import sanity_check

# Load the sample test data
sample_test_df = pd.read_csv('dataset/sample_test.csv')

# Download images
download_images(sample_test_df['image_link'], 'images/')

# Function to perform OCR on an image
def perform_ocr(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Perform OCR on all images
ocr_results = []
for index, row in sample_test_df.iterrows():
    image_path = f'images/{index}.jpg'
    if os.path.exists(image_path):
        text = perform_ocr(image_path)
        ocr_results.append(text)
    else:
        ocr_results.append("")

sample_test_df['ocr_text'] = ocr_results

# Function to extract entity values from OCR text
def extract_entity_value(ocr_text, entity_name):
    patterns = {
        "width": r"(\d+\.?\d*)\s?(centimetre|foot|inch|metre|millimetre|yard)",
        "depth": r"(\d+\.?\d*)\s?(centimetre|foot|inch|metre|millimetre|yard)",
        "height": r"(\d+\.?\d*)\s?(centimetre|foot|inch|metre|millimetre|yard)",
        "item_weight": r"(\d+\.?\d*)\s?(gram|kilogram|microgram|milligram|ounce|pound|ton)",
        "maximum_weight_recommendation": r"(\d+\.?\d*)\s?(gram|kilogram|microgram|milligram|ounce|pound|ton)",
        "voltage": r"(\d+\.?\d*)\s?(kilovolt|millivolt|volt)",
        "wattage": r"(\d+\.?\d*)\s?(kilowatt|watt)",
        "item_volume": r"(\d+\.?\d*)\s?(centilitre|cubic foot|cubic inch|cup|decilitre|fluid ounce|gallon|imperial gallon|litre|microlitre|millilitre|pint|quart)"
    }

    pattern = patterns.get(entity_name, "")
    match = re.search(pattern, ocr_text)
    if match:
        value = match.group(1)
        unit = match.group(2)
        return f"{value} {unit}"
    return ""

# Extract entity values for each row
sample_test_df['prediction'] = sample_test_df.apply(lambda row: extract_entity_value(row['ocr_text'], row['entity_name']), axis=1)

# Create the output DataFrame
output_df = sample_test_df[['index', 'prediction']]

# Save the output to a CSV file
output_df.to_csv('sample_test_out.csv', index=False)

# Run the sanity checker
sanity_check('dataset/sample_test.csv', 'sample_test_out.csv')
