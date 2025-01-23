import streamlit as st
from streamlit import session_state as ss
from io import StringIO
import pdfplumber
import pandas as pd
import regex as re  # Use the `regex` module for advanced regular expressions
from streamlit_pdf_viewer import pdf_viewer
from openai import OpenAI
from PyPDF2 import PdfReader, PdfWriter
# from PyPDF2 import PdfReader
import base64
import random
import json
import importlib
import gmft
import gmft.table_detection
import gmft.table_visualization
import gmft.table_function
import gmft.table_function_algorithm
import gmft.table_captioning
import gmft.pdf_bindings.bindings_pdfium
import gmft.pdf_bindings
import gmft.common
import pandas as pd

importlib.reload(gmft)
importlib.reload(gmft.common)
importlib.reload(gmft.table_captioning)
importlib.reload(gmft.table_detection)
importlib.reload(gmft.table_visualization)
importlib.reload(gmft.table_function)
importlib.reload(gmft.table_function_algorithm)
importlib.reload(gmft.pdf_bindings.bindings_pdfium)
importlib.reload(gmft.pdf_bindings)

from gmft.pdf_bindings import PyPDFium2Document
from gmft.auto import CroppedTable, AutoTableDetector

detector = AutoTableDetector()

from gmft.auto import AutoTableFormatter

formatter = AutoTableFormatter()

from gmft.auto import AutoFormatConfig

config_hdr = AutoFormatConfig()  # config may be passed like so
config_hdr.verbosity = 3
config_hdr.enable_multi_header = True
config_hdr.semantic_spanning_cells = True  # [Experimental] Merge headers

api_key = st.secrets["openai_api_key"]

# Pass the API key directly when creating the client
client = OpenAI(api_key=api_key)

st.set_page_config(
    page_title="PDF Table Extractor",
    # page_icon="📄",
    layout="wide"  # Use the wide layout
)

# Mock FORMAT and Client setup
FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "yes_no_response",
        "schema": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "enum": ["Yes", "No"]
                }
            },
            "required": ["response"],
            "additionalProperties": False
        },
        "strict": True
    }
}

JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "Tables_extracted",
        "schema": {
            "type": "object",
            "properties": {"tables": {"type": "object"}},
            "required": ["tables"],  # Ensure this matches the properties
            "additionalProperties": False,
        },
        "strict": False,
    },
}

def handle_complex_tables(pdf):
    assistant = client.beta.assistants.create(
        name="Table Extractor",
        instructions="You are an expert at extracting tables from pdfs",
        model="gpt-4o",
        response_format=JSON_SCHEMA,
        tools=[{"type": "file_search"}],
    )

    message_file = client.files.create(file=open(pdf, "rb"), purpose="assistants")
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "Please extract the data as csv from the given PDF. If the pdf has many tables, extract them differently. If any table spans across different pages, extract the text from all those pages and give as one csv. Keep the original order of rows and columns and include every row of each page. Please ensure that all values, even blank cells, match the original document exactly. Separate each cell by pipe (|). Analyze the table schema, including column names and data patterns, ensuring consistency with the column types and formatting. Do not include any other text outside the table",
                "attachments": [{"file_id": message_file.id, "tools": [{"type": "file_search"}]}],
            }
        ]
    )

    run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.id)
    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    table_output = messages[0].content[0].text.value
    json_object = json.loads(table_output)
    tables = json_object['tables']
    return tables

def extract_pages(input_pdf, output_pdf, pages):
    """
    Extract specific pages from a PDF and save them to a new PDF.

    Args:
    input_pdf (str): Path to the input PDF file.
    output_pdf (str): Path to the output PDF file.
    pages (list): List of page numbers to extract (0-indexed).
    """
    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    # Add specified pages to the writer
    for page_num in pages:
        writer.add_page(reader.pages[page_num-1])

    # Write to the output file
    with open(output_pdf, "wb") as output_file:
        writer.write(output_file)


# Dummy function for chat API call (replace with your actual API integration)
def is_table_valid(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        response_format=FORMAT,
        # max_tokens=200,
        temperature=0
    )
    output = response.choices[0].message.content.strip()
    output_dict = json.loads(output)
    return output_dict


# PDF ingestion logic
def extract_all_tables(pdf_path) -> tuple:
    """
        Extracts all tables from a given PDF file.
    """
    doc = PyPDFium2Document(pdf_path)

    tables = []
    table_pages = []
    for i, page in enumerate(doc):
        extracted_tables = detector.extract(page)
        if extracted_tables:
            tables += extracted_tables
            table_pages += [i + 1] * len(extracted_tables)

    return tables, table_pages

# Streamlit app
def pdf_viewer(input, width=700, height=800):
    # Convert binary data to base64 for embedding
    base64_pdf = base64.b64encode(input).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="{width}" height="{height}" style="border:none;"></iframe>'
    st.components.v1.html(pdf_display, height=height + 50)

def process_tables(complicated_pages):
    # print(f"The complicated pages are {complicated_pages}")
    extract_pages("temp.pdf", "temp_2.pdf", complicated_pages)
    # tables = {'Table 1': 'Resident Name|Order Description|Order Directions|Order Category|Start Date|End Date|Indications for Use\nBURROLA, ADIS (MH100797)|Aplisol Solution 5 UNIT/0.1ML (Tuberculin PPD)|Inject 0.1 ml intradermally in the evening every 365 day(s) for TB Screening Every evening shift annually. Record result within 72 hours. Perform Chest X-Ray if patient refused|Pharmacy|03/02/2024||TB Screening\nBURROLA, ADIS (MH100797)|Atorvastatin Calcium Oral Tablet 20 MG (Atorvastatin Calcium)|Give 1 tablet by mouth at bedtime for hyperlipidemia|Pharmacy|07/19/2024||hyperlipidemia\nBURROLA, ADIS (MH100797)|Cholecalciferol Tablet 1000 UNIT|Give 2 tablet by mouth one time a day for supplement|Pharmacy|08/31/2024||supplement\nBURROLA, ADIS (MH100797)|hydrALAZINE HCl Oral Tablet 25 MG (Hydralazine HCl)|Give 1 tablet by mouth every 6 hours as needed for SBP>170.|Pharmacy|03/01/2024||SBP>170.\nBURROLA, ADIS (MH100797)|Xarelto Oral Tablet 10 MG (Rivaroxaban)|Give 1 tablet by mouth one time a day for DVT prophylaxis|Pharmacy|10/08/2023||DVT prophylaxis\nCAMPBELL, ROBERT B (MH100302)|AmLODIPine Besylate Tablet 5 MG|Give 1 tablet by mouth one time a day for HTN Hold for SBP <100 or HR <60|Pharmacy|10/18/2024||HTN\nCAMPBELL, ROBERT B (MH100302)|Dulcolax Suppository 10 MG (Bisacodyl)|Insert 1 dose rectally every 24 hours as needed for Constipation PRN If MOM ineffective|Pharmacy|10/17/2024||Constipation\nCAMPBELL, ROBERT B (MH100302)|Enoxaparin Sodium Injection Solution Prefilled Syringe 40 MG/0.4ML (Enoxaparin Sodium)|Inject 40 mg subcutaneously in the morning for DVT PROPHYLAXIS|Pharmacy|10/18/2024||DVT PROPHYLAXIS\nCAMPBELL, ROBERT B (MH100302)|Famotidine Tablet 20 MG|Give 1 tablet by mouth two times a day for GERD|Pharmacy|10/17/2024||GERD\nCAMPBELL, ROBERT B (MH100302)|Fleet Enema Enema 7-19 GM/118ML (Sodium Phosphates)|Insert 1 dose rectally every 24 hours as needed for Constipation If Dulcolax is ineffective|Pharmacy|10/17/2024||Constipation\nCAMPBELL, ROBERT B (MH100302)|Flomax Capsule 0.4 MG (Tamsulosin HCl)|Give 1 capsule by mouth at bedtime for benign prostatic hyperplasia|Pharmacy|10/17/2024||benign prostatic hyperplasia\nCAMPBELL, ROBERT B (MH100302)|Folic Acid Tablet 1 MG|Give 1 tablet by mouth one time a day for SUPPLEMENT|Pharmacy|10/18/2024||SUPPLEMENT\nCAMPBELL, ROBERT B (MH100302)|Furosemide Oral Tablet 40 MG (Furosemide)|Give 1 tablet by mouth two times a day for CHRONIC BLE|Pharmacy|10/17/2024||CHRONIC BLE\nCAMPBELL, ROBERT B (MH100302)|Lidocaine Patch 4 %|Apply to right groin topically in the morning for pain management and remove per schedule|Pharmacy|10/18/2024||pain management\nCAMPBELL, ROBERT B (MH100302)|Melatonin Tablet 3 MG|Give 2 tablet by mouth at bedtime for CIRCADIAN RHYTHM|Pharmacy|10/17/2024||CIRCADIAN RHYTHM\nCAMPBELL, ROBERT B (MH100302)|Milk of Magnesia Suspension 1200 MG/15ML (Magnesium Hydroxide)|Give 30 ml by mouth every 24 hours as needed for Constipation|Pharmacy|10/17/2024||Constipation\nCAMPBELL, ROBERT B (MH100302)|Polyethylene Glycol 3350 Powder (Polyethylene Glycol 3350 (Bulk))|Give 1 scoop by mouth in the morning for bowel management mix 120 ml water/tea/coffee/juice|Pharmacy|10/18/2024||bowel management\nCAMPBELL, ROBERT B (MH100302)|Senna Oral Tablet (Sennosides)|Give 8.6 mg by mouth two times a day for constipation hold for loose bm|Pharmacy|10/17/2024||constipation\nCAMPBELL, ROBERT B (MH100302)|Silver Sulfadiazine Cream 1 %|Apply to L lateral foot/L shin topically every day shift for skin and wound. Cleanse with NSS, pat dry. Apply medicated cream f/b xrfm. then wrap with rolled gauze, compression with ace bandage.|Pharmacy|10/24/2024||skin and wound.\nCAMPBELL, ROBERT B (MH100302)|traZODone HCl Oral Tablet (Trazodone HCl)|Give 25 mg by mouth at bedtime every other day for Depression m/b inability to sleep >6 hours|Pharmacy|10/17/2024||Depression m/b inability to sleep >6 hours\nCAMPBELL, ROBERT B (MH100302)|Tylenol Tablet 325 MG (Acetaminophen)|Give 2 tablet by mouth every 6 hours as needed for Pain - Mild DO NOT EXCEED 3000MG IN 24HRS 1-Reposition 2-Dim lights/Quiet environment 3-Hot/Cold Applications 4-Relaxation Techniques 5-Distraction 6-Music 7-Massage 8-Other Document result/effectiveness or non-pharm int: E-Effective N-Not effective|Pharmacy|10/17/2024||Pain - Mild\nCASAUS, ANNE M. (MH100022)|Aspirin 81 Oral Tablet Chewable (Aspirin)|Give 1 tablet by mouth one time a day for CVA prophylaxis|Pharmacy|03/17/2024||CVA prophylaxis\nCASAUS, ANNE M. (MH100022)|Atorvastatin Calcium Oral Tablet 20 MG (Atorvastatin Calcium)|Give 1 tablet by mouth at bedtime for hyperlipidemia|Pharmacy|03/16/2024||hyperlipidemia\nCASAUS, ANNE M. (MH100022)|Austedo Oral Tablet (Deutetrabenazine)|Give 12 mg by mouth two times a day for Tardive dyskinesia|Pharmacy|04/02/2024||Tardive dyskinesia\nCASAUS, ANNE M. (MH100022)|Bisacodyl Oral Tablet Delayed Release 5 MG (Bisacodyl)|Give 1 tablet by mouth at bedtime for bowel management|Pharmacy|06/08/2024||bowel management\n'}
    tables = handle_complex_tables("temp_2.pdf")
    table_number = 1

    for table_name, table_content in tables.items():
        # Split the content into rows and then columns
        rows = table_content.strip().split("\n")
        table_data = [row.split("|") for row in rows]

        # Create a DataFrame
        df = pd.DataFrame(table_data)
        df.columns = df.iloc[0]  # Set the first row as column headers
        df = df[1:]  # Remove the first row after setting it as headers

        # Display the table with its number
        st.write(f"Table {table_number}")
        st.dataframe(df)

        # Increment the table number
        table_number += 1

def main_interface():
    st.title("PDF Table Extractor")
    complicated_pages=[]

    # File uploader
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    # Process PDF if uploaded
    if pdf_file is not None:
        # binary_data = pdf_file.getvalue()
        st.write("Preview of the uploaded PDF:")
        # pdf_viewer(input=binary_data)

    # Process PDF if uploaded
    if pdf_file:
        st.write("Processing PDF...")
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())

        # Extract first table and validate
        tables, table_pages = extract_all_tables("temp.pdf")
        if tables:
            st.write("Simple Tables")
            for i, table in enumerate(tables):
                # if i == 0:
                # Format the table for validation
                ft = formatter.extract(table)
                try:
                    csv_data = ft.df(config_overrides=config_hdr)
                except ValueError as ve:
                    print(f"The PDF tables are complicated. Use the premium version.")
                    break
                except Exception as e:
                    print(f"The PDF tables are complicated. Use the premium version.")
                    break

                csv_data = csv_data.dropna(axis=1, how="all")

                if csv_data.columns.duplicated().any():
                    csv_data.columns = [f"Column_{j + 1}" for j in range(csv_data.shape[1])]

                def clean_text(text):
                    if isinstance(text, str):
                        text = text.replace('"', '').replace('\n', ' ').replace('\t', ' ')
                    return text

                csv_data = csv_data.map(clean_text)

                rows_to_send = csv_data.head(5).to_dict(orient="records")

                prompt = f"""
                    The following data was extracted from a PDF file (Table {i + 1}):

                    First Five Rows:
                    {rows_to_send}

                    Analyze the structure and content of the data. Does it appear to be well-structured and properly extracted? Ignore the headers and small tables below 7 rows
                     *IMPORTANT*: For medical tables, respond with a NO. For chemistry and ANY OTHER TABLE, respond with a yes.  The text in the rows should be coherent and the column values should follow the same data pattern
                    Just respond with Yes/No. 
                   """
                # validation_response = is_table_valid(prompt)
                # validation_response = {"response": "Yes"}
                if len(csv_data) > 8:
                    validation_response = is_table_valid(prompt)
                else:
                    validation_response = {'response': 'Yes'}
                if validation_response["response"] != "Yes":
                    complicated_pages.append(table_pages[i])
                else:
                    st.write(f"Table {i + 1}")
                    st.dataframe(csv_data)

        if "retry" not in st.session_state:
            st.session_state.retry = False

        if complicated_pages:
            if not st.session_state.retry:
                st.write("Complicated Tables...")
                process_tables(complicated_pages)

            if st.button("Retry"):
                st.session_state.retry = True
                st.write("Reprocessing Tables...")
                process_tables(complicated_pages)


def main():
    # Check if the session state is set
    main_interface()


if __name__ == "__main__":
    main()
