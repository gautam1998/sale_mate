import chromadb
import PyPDF2
import re
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from langchain.docstore.document import Document

# Replace 'your_pdf_file.pdf' with the actual PDF file path
pdf_file_path = '/Users/igautam/Documents/Quarter5/capstone/saleMate/data/raw/Potential Scenarios For Best Buy.pdf'

def split_text_on_scenario(pdf_file_path):
    scenarios = []
    with open(pdf_file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        n = len(pdf_reader.pages)
        for page_num in range(n):
            page = pdf_reader.pages[page_num]
            #page = pdf_reader.getPage(page_num)
            text += page.extract_text()

        # Split the text into scenarios using the "Scenario" keyword
        text = re.sub(r'\n+', ' ', text)
        text = text.split("Scenario")

        # Remove the first empty element and strip each scenario
        for scenario in text[1:]:
            scenarios.append(scenario.strip())

    return scenarios

# Call the function to get a list of scenarios
scenarios = split_text_on_scenario(pdf_file_path)

documents = []

for i, scenario_text in enumerate(scenarios):

    doc = Document(
        page_content=scenario_text,
    )
    documents.append(doc)

emb_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=emb_model,
    cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
)

vector_db_folder = '/Users/igautam/Documents/Quarter5/capstone/saleMate/data/processed'
vector_db_path = os.path.join(vector_db_folder,
                              "scenarios_db")

db = Chroma.from_documents(documents,
                           embedding=embeddings,
                           persist_directory=vector_db_path)

db.persist()


# Run similarity search query
q = "I want to buy a laptop for gaming, can you help me out"
v = db.similarity_search(q, include_metadata=True)

print(v)