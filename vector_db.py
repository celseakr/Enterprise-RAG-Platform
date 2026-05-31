import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import nltk
nltk.download ('punkt_tab') #a pretrained model that knows how to split text into sentences
from nltk.tokenize import  sent_tokenize

# 1. Initialise local persistent databse storage
print ("Initialsing ChromaDB local storage... ")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Configure ChromaDB to use your local Ollama embedding engine
ollame_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text"
)

# 3. Create or load a vector Collection
collection = chroma_client.get_or_create_collection(
    name="my_first_collection",
    embedding_function=ollame_ef
)

# Read pdf
def load_pdf(file_path):
    reader = PdfReader(file_path) #uses PDFreader from pydf to open the PDF

    text = "" #creates an empty string, this will store all extracted text from the PDF
    for page in reader.pages: #loops through each page in pdf
        page_text = page.extract_text() #extract text from each page
        if page_text:  #check if text exists, to avoid errors, only continue if text was successfully extracted
            text += page_text + "\n"  #add the page text to full text, then add a new line (\n) so pages dont merge together
    return text
# In simple terms, This function:
# opens a PDF
# reads every page
# extracts text
# combines everything into one big text output

# Chunk texts by sentences
def chunk_text(text, max_char=500): #input:big text, output: smaller texts, each chunk is a max of 500 characters
    sentences=sent_tokenize(text)
    chunks = []
    current_chunk = ""  #temporary storage while working, add chunks here until they get to 500 max chars and then save it to chunks

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_char: #if adding the sentence doesnt exceed 500 characters
            current_chunk+= " " + sentence #keep adding to it, append sentence into current chunk
        else:
            chunks.append(current_chunk.strip()) #if it doesnt fit, save current chunk
            current_chunk = sentence #start new chunk with this sentence

    if current_chunk: #if there's still a last chunk after the loop ends, save that chunk
        chunks.append(current_chunk.strip())

    return chunks
    

print ("Loading PDF....")

pdf_text =load_pdf("pdf_dataset.pdf")

print("Chunking text....")

sample_documents = chunk_text(pdf_text)

print(f"Total chunks created: {len(sample_documents)}")

# for debugging purposes, just to know if my chunking works properly
for i, chunk in enumerate(sample_documents[:3]): #takes the first 3 chunks and adds an index with enumerate
    print(f"\n CHUNK{i}:") #adds the index
    print(chunk) #adds the chunk content

# Create unique IDs for each text snippet
document_ids = [f"id{x}"for x in range(len(sample_documents))]

# 5. Convert text to vectors and save to database
print("Generating vector embeddings and saving to chromaDB...")
collection.add(
    documents =sample_documents,
    ids = document_ids
)

print(len(sample_documents))
print("Vector database successfully built!")

# ========================================== # MILESTONE: RUN SEMANTIC SEARCH QUERIES # ========================================== 
# The database will find matching concepts even if the exact words don't match.

user_query = "donors request beneficiary data"

print(f"\n Running semantic search query: '{user_query}' ")

search_results = collection.query(
    query_texts=[user_query],
    n_results=2,
    include=["documents", "distances"]
)

#Check for distance
for doc, dist in zip( #zip is a python function that takes two or more lists and pairs them together
    search_results["documents"][0], #the 0 here brings out only the result for the first query, query indexed 0, this is important if there were many queries
    search_results["distances"][0]
):
    print(dist, doc[:200]) #show only the first 200 chars in the document

print("-" * 50) #just a divider
print("\n Top Relevant Matches Found in Database")
for i, doc in enumerate(search_results["documents"][0], 1): #adds an index to the results, starts at 1
    print(f"{i}. {doc}")