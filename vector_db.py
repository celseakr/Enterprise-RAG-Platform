import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

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

# Chunk texts
def chunk_text(text, chunk_size=500, overlap =50): #takes a text and splits it into chunks of 500 characters with 50 characters overlap
    chunks=[] # this will store all the smaller pieces of texts
    for i in range(0, len(text), chunk_size - overlap): #loop through text (start, end, step), step is like where you start each time so like 0,450,900 etc.Start at 450 jumps each time.
        chunk = text[i:i+chunk_size] # [0:0+500] and then because of the step next iteration is [450:450+500] and then [900:1400].. (this is so that there's an overlap in the chunks so that the meaning doesnt get lost)
        chunks.append(chunk) # add these chunks to the list

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

user_query = "Whom do donors regulary request data from?"

print(f"\n Running semantic search query: '{user_query}' ")

search_results = collection.query(
    query_texts=[user_query],
    n_results=3
)

print("\n Top Relevant Matches Found in Database")
for i, doc in enumerate(search_results["documents"][0], 1): #adds an index to the results, starts at 1
    print(f"{i}. {doc}")