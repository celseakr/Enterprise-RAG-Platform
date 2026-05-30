import chromadb
from chromadb.utils import embedding_functions

# 1. Initialise local persistent databse storage
print ("Initialsing ChromaDB local storage... ")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 2. Configure ChromaDB to use your local Ollama embedding engine
ollame_ef = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text"
)

# 3. Create or load a vector Collection
Collection = chroma_client.get_or_create_collection(
    name="my_first_collection",
    embedding_function=ollame_ef
)

# 4. Dummy text data to turn into mathematical vectors
sample_documents = [
    "Python is a versatile programming language widely used for AI and data science.", "The Golden Retriever is a popular breed of dog known for its friendly temperament.", "Vector databases like ChromaDB store embeddings to allow fast semantic search.", "A standard pizza crust is topped with tomato sauce, mozzarella cheese, and basil.", "Retrieval-Augmented Generation helps LLMs access external private facts safely."
]

# Create unique IDs for each text snippet
document_ids = [f"id{x}"for x in range(len(sample_documents))]

# 5. Convert text to vectors and save to database
print("Generating vector embeddings and saving to chromaDB...")
Collection.add(
    documents =sample_documents,
    ids = document_ids
)
print("Vector database successfully built!")

# ========================================== # MILESTONE: RUN SEMANTIC SEARCH QUERIES # ========================================== 
# The database will find matching concepts even if the exact words don't match.

user_query = "Tell me about building Ai systems or managing data"

print(f"\n Running semantic search query: '{user_query}' ")

search_results = Collection.query(
    query_texts=[user_query],
    n_results=2
)

for i, doc in enumerate(search_results["documents"][0]):
    print(f"{i+1}. {doc}")

print("\n Top Relevant Matches Found in Database")
for match in search_results['documents'][0]:
    print(f"--> {match}")