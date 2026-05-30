import ollama
print("Connecting to your local Llama 3 model...")

try:
    response = ollama.chat(model='llama3', messages=[
        {
            'role': 'user',
            'content': 'Explain what a RAG chatbot is in one short sentence',
        },
    ])
    print ("\n Success! AI Response:")
    print (response ['message'] ['content'])
except Exception as e:
    print("\n Error connecting to Ollama. Make sure the app is open!")
    print(e)