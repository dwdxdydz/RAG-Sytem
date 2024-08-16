import transformers
import sentence_transformers
import faiss
import numpy as np
import PyPDF2
import pytesseract
from PIL import Image

# Model and tokenizer
model_name = 'facebook/opt-13b'
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Embedding model
embedding_model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(texts):
    return embedding_model.encode(texts, convert_to_tensor=True)

# Vector database
dimension = 384
index = faiss.IndexFlatL2(dimension)

def add_to_index(embeddings):
    faiss.normalize_L2(embeddings)
    index.add(embeddings.numpy())

def search_index(query_embedding, k=5):
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding.numpy(), k)
    return distances, indices

# PDF processing
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    # Implement text cleaning logic here
    # Remove headers, footers, and other irrelevant information
    return cleaned_text

def pdf_to_embeddings(pdf_path, chunk_size=512):
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = preprocess_text(text)
    text_chunks = [cleaned_text[i:i+chunk_size] for i in range(0, len(cleaned_text), chunk_size)]
    embeddings = get_embeddings(text_chunks)
    return embeddings

# RAG system
class RAGSystem:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer
        self.index = index
        self.embedding_model = embedding_model
        self.pdf_embeddings = {}  # Store embeddings for efficient access

    def add_pdf_to_index(self, pdf_path):
        embeddings = pdf_to_embeddings(pdf_path)
        self.pdf_embeddings[pdf_path] = embeddings  # Store embeddings for later use
        add_to_index(embeddings)

    def generate_response(self, query):
        query_embedding = get_embeddings([query])
        distances, indices = search_index(query_embedding[0])

        # Retrieve relevant PDF chunks
        relevant_pdf_chunks = []
        for i in indices:
            for pdf_path, embeddings in self.pdf_embeddings.items():
                if i in embeddings:
                    chunk_index = np.where(embeddings == embeddings[i])[0][0]
                    start_index = chunk_index * chunk_size
                    end_index = min(start_index + chunk_size, len(self.pdf_embeddings[pdf_path]))
                    relevant_pdf_chunks.append(self.pdf_embeddings[pdf_path][start_index:end_index])
                    break

        # Combine relevant chunks into a single text
        combined_text = ' '.join(relevant_pdf_chunks)

        # Create input for LLM
        input_text = f"Context: {combined_text}\nQuestion: {query}"
        inputs = self.tokenizer(input_text, return_tensors='pt')

        # Generate response
        outputs = self.model.generate(inputs['input_ids'], max_length=150)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Example usage
rag_system = RAGSystem()
rag_system.add_pdf_to_index('your_pdf_file.pdf')
response = rag_system.generate_response("What is the main topic of the PDF?")
print(response)
