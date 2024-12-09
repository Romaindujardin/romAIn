import requests
import streamlit as st
from huggingface_hub import HfFolder
from sentence_transformers import SentenceTransformer
import faiss

# Récupérer le token Hugging Face depuis Streamlit Secrets
hf_token = st.secrets["huggingface"]["token"]

# Initialisation du modèle d'embedding et de l'index FAISS
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modèle rapide et léger
documents = [
    "My name is Romain Dujardin",
    "I'm 22 years old",
    "I'm a French student in AI engineer",
    "I study at Isen JUNIA in Lille since 2021, During my studies, I have learned about machine learning, deep learning, computer vision, natural language processing, reinforcement learning. I had lessons in mathematics, statistics, computer science, physics, electronics and project management",
    "Before Isen JUNIA, I was at ADIMAKER, an integrated preparatory class where I learned the basics of engineering",
    "I'm passionate about artificial intelligence, new technologies and computer science",
    "I'm based in Lille, France",
    "I have work on different project during my studies, like Project F.R.A.N.K who is a 3d project mixing AI on unity3D it is a horror game in a realistic universe, with advanced gameplay functions such as inventory management and item usage, all while being pursued by a monster under AI. And i have also worked on a local drive project on django named DriveMe. all this project are available on my github",
    "I'm currently looking for an internationally internship in AI, starting in April 2025",
    "My email is dujardin.romain@icloud.com and My phone number is 07 83 19 30 23",
    "I had professional experience as a pharmaceutical driver, accountant, machine operator or food truck clerk",
    "I have a driving license",
    "I have a BAC STI2D with honors.",
    "I code in python, django, react and I master tools like rag, hyde, pytorsh"
]

# Créer des embeddings pour chaque document
doc_embeddings = embedding_model.encode(documents)

# Créer un index FAISS
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Fonction pour rechercher les documents pertinents
def find_relevant_docs(query, k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [documents[idx] for idx in indices[0]]

# Fonction pour utiliser Mistral via l'API
def mistral_via_api(prompt):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    
    if hf_token is None:
        return "Error: No tokens found. Log in with `huggingface-cli login`."

    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.4,
            "top_k": 50,
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error : {response.status_code} - {response.json()}"

# Pipeline RAG - Combiner recherche et génération
def rag_pipeline(query, k=4):
    relevant_docs = find_relevant_docs(query, k)
    context = "\n".join(relevant_docs)
    prompt = f"Context : {context}\n\nQuestion : {query}\n\nAnswer :"
    return mistral_via_api(prompt)

# Interface Streamlit
st.title("romAIn")
st.write("Entrez votre question ci-dessous pour obtenir une réponse générée à partir de mes informations pertinentes.")

# Champ de texte pour l'utilisateur
query = st.text_input("Votre question")

if query:
    with st.spinner("Génération de la réponse..."):
        answer = rag_pipeline(query)
    st.write("Réponse générée :")
    st.write(answer)
