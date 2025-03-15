import requests
import streamlit as st
from huggingface_hub import HfFolder
from sentence_transformers import SentenceTransformer
import faiss
import time
import os
import streamlit as st 
import streamlit_3d as sd
import re  # Import pour le traitement du texte
import streamlit as st
from streamlit_stl import stl_from_file, stl_from_text
import streamlit as st
import plotly.graph_objects as go



# Récupérer le token Hugging Face depuis Streamlit Secrets
hf_token = st.secrets["huggingface"]["token"]


# Initialisation du modèle d'embedding et de l'index FAISS
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modèle rapide et léger
documents = [
    "My name is Romain Dujardin",
    "I'm 22 years old",
    "I'm a French student in AI engineer",
    "I currently study at Isen JUNIA in Lille since 2021 (school), During my studies, I have learned about machine learning, deep learning, computer vision, natural language processing, reinforcement learning. I had lessons in mathematics, statistics, computer science, physics, electronics and project management",
    "Before Isen JUNIA, I was at ADIMAKER, an integrated preparatory class where I learned the basics of engineering",
    "I'm passionate about artificial intelligence, new technologies and computer science",
    "I'm based in Lille, France",
    "I have work on different project during my studies, like Project F.R.A.N.K who is a 3d project mixing AI on unity3D it is a horror game in a realistic universe, with advanced gameplay functions such as inventory management and item usage, all while being pursued by a monster under AI. And i have also worked on a local drive project on django named DriveMe. all this project are available on my github",
    "During these different projects I first learned to manage a team as a project manager and therefore at the same time to work in a team, I also put into practice what I see in progress in concrete examples . in addition I was able to deal with problem solving on certain projects",
    "I'm currently looking for a contract in AI, starting in september 2025 to validate my diploma",
    "My email is dujardin.romain@icloud.com and My phone number is 07 83 19 30 23",
    "I had professional experience as a pharmaceutical driver, accountant, machine operator or food truck clerk",
    "I have a driving license and my personal vehicle",
    "I graduated with the sti2d baccalaureate with honors when I was in college",
    "I code in python, C, CPP, django, JavaScript and react. I master tools like rag, hyde, pytorsh"
    "I currently work on an inclusive LLM for disabled people, a project that I am developing with a team of 5 people. We use HyDE system to develop the project",
    "My hobbies are video games, reading, sports, cinema, music and cooking",
    "my favorite sport is football, my favorite team is the LOSC",
    "My qualities are my adaptability, my curiosity, my rigor, my creativity, my autonomy, my team spirit and my ability to learn quickly. My softkills are my ability to communicate, my ability to adapt, my ability to work in a team, my ability to solve problems and my ability to manage my time and my hardskills are my ability to code in python and other langages, i also know some tools like rag, hyde, pytorsh",
    "I'm speaking French (fluent) and English B2 (got toeic 790/990)",
    "If I had to cite a default it would be that I like to do everything, what I mean by that is that when I work on a new project I am enthusiastic and want to do everything and touch everything on it."
    "My favorite movie is Lucy."
]

# Créer des embeddings pour chaque document
doc_embeddings = embedding_model.encode(documents)

# Créer un index FAISS
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

def find_relevant_docs(query, k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    print(f"Query: {query}")  # Debug
    print(f"Distances trouvées: {distances[0]}")  

    # Nouveau seuil ajusté
    threshold = 1.5  

    # Vérifier si au moins un document est en dessous du seuil
    if all(dist > threshold for dist in distances[0]):
        print("Aucun document pertinent trouvé.")  
        return [], []  

    return [documents[idx] for idx in indices[0]], distances[0]

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
            "temperature": 0.5,
            "top_k": 10,
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error : {response.status_code} - {response.json()}"


# Pipeline RAG - Combiner recherche et génération
def rag_pipeline(query, k=2):
    relevant_docs, distances = find_relevant_docs(query, k)

    if not relevant_docs:  # Si aucun document pertinent n'a été trouvé
        return "Je suis désolé, je ne peux pas répondre."

    context = "\n".join(relevant_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer: Provide the answer only directly, without repeating the question or context or any additional text. Only respond to the question provided, using the context else do not answer."
    response = mistral_via_api(prompt)

    # Nettoyage de la réponse
    unwanted_phrases = [
        "Provide the answer only directly, without repeating the question or context or any additional text.",
        "Only respond to the question provided, using the context else do not answer.",
        "Do not answer any other implicit or unrelated questions.",
        "Answer:"
    ]
    for phrase in unwanted_phrases:
        response = response.replace(phrase, "").strip()

    # Extraction après "Answer:"
    answer_match = re.search(r"Answer:\s*(.*)", response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    return response

st.set_page_config(layout="wide")

st.markdown(
        """
        <style>
        .centered {
            text-align: center;
        }
        .h1 {
            font-size: 7vw;
        }
        .p {
            font-size: 1vw;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown('<h1 class="centered h1">Welcome to, <span style="opacity: 0.5;">rom</span>A</span>I<span style="opacity: 0.5;">n</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="centered p">here is <span style="opacity: 0.5;">rom</span>A</span>I<span style="opacity: 0.5;">n</span>, an AI in the image of Romain Dujardin. Ask him questions in English and he will answer them as best he can.</p>', unsafe_allow_html=True)
st.markdown('<p class="centered p">(Can be made mistake)</p>', unsafe_allow_html=True)

        
# Champ de texte pour l'utilisateur
query = st.text_input("Your question:")

if query:
    with st.spinner("Generating the response..."):
        answer = rag_pipeline(query)
    # Afficher uniquement la réponse
    st.subheader("Answer:")
    st.write(answer)
