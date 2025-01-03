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
    "I study at Isen JUNIA in Lille since 2021, During my studies, I have learned about machine learning, deep learning, computer vision, natural language processing, reinforcement learning. I had lessons in mathematics, statistics, computer science, physics, electronics and project management",
    "Before Isen JUNIA, I was at ADIMAKER, an integrated preparatory class where I learned the basics of engineering",
    "I'm passionate about artificial intelligence, new technologies and computer science",
    "I'm based in Lille, France",
    "I have work on different project during my studies, like Project F.R.A.N.K who is a 3d project mixing AI on unity3D it is a horror game in a realistic universe, with advanced gameplay functions such as inventory management and item usage, all while being pursued by a monster under AI. And i have also worked on a local drive project on django named DriveMe. all this project are available on my github",
    "I'm currently looking for an internationally internship in AI, starting in April 2025",
    "My email is dujardin.romain@icloud.com and My phone number is 07 83 19 30 23",
    "I had professional experience as a pharmaceutical driver, accountant, machine operator or food truck clerk",
    "I have a driving license",
    "I graduated with the sti2d baccalaureate with honors",
    "I code in python, django, react and I master tools like rag, hyde, pytorsh"
    "I currently work on an inclusive LLM for disabled people, a project that I am developing with a team of 5 people. We use HyDE system to develop the project",
    "My hobbies are video games, reading, sports, cinema, music and cooking",
    "My qualities are my adaptability, my curiosity, my rigor, my creativity, my autonomy, my team spirit and my ability to learn quickly. My softkills are my ability to communicate, my ability to adapt, my ability to work in a team, my ability to solve problems and my ability to manage my time and my hardskills are my ability to code in python and other langages, i also know some tools like rag, hyde, pytorsh",
    "I'm speaking French (fluent) and English (got toeic 790/990)",
    "If I had to cite a default it would be that I like to control everything, do everything on a project, I have a little difficulty delegating the work"
    "My favorite movie is Lucy."
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
            "temperature": 0.1,
            "top_k": 10,
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
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer: Provide the answer only, without repeating the question or context. Only respond to the question provided, using the context."
    response = mistral_via_api(prompt)

    # Nettoyage de la réponse
    # Suppression explicite de la consigne
    unwanted_phrases = [
        "Provide the answer only, without repeating the question or context.",
        "Only respond to the question provided, using the context.",
        "Do not answer any other implicit or unrelated questions."
    ]
    for phrase in unwanted_phrases:
        response = response.replace(phrase, "").strip()

    # Extraction après "Answer:"
    answer_match = re.search(r"Answer:\s*(.*)", response, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    return response




              


        #   # Charger le fichier OBJ directement
        #   obj_file_path = "test.obj"  # Remplace par le chemin de ton fichier
        #   try:
        #       with open(obj_file_path, "r") as file:
        #           obj_data = file.read()
        #   except FileNotFoundError:
        #       st.error(f"Le fichier '{obj_file_path}' est introuvable. Vérifie le chemin.")
        #       st.stop()

        #   # Extraction des données du fichier OBJ
        #   vertices = []
        #   faces = []
        #   vertex_colors = []  # Pour stocker les couleurs des sommets

        #   for line in obj_data.split("\n"):
        #       parts = line.strip().split()
        #       if not parts or parts[0] not in {"v", "f"}:  # Ignorer les lignes vides ou inutiles
        #           continue
        #       if parts[0] == "v":  # Vertex avec couleur potentielle
        #           try:
        #               if len(parts) >= 4:  # Minimum : v x y z
        #                   vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        #                   if len(parts) == 7:  # Si les couleurs sont présentes
        #                       vertex_colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
        #                   else:
        #                       vertex_colors.append([1, 1, 1])  # Couleur blanche par défaut
        #           except ValueError:
        #               st.warning(f"Ligne invalide ignorée : {line}")
        #       elif parts[0] == "f":  # Face
        #           try:
        #               face = [int(p.split('/')[0]) - 1 for p in parts[1:]]
        #               faces.append(face)
        #           except ValueError:
        #               st.warning(f"Ligne de face invalide ignorée : {line}")

        #   # Vérification des données chargées
        #   if not vertices or not faces:
        #       st.error("Impossible de lire correctement les sommets ou les faces depuis le fichier OBJ.")
        #       st.stop()

        #   # Préparation des données pour Plotly
        #   x, y, z = zip(*vertices)
        #   i, j, k = zip(*faces)
        #   r, g, b = zip(*vertex_colors)
        #   colors = ['rgb({}, {}, {})'.format(int(r_ * 255), int(g_ * 255), int(b_ * 255)) for r_, g_, b_ in zip(r, g, b)]

        #   # Création de la figure 3D avec couleurs
        #   fig = go.Figure(data=[
        #       go.Mesh3d(
        #           x=x, y=y, z=z,
        #           i=i, j=j, k=k,
        #           vertexcolor=colors,  # Ajout des couleurs par sommet
        #           opacity=1.0,
        #           hoverinfo="skip",  # Désactive l'affichage des coordonnées au survol
        #       )
        #   ])

        #   # Suppression des axes, de la grille et des marges
        #   fig.update_layout(
        #       scene=dict(
        #           xaxis=dict(visible=False),  # Désactive l'axe X
        #           yaxis=dict(visible=False),  # Désactive l'axe Y
        #           zaxis=dict(visible=False),  # Désactive l'axe Z
        #           camera=dict(
        #               eye=dict(x=0, y=0, z=1.5),  # Position de la caméra
        #               up=dict(x=1, y=0, z=1),         # Orientation "haut" de la caméra
        #               center=dict(x=0, y=0, z=0)      # Point vers lequel la caméra regarde
        #           )
        #       ),
        #       margin=dict(l=0, r=0, t=0, b=0),  # Supprime les marges autour de l'objet
        #   )

        #   # Affichage dans Streamlit
        #   st.plotly_chart(fig, use_container_width=True)
        # Afficher l'image A de base

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
