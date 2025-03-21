import requests
import streamlit as st
from huggingface_hub import HfFolder, InferenceClient
from sentence_transformers import SentenceTransformer
import faiss
import time
import os
import re
import numpy as np
from io import BytesIO
import base64
import plotly.graph_objects as go
import pyaudio
import wave
import tempfile

# Récupérer le token Hugging Face depuis Streamlit Secrets
hf_token = st.secrets["huggingface"]["token"]

# Initialisation du client d'inférence
client = InferenceClient(api_key=hf_token)

# Initialisation du modèle d'embedding et de l'index FAISS
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
documents = [
    "My name is Romain Dujardin",
    "I'm 22 years old",
    "I'm a French student in AI engineering",
    "I currently study at Isen JUNIA in Lille since 2021 (school), During my studies, I have learned about machine learning, deep learning, computer vision, natural language processing, reinforcement learning. I had lessons in mathematics, statistics, computer science, physics, electronics and project management",
    "Before Isen JUNIA, I was at ADIMAKER, an integrated preparatory class where I learned the basics of engineering",
    "I'm passionate about artificial intelligence, new technologies and computer science",
    "I'm based in Lille, France",
    "I have work on different project during my studies, like Project F.R.A.N.K who is a 3d project mixing AI on unity3D it is a horror game in a realistic universe, with advanced gameplay functions such as inventory management and item usage, all while being pursued by a monster under AI. And i have also worked on a local drive project on django named DriveMe. all this project are available on my github",
    "During these different projects I first learned to manage a team as a project manager and therefore at the same time to work in a team, I also put into practice what I see in progress in concrete examples . in addition I was able to deal with problem solving on certain projects",
    "I'm looking for a contract in AI",
    "I need a contract to validate my diploma",    
    "My email is dujardin.romain@icloud.com and My phone number is 07 83 19 30 23",
    "I had professional experience as a pharmaceutical driver, accountant, machine operator or food truck clerk",
    "I have a driving license and my personal vehicle",
    "I graduated with the sti2d baccalaureate with honors when I was in college",
    "I code in python, C, CPP, django, JavaScript and react. I master tools like rag, hyde, pytorsh",
    "I currently work on an inclusive LLM for disabled people, a project that I am developing with a team of 5 people. We use HyDE system to develop the project",
    "My hobbies are video games, reading, sports, cinema, music and cooking",
    "my favorite sport is football, my favorite team is the LOSC",
    "My qualities are my adaptability, my curiosity, my rigor, my creativity, my autonomy, my team spirit and my ability to learn quickly. My softkills are my ability to communicate, my ability to adapt, my ability to work in a team, my ability to solve problems and my ability to manage my time and my hardskills are my ability to code in python and other langages, i also know some tools like rag, hyde, pytorsh",
    "I'm speaking French (fluent) and English B2 (got toeic 790/990)",
    "If I had to cite a default it would be that I like to do everything, what I mean by that is that when I work on a new project I am enthusiastic and want to do everything and touch everything on it.",
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
    
    # Seuil ajusté sur la meilleure correspondance
    threshold = 1.41  

    # Vérifier si le meilleur document est en dessous du seuil
    if distances[0][0] > threshold:
        return [], []  

    return [documents[idx] for idx in indices[0]], distances[0]

# Fonction pour utiliser Mistral via l'API
def mistral_via_api(prompt):
    API_URL = "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2"
    
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

# Pipeline RAG amélioré - Réponse à la première personne
def rag_pipeline(query, k=2):
    relevant_docs, distances = find_relevant_docs(query, k)

    if not relevant_docs:  # Si aucun document pertinent n'a été trouvé
        return "I'm sorry, I don't have enough information to answer that question."

    context = "\n".join(relevant_docs)
    
    # Prompt modifié pour favoriser des réponses à la première personne
    prompt = f"""Context: {context}

Question: {query}

You are Romain Dujardin, a 22-year-old AI engineering student in Lille. Answer the question directly in first person as if you are Romain himself. Don't use phrases like "based on the context" or "according to the information". Just answer naturally as Romain would in conversation. Be friendly and direct. Only use information from the context.

Answer:"""

    response = mistral_via_api(prompt)

    # Extraction de la réponse après "Answer:"
    answer_match = re.search(r"Answer:\s*(.*)", response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Si le pattern "Answer:" n'est pas trouvé, prendre tout après la dernière apparition du prompt
        answer = response.split(prompt)[-1].strip()
    
    # Nettoyage supplémentaire des phrases restantes à éviter
    unwanted_phrases = [
        "As Romain Dujardin,",
        "As Romain, ",
        "Based on the context,",
        "According to the information,",
        "Based on the information provided,",
        "I don't have information about",
        "There is no information about",
        "The context doesn't mention",
    ]
    for phrase in unwanted_phrases:
        answer = answer.replace(phrase, "").strip()
    
    # Corriger les références à "Romain" par "I" ou "my"
    answer = answer.replace("Romain's", "my")
    answer = answer.replace("Romain is", "I am")
    answer = answer.replace("Romain has", "I have")
    answer = answer.replace("Romain", "I")
    
    return answer

# Fonction pour transcrire l'audio avec Whisper
def transcribe_audio(audio_file):
    try:
        transcription = client.automatic_speech_recognition(
            audio=audio_file,
            model="openai/whisper-large-v3-turbo"
              # Utilisation de generate_kwargs
        )
        return transcription
    except Exception as e:
        print(f"Erreur lors de la transcription: {str(e)}")
        return None


# Fonction pour générer de l'audio à partir du texte
def text_to_speech(text, voice="facebook/mms-tts-eng"):
    try:
        # Utiliser l'API Hugging Face pour la synthèse vocale
        audio = client.text_to_speech(text, model=voice)
        return audio
    except Exception as e:
        st.error(f"Error during speech synthesis: {str(e)}")
        return None

# Fonction pour créer un lecteur audio HTML à partir des données audio
def record_audio(duration=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    # Assurons-nous que duration est un nombre (float ou int)
    try:
        duration = float(duration)  # Conversion explicite en float
    except ValueError:
        st.error("La valeur de 'duration' n'est pas un nombre valide.")
        return None, None

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    st.info(f"Enregistrement en cours... ({duration} secondes)")

    frames = []

    # Vérification après conversion de duration
    try:
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
    except TypeError as e:
        st.error(f"Erreur avec la multiplication des valeurs : {e}")
        return None, None

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_bytes = b''.join(frames)

    return audio_bytes, RATE

# Improved audio player function with unique ID generation
def get_audio_player_html(audio_bytes):
    if audio_bytes is None:
        return None
    
    # Generate a truly unique ID using timestamp
    unique_id = f"audio_{int(time.time() * 1000)}"
    
    # Encode to base64 for HTML
    b64 = base64.b64encode(audio_bytes).decode()
    
    # Create HTML audio player with unique ID and autoplay
    audio_player = f"""
    <audio id="{unique_id}" controls autoplay="true">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    <script>
        // Force the browser to recognize the new audio element
        document.getElementById("{unique_id}").load();
    </script>
    """
    
    return audio_player


# Configuration de la page Streamlit
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
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
    }
    .answer-box {
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .tab-container {
        border-radius: 10px;
        padding: 20px;
        margin-top: 10px;
    }
    .recorder-button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 24px;
        cursor: pointer;
        margin: 10px auto;
        display: block;
    }
    .recorder-button.recording {
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(255, 75, 75, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(255, 75, 75, 0);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="centered h1">Welcome to, <span style="opacity: 0.5;">rom</span>A</span>I<span style="opacity: 0.5;">n</span></h1>', unsafe_allow_html=True)
st.markdown('<p class="centered p">here is <span style="opacity: 0.5;">rom</span>A</span>I<span style="opacity: 0.5;">n</span>, an AI in the image of Romain Dujardin. Ask questions in English and he will answer them as best he can.</p>', unsafe_allow_html=True)

# Créer des onglets pour les différentes méthodes d'interaction
tabs = st.tabs(["Text Input", "Voice Input"])

with tabs[0]:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    # Onglet pour l'entrée de texte
    query = st.text_input("Your question:")
    
    if query:
        with st.spinner("Thinking..."):
            answer = rag_pipeline(query)
        
        # Afficher la réponse textuelle
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.write(answer)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Option pour écouter la réponse
        if st.button("Listen to the answer"):
            with st.spinner("Generating audio..."):
                audio_bytes = text_to_speech(answer)
                if audio_bytes:
                    st.markdown(get_audio_player_html(audio_bytes), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    # Onglet pour l'entrée vocale
    st.subheader("Ask your question by voice")
    # Create a container for the audio player that will be updated with each recording
    audio_player_container = st.empty()
    text_results_container = st.empty()
    
    if st.button("🎤", key="record_button", help="Click to start recording"):
        # Audio recording
        with st.spinner("Recording..."):
            audio_bytes, sample_rate = record_audio(duration=5)
            
            if audio_bytes and sample_rate:
                # Process the audio without displaying intermediate steps
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                    with wave.open(temp_audio_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(sample_rate)
                        wf.writeframes(audio_bytes)
                
                # Transcribe silently
                with open(temp_audio_path, "rb") as audio_file:
                    with st.spinner("Processing..."):
                        transcription = transcribe_audio(audio_file.read())
                
                if transcription:
                    # Process in background
                    with st.spinner("Thinking..."):
                        answer = rag_pipeline(transcription)
                    
                    # Generate voice response
                    with st.spinner("Generating voice response..."):
                        audio_response = text_to_speech(answer, voice="facebook/mms-tts-eng")
                        
                        if audio_response:
                            # Update the audio player container with new content
                            audio_player_html = get_audio_player_html(audio_response)
                            audio_player_container.markdown(audio_player_html, unsafe_allow_html=True)
                            
                            # Update the text container with expandable details
                            with text_results_container.expander("Show text details"):
                                st.write("Your question:", transcription)
                                st.write("Response:", answer)
                else:
                    st.error("No transcription was generated. Please try again.")
                
                # Cleanup
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            else:
                st.error("Recording failed. Please try again.")
    else:
        # Allow users to upload audio files as an alternative
        uploaded_file = st.file_uploader("Or upload an audio file", type=["mp3", "wav", "m4a"])
        if uploaded_file is not None:
            with st.spinner("Processing uploaded audio..."):
                # Process the uploaded file
                transcription = transcribe_audio(uploaded_file)
                
                if transcription:
                    with st.spinner("Thinking..."):
                        answer = rag_pipeline(transcription)
                    
                    with st.spinner("Generating voice response..."):
                        audio_response = text_to_speech(answer)
                        
                        if audio_response:
                            # Update the audio player container with new content
                            audio_player_html = get_audio_player_html(audio_response)
                            audio_player_container.markdown(audio_player_html, unsafe_allow_html=True)
                            
                            # Update the text container with expandable details
                            with text_results_container.expander("Show text details"):
                                st.write("Your question:", transcription)
                                st.write("Response:", answer)
                else:
                    st.error("No transcription was generated from the uploaded file.")
                    
    st.markdown('</div>', unsafe_allow_html=True)

# Ajouter un script JavaScript pour recevoir les données audio de l'enregistreur
st.markdown(
    """
    <script>
    // Écouter les messages depuis l'iframe
    window.addEventListener('message', function(event) {
        if (event.data.type === 'streamlit:setComponentValue') {
            // Stocker les données audio dans la session state
            window.parent.postMessage({
                type: "streamlit:setComponentValue",
                value: event.data.value,
                key: "audio_data"
            }, "*");
        }
    });
    </script>
    """,
    unsafe_allow_html=True
)