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

# Configuration de la page Streamlit
st.set_page_config(layout="wide")

# Add session state for language selection if it doesn't exist
if 'language' not in st.session_state:
    st.session_state['language'] = 'EN'  # Default to English

# Styling
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
    .language-selector {
        position: absolute;
        top: 10px;
        right: 20px;
        z-index: 1000;
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

# Language selection in the top-right corner
st.markdown(
    """
    <script>
    function changeLanguage(lang) {
        // Use Streamlit's setComponentValue to update the session state
        window.parent.postMessage({
            type: "streamlit:setComponentValue",
            value: lang,
            dataType: "json",
            key: "selected_language"
        }, "*");
        
        // Reload the page to apply the change
        setTimeout(() => {
            window.parent.location.reload();
        }, 100);
    }
    </script>
    """,
    unsafe_allow_html=True
)

# Handle language selection from JavaScript
if 'selected_language' in st.session_state:
    st.session_state['language'] = st.session_state['selected_language']

# Alternative language selector using Streamlit components
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    lang_options = ["🇫🇷 FR", "🇬🇧 EN"]
    selected_lang_option = st.radio("", lang_options, index=1 if st.session_state['language'] == 'EN' else 0, horizontal=True)
    st.session_state['language'] = 'EN' if selected_lang_option == "🇬🇧 EN" else 'FR'

# Content based on selected language
CURRENT_LANG = st.session_state['language']

# Define language-specific elements
UI_TEXT = {
    'EN': {
        'title': 'Welcome to, <span style="opacity: 0.5;">rom</span>A</span>I<span style="opacity: 0.5;">n</span>',
        'subtitle': 'here is <span style="opacity: 0.5;">rom</span>A</span>I<span style="opacity: 0.5;">n</span>, an AI in the image of Romain Dujardin. Ask questions in English and he will answer them as best he can.',
        'text_tab': 'Text Input',
        'voice_tab': 'Voice Input',
        'question_placeholder': 'Your question:',
        'thinking': 'Thinking...',
        'listen_button': 'Listen to the answer',
        'generating_audio': 'Generating audio...',
        'voice_subtitle': 'Ask your question by voice',
        'recording': 'Recording...',
        'processing': 'Processing...',
        'generating_voice': 'Generating voice response...',
        'show_details': 'Show text details',
        'your_question': 'Your question:',
        'response': 'Response:',
        'no_transcription': 'No transcription was generated. Please try again.',
        'recording_failed': 'Recording failed. Please try again.',
        'upload_audio': 'Or upload an audio file',
        'processing_upload': 'Processing uploaded audio...',
        'upload_error': 'No transcription was generated from the uploaded file.'
    },
    'FR': {
        'title': 'Bienvenue sur, <span style="opacity: 0.5;">rom</span>A</span>I<span style="opacity: 0.5;">n</span>',
        'subtitle': 'voici <span style="opacity: 0.5;">rom</span>A</span>I<span style="opacity: 0.5;">n</span>, une IA à l\'image de Romain Dujardin. Posez des questions en français et il y répondra du mieux qu\'il peut.',
        'text_tab': 'Saisie de texte',
        'voice_tab': 'Saisie vocale',
        'question_placeholder': 'Votre question:',
        'thinking': 'Réflexion en cours...',
        'listen_button': 'Écouter la réponse',
        'generating_audio': 'Génération de l\'audio...',
        'voice_subtitle': 'Posez votre question par la voix',
        'recording': 'Enregistrement en cours...',
        'processing': 'Traitement en cours...',
        'generating_voice': 'Génération de la réponse vocale...',
        'show_details': 'Afficher les détails du texte',
        'your_question': 'Votre question:',
        'response': 'Réponse:',
        'no_transcription': 'Aucune transcription n\'a été générée. Veuillez réessayer.',
        'recording_failed': 'L\'enregistrement a échoué. Veuillez réessayer.',
        'upload_audio': 'Ou téléchargez un fichier audio',
        'processing_upload': 'Traitement de l\'audio téléchargé...',
        'upload_error': 'Aucune transcription n\'a été générée à partir du fichier téléchargé.'
    }
}

# Initialisation du modèle d'embedding et de l'index FAISS
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Define language-specific documents
documents_en = [
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

documents_fr = [
    "Je m'appelle Romain Dujardin",
    "J'ai 22 ans",
    "Je suis un étudiant français en école d'ingénieur dans l'IA",
    "J'étudie actuellement à JUNIA ISEN à Lille depuis 2021. Pendant mes études, j'ai appris le machine learning, le deep learning, la vision par ordinateur, le traitement du langage naturel, l'apprentissage par renforcement. J'ai eu des cours de mathématiques, statistiques, informatique, physique, électronique et gestion de projet",
    "Avant Isen JUNIA, j'étais à ADIMAKER, une classe préparatoire intégrée où j'ai appris les bases de l'ingénierie",
    "Je suis passionné par l'intelligence artificielle, les nouvelles technologies et l'informatique",
    "J'habite à Lille, en France",
    "J'ai travaillé sur différents projets pendant mes études, comme le Projet F.R.A.N.K qui est un projet 3D mélangeant l'IA sur unity3D, c'est un jeu d'horreur dans un univers réaliste, avec des fonctions de gameplay avancées comme la gestion d'inventaire et l'utilisation d'objets, tout en étant poursuivi par un monstre sous IA. Et j'ai aussi travaillé sur un projet de drive local sur django nommé DriveMe. Tous ces projets sont disponibles sur mon github",
    "Durant ces différents projets j'ai d'abord appris à gérer une équipe en tant que chef de projet et donc en même temps à travailler en équipe, j'ai également mis en pratique ce que je vois en cours dans des exemples concrets. En plus, j'ai pu traiter la résolution de problèmes sur certains projets",
    "Je recherche une alternance en IA pour septembre 2025",
    "J'ai besoin d'un contrat pour valider mon diplôme",
    "Mon email est dujardin.romain@icloud.com et mon numéro de téléphone est le 07 83 19 30 23",
    "J'ai eu des expériences professionnelles en tant que chauffeur pharmaceutique, comptable, opérateur de machine ou commis de food truck",
    "J'ai le permis de conduire et mon véhicule personnel",
    "J'ai obtenu le baccalauréat sti2d avec mention quand j'étais au lycée",
    "Je code en python, C, CPP, django, JavaScript et react. Je maîtrise des outils comme rag, hyde, pytorsh",
    "Je travaille actuellement sur un LLM inclusif pour les personnes handicapées, un projet que je développe avec une équipe de 5 personnes. Nous utilisons le système HyDE pour développer le projet",
    "Mes hobbies sont les jeux vidéo, la lecture, le sport, le cinéma, la musique et la cuisine",
    "Mon sport préféré est le football, mon équipe préférée est le LOSC",
    "Mes qualités sont mon adaptabilité, ma curiosité, ma rigueur, ma créativité, mon autonomie, mon esprit d'équipe et ma capacité à apprendre rapidement. Mes softkills sont ma capacité à communiquer, ma capacité à m'adapter, ma capacité à travailler en équipe, ma capacité à résoudre des problèmes et ma capacité à gérer mon temps et mes hardskills sont ma capacité à coder en python et autres langages, je connais aussi des outils comme rag, hyde, pytorsh",
    "Je parle français (couramment) et anglais B2 (j'ai obtenu le toeic 790/990)",
    "Si je devais citer un défaut, ce serait que j'aime tout faire, ce que je veux dire par là c'est que quand je travaille sur un nouveau projet, je suis enthousiaste et je veux tout faire et tout toucher dessus.",
    "Mon film préféré est Lucy."
]

# Select documents based on the current language
documents = documents_fr if CURRENT_LANG == 'FR' else documents_en

# Create embeddings for each document
doc_embeddings = embedding_model.encode(documents)

# Create a FAISS index
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

# Function to use Mistral via API with language support
def mistral_via_api(prompt, lang='EN'):
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
        error_msg = "Error: " if lang == 'EN' else "Erreur: "
        return f"{error_msg} {response.status_code} - {response.json()}"

# Improved RAG pipeline with first-person response and language support
def rag_pipeline(query, k=2, lang='EN'):
    relevant_docs, distances = find_relevant_docs(query, k)

    if not relevant_docs:  # If no relevant documents were found
        if lang == 'FR':
            return "Je suis désolé, je n'ai pas assez d'informations pour répondre à cette question."
        else:
            return "I'm sorry, I don't have enough information to answer that question."

    context = "\n".join(relevant_docs)
    
    # Modified prompt to favor first-person responses based on language
    if lang == 'FR':
        prompt = f"""Contexte: {context}

Question: {query}

Tu es Romain Dujardin, un étudiant en ingénierie IA de 22 ans à Lille. Réponds à la question directement à la première personne comme si tu étais Romain lui-même. N'utilise pas de phrases comme "d'après le contexte" ou "selon les informations". Réponds simplement naturellement comme Romain le ferait dans une conversation. Sois amical et direct. Utilise uniquement les informations du contexte.

Réponse:"""
    else:
        prompt = f"""Context: {context}

Question: {query}

You are Romain Dujardin, a 22-year-old AI engineering student in Lille. Answer the question directly in first person as if you are Romain himself. Don't use phrases like "based on the context" or "according to the information". Just answer naturally as Romain would in conversation. Be friendly and direct. Only use information from the context.

Answer:"""

    response = mistral_via_api(prompt, lang)

    # Extract the answer after "Answer:" or "Réponse:"
    search_pattern = r"Réponse:\s*(.*)" if lang == 'FR' else r"Answer:\s*(.*)"
    answer_match = re.search(search_pattern, response, re.DOTALL)
    
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # If the pattern is not found, take everything after the last occurrence of the prompt
        answer = response.split(prompt)[-1].strip()
    
    # Additional cleaning of unwanted phrases based on language
    if lang == 'FR':
        unwanted_phrases = [
            "En tant que Romain Dujardin,",
            "En tant que Romain, ",
            "Basé sur le contexte,",
            "Selon les informations,",
            "D'après les informations fournies,",
            "Je n'ai pas d'informations sur",
            "Il n'y a pas d'informations sur",
            "Le contexte ne mentionne pas",
        ]
        # Correct references to "Romain" with "Je" or "mon"
        answer = answer.replace("Romain est", "Je suis")
        answer = answer.replace("de Romain", "mon")
        answer = answer.replace("Romain a", "J'ai")
    else:
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
        # Correct references to "Romain" with "I" or "my"
        answer = answer.replace("Romain's", "my")
        answer = answer.replace("Romain is", "I am")
        answer = answer.replace("Romain has", "I have")
    
    for phrase in unwanted_phrases:
        answer = answer.replace(phrase, "").strip()
    
    return answer

# Function to transcribe audio with Whisper with language support
def transcribe_audio(audio_file, lang='EN'):
    try:
        # Choose the appropriate model based on language
        model = "openai/whisper-large-v3-turbo"
        
        transcription = client.automatic_speech_recognition(
            audio=audio_file,
            model=model
        )
        return transcription
    except Exception as e:
        print(f"Erreur lors de la transcription: {str(e)}")
        return None


# Function to generate audio from text with language support
def text_to_speech(text, lang='EN'):
    try:
        # Choose the appropriate voice model based on language
        voice = "facebook/mms-tts-fra" if lang == 'FR' else "facebook/mms-tts-eng"
        
        # Use the Hugging Face API for speech synthesis
        audio = client.text_to_speech(text, model=voice)
        return audio
    except Exception as e:
        error_msg = "Error during speech synthesis: " if lang == 'EN' else "Erreur lors de la synthèse vocale: "
        st.error(f"{error_msg} {str(e)}")
        return None

# Function to record audio
def record_audio(duration=5, local_mode=True):
    """
    Record audio with cross-platform support.
    
    Args:
        duration (int/float): Recording duration in seconds (only used in local mode)
        local_mode (bool): Whether to use PyAudio (local) or Streamlit audio input (cloud)
    
    Returns:
        tuple: (audio_bytes, sample_rate) or (None, None) if recording fails
    """
    if local_mode:
        try:
            import pyaudio
            import wave
        except ImportError:
            st.error("PyAudio is not available. Using Streamlit audio input.")
            return None, None

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        # Make sure that duration is a number (float or int)
        try:
            duration = float(duration)  # Explicit conversion to float
        except ValueError:
            error_msg = "The 'duration' value is not a valid number." if CURRENT_LANG == 'EN' else "La valeur de 'duration' n'est pas un nombre valide."
            st.error(error_msg)
            return None, None

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        recording_msg = f"Recording in progress... ({duration} seconds)" if CURRENT_LANG == 'EN' else f"Enregistrement en cours... ({duration} secondes)"
        st.info(recording_msg)

        frames = []

        # Verification after duration conversion
        try:
            for i in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK)
                frames.append(data)
        except TypeError as e:
            error_msg = f"Error with value multiplication: {e}" if CURRENT_LANG == 'EN' else f"Erreur avec la multiplication des valeurs: {e}"
            st.error(error_msg)
            return None, None

        stream.stop_stream()
        stream.close()
        p.terminate()

        audio_bytes = b''.join(frames)

        return audio_bytes, RATE
    
    else:
        # Streamlit cloud mode using st.audio_input
        audio_value = st.audio_input("Record a voice message", 
                                     key="cloud_audio_recorder", 
                                     sample_rate=16000, 
                                     sample_width=2,  # 16-bit
                                     channels=1)
        
        if audio_value is not None:
            # Convert Streamlit audio input to bytes
            return audio_value.getvalue(), 16000
        
        return None, None

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

# Display the main page content
st.markdown(f'<h1 class="centered h1">{UI_TEXT[CURRENT_LANG]["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="centered p">{UI_TEXT[CURRENT_LANG]["subtitle"]}</p>', unsafe_allow_html=True)

# Create tabs for different interaction methods
tabs = st.tabs([UI_TEXT[CURRENT_LANG]["text_tab"], UI_TEXT[CURRENT_LANG]["voice_tab"]])

with tabs[0]:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    # Tab for text input
    query = st.text_input(UI_TEXT[CURRENT_LANG]["question_placeholder"])
    
    if query:
        with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
            answer = rag_pipeline(query, lang=CURRENT_LANG)
        
        # Display the text response
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.write(answer)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Option to listen to the answer
        if st.button(UI_TEXT[CURRENT_LANG]["listen_button"]):
            with st.spinner(UI_TEXT[CURRENT_LANG]["generating_audio"]):
                audio_bytes = text_to_speech(answer, lang=CURRENT_LANG)
                if audio_bytes:
                    st.markdown(get_audio_player_html(audio_bytes), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    # Tab for voice input
    st.subheader(UI_TEXT[CURRENT_LANG]["voice_subtitle"])
    
    # Create a container for the audio player that will be updated with each recording
    audio_player_container = st.empty()
    text_results_container = st.empty()
    
    # Try local recording first, fallback to Streamlit audio input
    try:
        if st.button("🎤", key="record_button", help="Click to start recording"):
            # Attempt local recording first
            audio_bytes, sample_rate = record_audio(duration=5, local_mode=True)
            
            # If local recording fails, try Streamlit audio input
            if audio_bytes is None:
                audio_bytes, sample_rate = record_audio(duration=5, local_mode=False)
            
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
                    with st.spinner(UI_TEXT[CURRENT_LANG]["processing"]):
                        transcription = transcribe_audio(audio_file.read(), lang=CURRENT_LANG)
                
                if transcription:
                    # Process in background
                    with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
                        answer = rag_pipeline(transcription, lang=CURRENT_LANG)
                    
                    # Generate voice response
                    with st.spinner(UI_TEXT[CURRENT_LANG]["generating_voice"]):
                        audio_response = text_to_speech(answer, lang=CURRENT_LANG)
                        
                        if audio_response:
                            # Update the audio player container with new content
                            audio_player_html = get_audio_player_html(audio_response)
                            audio_player_container.markdown(audio_player_html, unsafe_allow_html=True)
                            
                            # Update the text container with expandable details
                            with text_results_container.expander(UI_TEXT[CURRENT_LANG]["show_details"]):
                                st.write(f"{UI_TEXT[CURRENT_LANG]['your_question']} {transcription}")
                                st.write(f"{UI_TEXT[CURRENT_LANG]['response']} {answer}")
                else:
                    st.error(UI_TEXT[CURRENT_LANG]["no_transcription"])
                
                # Cleanup
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            else:
                st.error(UI_TEXT[CURRENT_LANG]["recording_failed"])
    except Exception as e:
        st.error(f"Recording error: {str(e)}")
    
    # Existing audio file upload option
    uploaded_file = st.file_uploader(UI_TEXT[CURRENT_LANG]["upload_audio"], type=["mp3", "wav", "m4a"])
    if uploaded_file is not None:
        with st.spinner(UI_TEXT[CURRENT_LANG]["processing_upload"]):
            # Process the uploaded file
            transcription = transcribe_audio(uploaded_file, lang=CURRENT_LANG)
            
            if transcription:
                with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
                    answer = rag_pipeline(transcription, lang=CURRENT_LANG)
                
                with st.spinner(UI_TEXT[CURRENT_LANG]["generating_voice"]):
                    audio_response = text_to_speech(answer, lang=CURRENT_LANG)
                    
                    if audio_response:
                        # Update the audio player container with new content
                        audio_player_html = get_audio_player_html(audio_response)
                        audio_player_container.markdown(audio_player_html, unsafe_allow_html=True)
                        
                        # Update the text container with expandable details
                        with text_results_container.expander(UI_TEXT[CURRENT_LANG]["show_details"]):
                            st.write(UI_TEXT[CURRENT_LANG]["your_question"], transcription)
                            st.write(UI_TEXT[CURRENT_LANG]["response"], answer)
            else:
                st.error(UI_TEXT[CURRENT_LANG]["upload_error"])
                
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