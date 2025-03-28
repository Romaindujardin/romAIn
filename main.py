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
# Removed pyaudio import
import wave
import tempfile
from st_audiorec import st_audiorec # CORRECTED Import

# Récupérer le token Hugging Face depuis Streamlit Secrets
hf_token = st.secrets["huggingface"]["token"]

# Initialisation du client d'inférence
client = InferenceClient(api_key=hf_token)

# Configuration de la page Streamlit
st.set_page_config(layout="wide")

# Add session state for language selection if it doesn't exist
if 'language' not in st.session_state:
    st.session_state['language'] = 'EN'  # Default to English

# Styling - Kept the same styling, adjust if needed for the new component
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
    /* You might need to inspect the element for st_audiorec's specific classes if you want deep styling */
    div[data-testid="stAudioRec"] { /* Example selector, might change */
         margin: 10px auto; /* Center the component */
         display: block;
         width: fit-content; /* Adjust width as needed */
    }
    .language-selector {
        position: absolute;
        top: 10px;
        right: 20px;
        z-index: 1000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Language selection
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    lang_options = ["🇫🇷 FR", "🇬🇧 EN"]
    selected_lang_option = st.radio(
        "Language",
        lang_options,
        index=1 if st.session_state['language'] == 'EN' else 0,
        horizontal=True,
        label_visibility="collapsed"
    )
    new_lang = 'EN' if selected_lang_option == "🇬🇧 EN" else 'FR'
    if st.session_state['language'] != new_lang:
        st.session_state['language'] = new_lang
        st.rerun()

CURRENT_LANG = st.session_state['language']

# UI Text Definitions (Kept the same)
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
        'record_instruction': 'Use the recorder below to ask your question:', # Updated instruction slightly
        'processing': 'Processing audio...',
        'generating_voice': 'Generating voice response...',
        'show_details': 'Show text details',
        'your_question': 'Your question:',
        'response': 'Response:',
        'no_transcription': 'No transcription was generated. Please try again.',
        'recording_failed': 'Audio recording failed or no audio received.',
        'upload_audio': 'Or upload an audio file',
        'processing_upload': 'Processing uploaded audio...',
        'upload_error': 'No transcription was generated from the uploaded file.',
        'audio_playback_error': 'Error playing audio response.',
        'processing_error': 'Error processing recorded audio.',
        'upload_process_error': 'Error processing uploaded file.',
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
        'record_instruction': 'Utilisez l\'enregistreur ci-dessous pour poser votre question :', # Updated instruction slightly
        'processing': 'Traitement de l\'audio...',
        'generating_voice': 'Génération de la réponse vocale...',
        'show_details': 'Afficher les détails du texte',
        'your_question': 'Votre question:',
        'response': 'Réponse:',
        'no_transcription': 'Aucune transcription n\'a été générée. Veuillez réessayer.',
        'recording_failed': 'L\'enregistrement audio a échoué ou aucun audio n\'a été reçu.',
        'upload_audio': 'Ou téléchargez un fichier audio',
        'processing_upload': 'Traitement de l\'audio téléchargé...',
        'upload_error': 'Aucune transcription n\'a été générée à partir du fichier téléchargé.',
        'audio_playback_error': 'Erreur lors de la lecture de la réponse audio.',
        'processing_error': 'Erreur lors du traitement de l\'audio enregistré.',
        'upload_process_error': 'Erreur lors du traitement du fichier téléchargé.',
    }
}


# Model/FAISS Initialization (Kept the same)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

embedding_model = load_embedding_model()

documents_en = [ # Kept the same documents
    "My name is Romain Dujardin", "I'm 22 years old", "I'm a French student in AI engineering",
    "I currently study at Isen JUNIA in Lille since 2021 (school), During my studies, I have learned about machine learning, deep learning, computer vision, natural language processing, reinforcement learning. I had lessons in mathematics, statistics, computer science, physics, electronics and project management",
    "Before Isen JUNIA, I was at ADIMAKER, an integrated preparatory class where I learned the basics of engineering",
    "I'm passionate about artificial intelligence, new technologies and computer science", "I'm based in Lille, France",
    "I have work on different project during my studies, like Project F.R.A.N.K who is a 3d project mixing AI on unity3D it is a horror game in a realistic universe, with advanced gameplay functions such as inventory management and item usage, all while being pursued by a monster under AI. And i have also worked on a local drive project on django named DriveMe. all this project are available on my github",
    "During these different projects I first learned to manage a team as a project manager and therefore at the same time to work in a team, I also put into practice what I see in progress in concrete examples . in addition I was able to deal with problem solving on certain projects",
    "I'm looking for a contract in AI", "I need a contract to validate my diploma",
    "My email is dujardin.romain@icloud.com and My phone number is 07 83 19 30 23",
    "I had professional experience as a pharmaceutical driver, accountant, machine operator or food truck clerk",
    "I have a driving license and my personal vehicle", "I graduated with the sti2d baccalaureate with honors when I was in college",
    "I code in python, C, CPP, django, JavaScript and react. I master tools like rag, hyde, pytorsh",
    "I currently work on an inclusive LLM for disabled people, a project that I am developing with a team of 5 people. We use HyDE system to develop the project",
    "My hobbies are video games, reading, sports, cinema, music and cooking", "my favorite sport is football, my favorite team is the LOSC",
    "My qualities are my adaptability, my curiosity, my rigor, my creativity, my autonomy, my team spirit and my ability to learn quickly. My softkills are my ability to communicate, my ability to adapt, my ability to work in a team, my ability to solve problems and my ability to manage my time and my hardskills are my ability to code in python and other langages, i also know some tools like rag, hyde, pytorsh",
    "I'm speaking French (fluent) and English B2 (got toeic 790/990)",
    "If I had to cite a default it would be that I like to do everything, what I mean by that is that when I work on a new project I am enthusiastic and want to do everything and touch everything on it.",
    "My favorite movie is Lucy."
]
documents_fr = [ # Kept the same documents
    "Je m'appelle Romain Dujardin", "J'ai 22 ans", "Je suis un étudiant français en école d'ingénieur dans l'IA",
    "J'étudie actuellement à JUNIA ISEN à Lille depuis 2021. Pendant mes études, j'ai appris le machine learning, le deep learning, la vision par ordinateur, le traitement du langage naturel, l'apprentissage par renforcement. J'ai eu des cours de mathématiques, statistiques, informatique, physique, électronique et gestion de projet",
    "Avant Isen JUNIA, j'étais à ADIMAKER, une classe préparatoire intégrée où j'ai appris les bases de l'ingénierie",
    "Je suis passionné par l'intelligence artificielle, les nouvelles technologies et l'informatique", "J'habite à Lille, en France",
    "J'ai travaillé sur différents projets pendant mes études, comme le Projet F.R.A.N.K qui est un projet 3D mélangeant l'IA sur unity3D, c'est un jeu d'horreur dans un univers réaliste, avec des fonctions de gameplay avancées comme la gestion d'inventaire et l'utilisation d'objets, tout en étant poursuivi par un monstre sous IA. Et j'ai aussi travaillé sur un projet de drive local sur django nommé DriveMe. Tous ces projets sont disponibles sur mon github",
    "Durant ces différents projets j'ai d'abord appris à gérer une équipe en tant que chef de projet et donc en même temps à travailler en équipe, j'ai également mis en pratique ce que je vois en cours dans des exemples concrets. En plus, j'ai pu traiter la résolution de problèmes sur certains projets",
    "Je recherche une alternance en IA pour septembre 2025", "J'ai besoin d'un contrat pour valider mon diplôme",
    "Mon email est dujardin.romain@icloud.com et mon numéro de téléphone est le 07 83 19 30 23",
    "J'ai eu des expériences professionnelles en tant que chauffeur pharmaceutique, comptable, opérateur de machine ou commis de food truck",
    "J'ai le permis de conduire et mon véhicule personnel", "J'ai obtenu le baccalauréat sti2d avec mention quand j'étais au lycée",
    "Je code en python, C, CPP, django, JavaScript et react. Je maîtrise des outils comme rag, hyde, pytorsh",
    "Je travaille actuellement sur un LLM inclusif pour les personnes handicapées, un projet que je développe avec une équipe de 5 personnes. Nous utilisons le système HyDE pour développer le projet",
    "Mes hobbies sont les jeux vidéo, la lecture, le sport, le cinéma, la musique et la cuisine",
    "Mon sport préféré est le football, mon équipe préférée est le LOSC",
    "Mes qualités sont mon adaptabilité, ma curiosité, ma rigueur, ma créativité, mon autonomie, mon esprit d'équipe et ma capacité à apprendre rapidement. Mes softkills sont ma capacité à communiquer, ma capacité à m'adapter, ma capacité à travailler en équipe, ma capacité à résoudre des problèmes et ma capacité à gérer mon temps et mes hardskills sont ma capacité à coder en python et autres langages, je connais aussi des outils comme rag, hyde, pytorsh",
    "Je parle français (couramment) et anglais B2 (j'ai obtenu le toeic 790/990)",
    "Si je devais citer un défaut, ce serait que j'aime tout faire, ce que je veux dire par là c'est que quand je travaille sur un nouveau projet, je suis enthousiaste et je veux tout faire et tout toucher dessus.",
    "Mon film préféré est Lucy."
]


@st.cache_resource(show_spinner=False)
def build_faiss_index(_documents):
    doc_embeddings = embedding_model.encode(_documents)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    return index, _documents

documents = documents_fr if CURRENT_LANG == 'FR' else documents_en
index, current_documents = build_faiss_index(documents)

def find_relevant_docs(query, k=3): # Kept the same
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)
    threshold = 1.41
    if distances.size == 0 or distances[0][0] > threshold:
        return [], []
    return [current_documents[idx] for idx in indices[0] if idx < len(current_documents)], distances[0]

# Mistral API Call (Kept the same, including error handling)
def mistral_via_api(prompt, lang='EN'):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    if hf_token is None: return "Error: No tokens found."
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.1, "top_k": 10, "return_full_text": False}}
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]: return result[0]["generated_text"]
        elif isinstance(result, dict) and "generated_text" in result: return result["generated_text"]
        else: st.warning(f"Unexpected API response format: {result}"); return str(result)
    except requests.exceptions.RequestException as e:
        error_msg = UI_TEXT[lang].get('api_request_error', "API Request Error: ")
        st.error(f"{error_msg}{e}")
        try: st.error(f"API Error Details: {response.json()}")
        except: pass
        return None
    except Exception as e:
        error_msg = UI_TEXT[lang].get('api_call_error', "Error calling Mistral API: ")
        st.error(f"{error_msg}{e}")
        return None

# RAG Pipeline (Kept the same, including prompt adjustments and cleaning)
def rag_pipeline(query, k=2, lang='EN'):
    relevant_docs, distances = find_relevant_docs(query, k)
    no_info_msg = UI_TEXT[lang].get('response_no_info', ("Je suis désolé je ne suis pas en capacité de repondre à cette question..." if lang == 'FR' else "I'm sorry I'm not able to answer this question..."))
    if not relevant_docs: return no_info_msg
    context = "\n".join(relevant_docs)
    if lang == 'FR':
        prompt = f"""Contexte: {context}\n\nQuestion: {query}\n\nTu es Romain Dujardin... Réponds à la première personne en te basant STRICTEMENT et EXCLUSIVEMENT sur les informations fournies dans le Contexte. N'utilise AUCUNE connaissance extérieure. Si l'information n'est PAS PRÉSENTE dans le contexte, réponds EXACTEMENT 'Je ne dispose pas de cette information dans mon contexte actuel.' NE spécule PAS et N'invente RIEN.\n\nRéponse:"""
    else:
        prompt = f"""Context: {context}\n\nQuestion: {query}\n\nYou are Romain Dujardin... Answer in first person based STRICTLY and EXCLUSIVELY on the information provided in the Context. Do NOT use any external knowledge. If the information is NOT PRESENT in the context, reply EXACTLY 'I do not have that information in my current context.' DO NOT speculate or invent anything.\n\nAnswer:"""
    response_text = mistral_via_api(prompt, lang)
    if response_text is None: return UI_TEXT[lang].get('response_generation_error', "Sorry, error generating response.")
    answer = response_text.strip()
    if lang == 'FR':
        unwanted = ["En tant que Romain Dujardin,", "En tant que Romain, ", "Basé sur le contexte,", "..."] # Add all unwanted phrases
        answer = answer.replace("Romain est", "Je suis").replace("Romain a", "J'ai").replace("Romain", "Romain")
        answer = answer.replace("il est", "je suis").replace("il a", "j'ai")
    else:
        unwanted = ["As Romain Dujardin,", "As Romain, ", "Based on the context,", "..."] # Add all unwanted phrases
        answer = answer.replace("Romain's", "my").replace("Romain is", "I am").replace("Romain has", "I have").replace("Romain", "Romain")
        answer = answer.replace("he is", "I am").replace("he has", "I have")
    for phrase in unwanted:
        answer = re.sub(rf'^{re.escape(phrase)}\s*', '', answer, flags=re.IGNORECASE)
    if not answer or answer.lower().strip() in ["answer:", "réponse:"]: return no_info_msg
    return answer

# Transcription Function (Kept the same)
def transcribe_audio(audio_data, lang='EN'):
    try:
        model = "openai/whisper-large-v3"
        transcription_response = client.automatic_speech_recognition(audio=audio_data, model=model)
        if isinstance(transcription_response, dict) and 'text' in transcription_response: return transcription_response['text']
        elif isinstance(transcription_response, str): return transcription_response
        else: st.error(f"Unexpected transcription response format: {transcription_response}"); return None
    except Exception as e: st.error(f"Error during transcription: {str(e)}"); return None

# Text-to-Speech Function (Kept the same)
def text_to_speech(text, lang='EN'):
    try:
        voice = "facebook/mms-tts-fra" if lang == 'FR' else "facebook/mms-tts-eng"
        audio_bytes = client.text_to_speech(text, model=voice)
        return audio_bytes
    except Exception as e: st.error(f"{UI_TEXT[lang].get('tts_error', 'TTS Error:')} {str(e)}"); return None


# --- Main App Layout ---
st.markdown(f'<h1 class="centered h1">{UI_TEXT[CURRENT_LANG]["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="centered p">{UI_TEXT[CURRENT_LANG]["subtitle"]}</p>', unsafe_allow_html=True)

tab_titles = [UI_TEXT[CURRENT_LANG]["text_tab"], UI_TEXT[CURRENT_LANG]["voice_tab"]]
tabs = st.tabs(tab_titles)

# --- Text Input Tab --- (Kept the same)
with tabs[0]:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    query = st.text_input(UI_TEXT[CURRENT_LANG]["question_placeholder"], key="text_query_input")
    if query:
        with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
            answer = rag_pipeline(query, lang=CURRENT_LANG)
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.write(answer)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button(UI_TEXT[CURRENT_LANG]["listen_button"], key="text_listen_button"):
            with st.spinner(UI_TEXT[CURRENT_LANG]["generating_audio"]):
                audio_bytes = text_to_speech(answer, lang=CURRENT_LANG)
                if audio_bytes: st.audio(audio_bytes, format="audio/wav")
                else: st.error(UI_TEXT[CURRENT_LANG]['audio_playback_error'])
    st.markdown('</div>', unsafe_allow_html=True)

# --- Voice Input Tab --- (MODIFIED TO USE st_audiorec)
with tabs[1]:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.subheader(UI_TEXT[CURRENT_LANG]["voice_subtitle"])

    audio_player_container = st.container()
    text_results_container = st.container()

    # Use the st_audiorec component
    st.markdown(f"<p style='text-align: center;'>{UI_TEXT[CURRENT_LANG]['record_instruction']}</p>", unsafe_allow_html=True)
    wav_audio_data = st_audiorec() # CORRECTED component call

    # Process recorded audio if data is received
    # wav_audio_data contains the audio data in WAV format bytes
    if wav_audio_data is not None:
        transcription = None
        try:
            # Pass the WAV bytes directly to the transcription function
            with st.spinner(UI_TEXT[CURRENT_LANG]["processing"]):
                 transcription = transcribe_audio(wav_audio_data, lang=CURRENT_LANG)

        except Exception as e:
             st.error(f"{UI_TEXT[CURRENT_LANG]['processing_error']} {e}")
             transcription = None

        if transcription and isinstance(transcription, str) and transcription.strip():
            audio_player_container.empty()
            with text_results_container: st.empty()

            with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
                answer = rag_pipeline(transcription, lang=CURRENT_LANG)

            with st.spinner(UI_TEXT[CURRENT_LANG]["generating_voice"]):
                audio_response = text_to_speech(answer, lang=CURRENT_LANG)

                if audio_response:
                    with audio_player_container:
                         st.audio(audio_response, format="audio/wav") # Use st.audio

                    with text_results_container.expander(UI_TEXT[CURRENT_LANG]["show_details"]):
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['your_question']}**"); st.write(transcription)
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['response']}**"); st.write(answer)
                else:
                     # If TTS failed, show text response anyway
                     with text_results_container.expander(UI_TEXT[CURRENT_LANG]["show_details"]):
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['your_question']}**"); st.write(transcription)
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['response']}**"); st.write(answer)
                        st.warning(UI_TEXT[CURRENT_LANG]['audio_playback_error'])

        elif transcription is None:
            st.error(UI_TEXT[CURRENT_LANG]["no_transcription"])
        else:
             st.warning(UI_TEXT[CURRENT_LANG]["no_transcription"] + " (Transcription was empty)")

    # --- Upload Option --- (Kept the same)
    st.markdown("---")
    uploaded_file = st.file_uploader(
        UI_TEXT[CURRENT_LANG]["upload_audio"],
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        key="audio_uploader"
    )
    if uploaded_file is not None:
        transcription = None
        try:
            uploaded_bytes = uploaded_file.getvalue()
            with st.spinner(UI_TEXT[CURRENT_LANG]["processing"]):
                transcription = transcribe_audio(uploaded_bytes, lang=CURRENT_LANG)
        except Exception as e:
            st.error(f"{UI_TEXT[CURRENT_LANG]['upload_process_error']} {e}")
            transcription = None

        if transcription and isinstance(transcription, str) and transcription.strip():
            audio_player_container.empty()
            with text_results_container: st.empty()

            with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
                answer = rag_pipeline(transcription, lang=CURRENT_LANG)

            with st.spinner(UI_TEXT[CURRENT_LANG]["generating_voice"]):
                audio_response = text_to_speech(answer, lang=CURRENT_LANG)

                if audio_response:
                     with audio_player_container: st.audio(audio_response, format="audio/wav")
                     with text_results_container.expander(UI_TEXT[CURRENT_LANG]["show_details"]):
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['your_question']}**"); st.write(transcription)
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['response']}**"); st.write(answer)
                else:
                     with text_results_container.expander(UI_TEXT[CURRENT_LANG]["show_details"]):
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['your_question']}**"); st.write(transcription)
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['response']}**"); st.write(answer)
                        st.warning(UI_TEXT[CURRENT_LANG]['audio_playback_error'])

        elif transcription is None:
            st.error(UI_TEXT[CURRENT_LANG]["upload_error"])
        else:
            st.warning(UI_TEXT[CURRENT_LANG]["upload_error"] + " (Transcription was empty)")

    st.markdown('</div>', unsafe_allow_html=True)