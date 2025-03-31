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
# import wave # Plus forcément nécessaire directement
import tempfile
# from st_audiorec import st_audiorec # ### MODIFIÉ ### Ancienne dépendance
from audiorecorder import audiorecorder # ### NOUVEAU ### Nouvelle dépendance
from pydub import AudioSegment # ### NOUVEAU ### Nécessaire pour manipuler la sortie d'audiorecorder

# --- Configuration ---
hf_token = st.secrets["huggingface"]["token"]
client = InferenceClient(api_key=hf_token)
st.set_page_config(layout="wide")

if 'language' not in st.session_state:
    st.session_state['language'] = 'EN'

# Flag pour gérer la première tentative d'enregistrement audio
if 'audio_permission_checked' not in st.session_state:
    st.session_state['audio_permission_checked'] = False

# --- Styling ---
st.markdown(
    """
    <style>
    /* ... styles identiques ... */
    /* Styles spécifiques pour audiorecorder si nécessaire */
    .stAudioRecorder {
        /* Exemple: centrer le bouton/visualiseur */
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem 0;
    }
    /* Vous pouvez essayer d'ajouter des styles ici, mais la portée peut être limitée */
    </style>
    """,
    unsafe_allow_html=True
)

# --- Language selection ---
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

# --- UI Text Definitions ---
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
        # ### MODIFIÉ ### Adapter les instructions si besoin pour le nouveau composant
        'record_instruction': 'Click the icon below to start/stop recording and on the disk to send:',
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
        # Ajout potentiel pour le nouveau composant
        'recorder_visualizer_text': 'Recording...', # Texte affiché si show_visualizer=False avec visualizer activé
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
        # ### MODIFIÉ ### Adapter les instructions si besoin pour le nouveau composant
        'record_instruction': 'Cliquez sur l\'icône ci-dessous pour démarrer/arrêter l\'enregistrement et sur la disquette pour l\'envoyer :',
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
        # Ajout potentiel pour le nouveau composant
        'recorder_visualizer_text': 'Enregistrement...', # Texte affiché si show_visualizer=False avec visualizer activé
    }
}

# --- Model Loading ---
@st.cache_resource
def load_embedding_model():
    print("DEBUG: Loading embedding model...")
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

embedding_model = load_embedding_model()

# --- Document Definitions ---
documents_en = [ # ... contenu EN identique ...
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
documents_fr = [ # ... contenu FR identique ...
    "Je m'appelle Romain Dujardin",
    "J'ai 22 ans", # <-- MODIFIÉ ICI pour une phrase plus naturelle
    "Je suis un étudiant français en école d'ingénieur dans l'IA",
    "J'étudie actuellement à JUNIA ISEN à Lille depuis 2021. Pendant mes études, j'ai appris le machine learning, le deep learning, la vision par ordinateur, le traitement du langage naturel, l'apprentissage par renforcement. J'ai eu des cours de mathématiques, statistiques, informatique, physique, électronique et gestion de projet",
    "Avant Isen JUNIA, j'étais à ADIMAKER, une classe préparatoire intégrée où j'ai appris les bases de l'ingénierie",
    "Je suis passionné par l'intelligence artificielle, les nouvelles technologies et l'informatique", "J'habite à Lille, en France",
    "Concernant mes projets, j'ai notamment travaillé sur le Projet F.R.A.N.K qui est un projet 3D mélangeant l'IA sur unity3D, c'est un jeu d'horreur dans un univers réaliste, avec des fonctions de gameplay avancées comme la gestion d'inventaire et l'utilisation d'objets, tout en étant poursuivi par un monstre sous IA. Et j'ai aussi travaillé sur un projet de drive local sur django nommé DriveMe. Tous ces projets sont disponibles sur mon github",
    "Durant ces différents projets j'ai d'abord appris à gérer une équipe en tant que chef de projet et donc en même temps à travailler en équipe, j'ai également mis en pratique ce que je vois en cours dans des exemples concrets. En plus, j'ai pu traiter la résolution de problèmes sur certains projets",
    "Je recherche une alternance en IA pour septembre 2025", "J'ai besoin d'un contrat pour valider mon diplôme",
    "voici mon email : dujardin.romain@icloud.com et mon numéro de téléphone est le 07 83 19 30 23",
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


# --- FAISS Index Building ---
@st.cache_resource(show_spinner=False)
def build_faiss_index_en():
    print("DEBUG: Building and caching EN FAISS index...")
    docs = documents_en
    doc_embeddings = embedding_model.encode(docs)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    return index, docs

@st.cache_resource(show_spinner=False)
def build_faiss_index_fr():
    print("DEBUG: Building and caching FR FAISS index...")
    docs = documents_fr
    doc_embeddings = embedding_model.encode(docs)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    return index, docs

# --- Select the correct index ---
if CURRENT_LANG == 'FR':
    index, current_documents = build_faiss_index_fr()
    print("DEBUG: Using FR index.")
else:
    index, current_documents = build_faiss_index_en()
    print("DEBUG: Using EN index.")


# --- find_relevant_docs ---
def find_relevant_docs(query, k=3): # ### Augmenté k à 3 ###
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)

    print(f"DEBUG Query: {query}")
    print(f"DEBUG Found indices: {indices[0]}, Distances: {distances[0]}")

    threshold = 15 # Ajustez si nécessaire
    relevant_docs = []
    relevant_distances = []

    if distances.size > 0:
        for i, idx in enumerate(indices[0]):
            if idx < len(current_documents) and distances[0][i] <= threshold:
                relevant_docs.append(current_documents[idx])
                relevant_distances.append(distances[0][i])

    print(f"DEBUG Retrieved docs AFTER threshold: {relevant_docs}")
    print(f"DEBUG Retrieved distances AFTER threshold: {relevant_distances}")

    if not relevant_docs:
        return [], []
    return relevant_docs, relevant_distances


# --- Fonctions API (Mistral, Transcribe, TTS) ---
def mistral_via_api(prompt, lang='EN'):
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    if hf_token is None: return "Error: No tokens found."
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.5, "top_k": 10, "return_full_text": False}}
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

def transcribe_audio(audio_data, lang='EN'):
    # Cette fonction attend des bytes en entrée
    try:
        model = "openai/whisper-large-v3"
        transcription_response = client.automatic_speech_recognition(audio=audio_data, model=model)
        if isinstance(transcription_response, dict) and 'text' in transcription_response: return transcription_response['text']
        elif isinstance(transcription_response, str): return transcription_response
        else: st.error(f"Unexpected transcription response format: {transcription_response}"); return None
    except Exception as e: st.error(f"Error during transcription: {str(e)}"); return None

def text_to_speech(text, lang='EN'):
    try:
        voice = "facebook/mms-tts-fra" if lang == 'FR' else "facebook/mms-tts-eng"
        audio_bytes = client.text_to_speech(text, model=voice)
        return audio_bytes
    except Exception as e: st.error(f"{UI_TEXT[lang].get('tts_error', 'TTS Error:')} {str(e)}"); return None

# --- RAG Pipeline ---
def rag_pipeline(query, k=3, lang='EN'):
    relevant_docs, distances = find_relevant_docs(query, k=k)

    print("-" * 20)
    print(f"RAG Pipeline Input Query: {query}")
    print(f"RAG Found Relevant Docs (k={k}, after threshold):")
    if relevant_docs:
        for doc, dist in zip(relevant_docs, distances):
             print(f"  - Distance: {dist:.4f}, Doc: {doc[:100]}...")
    else:
        print("  - No relevant documents found after threshold.")
    print("-" * 20)

    no_info_msg = UI_TEXT[lang].get('response_no_info', ("Je suis désolé, je ne peux pas repondre à cette question..." if lang == 'FR' else "I'm sorry, I cannot answer this question..."))
    if not relevant_docs:
        print("RAG Pipeline: No relevant docs found, returning no_info_msg")
        return no_info_msg

    context = "\n".join(relevant_docs)

    if lang == 'FR':
        prompt = f"""Contexte: {context}\n\nQuestion: {query}\n\nTu es Romain Dujardin... Réponds à la première personne en utilisant seulement les informations du contexte fourni... Si le contexte ne permet pas de répondre, dis le clairement (par exemple: 'Je n'ai pas l'information pour répondre à cela.'). N'invente rien.\n\nRéponse:"""
    else:
        prompt = f"""Context: {context}\n\nQuestion: {query}\n\nYou are Romain Dujardin... Answer in the first person using only information from the provided context... If the context doesn't provide the answer, state that clearly (e.g., 'I don't have the information to answer that.'). Do not invent anything.\n\nAnswer:"""

    print(f"RAG Pipeline: Prompt sent to Mistral:\n{prompt}\n")

    response_text = mistral_via_api(prompt, lang)

    if response_text is None:
        print("RAG Pipeline: Mistral API returned None.")
        return UI_TEXT[lang].get('response_generation_error', "Sorry, error generating response.")

    answer = response_text.strip()
    print(f"RAG Pipeline: Raw response from Mistral: '{answer}'")

    # --- Cleaning ---
    if lang == 'FR':
        unwanted = ["En tant que Romain Dujardin,", "En tant que Romain, ", "Basé sur le contexte,", "Selon le contexte,", "D'après le contexte,", "Réponse:", "..."]
        answer = answer.replace("Romain est", "Je suis").replace("Romain a", "J'ai").replace("Romain", "Romain")
        answer = answer.replace("il est", "je suis").replace("il a", "j'ai")
    else:
        unwanted = ["As Romain Dujardin,", "As Romain, ", "Based on the context,", "According to the context,", "Answer:", "..."]
        answer = answer.replace("Romain's", "my").replace("Romain is", "I am").replace("Romain has", "I have").replace("Romain", "Romain")
        answer = answer.replace("he is", "I am").replace("he has", "I have")

    cleaned_answer = answer
    for phrase in unwanted:
        cleaned_answer = re.sub(rf'^\s*{re.escape(phrase)}\s*', '', cleaned_answer, flags=re.IGNORECASE)

    print(f"RAG Pipeline: Cleaned answer: '{cleaned_answer}'")

    if not cleaned_answer or cleaned_answer.lower().strip() in ["answer:", "réponse:", ".", ""]:
        print("RAG Pipeline: Answer became empty/uninformative after cleaning, returning no_info_msg")
        return no_info_msg

    no_answer_indicators_fr = ["je ne sais pas", "je n'ai pas l'information", "contexte ne fournit pas", "pas mentionné"]
    no_answer_indicators_en = ["i don't know", "i do not know", "context does not provide", "not mentioned", "don't have the information"]
    indicators = no_answer_indicators_fr if lang == 'FR' else no_answer_indicators_en
    if any(indicator in cleaned_answer.lower() for indicator in indicators):
         print(f"RAG Pipeline: LLM indicated it couldn't answer. Returning: '{cleaned_answer}'")
         # pass # Keep the LLM's answer indicating it doesn't know

    return cleaned_answer

# --- Main App Layout ---
st.markdown(
    f'<div style="text-align: center;">'
    f'<h1 class="centered h1">{UI_TEXT[CURRENT_LANG]["title"]}</h1>'
    f'<p class="centered p">{UI_TEXT[CURRENT_LANG]["subtitle"]}</p>'
    f'</div>',
    unsafe_allow_html=True
)

tab_titles = [UI_TEXT[CURRENT_LANG]["text_tab"], UI_TEXT[CURRENT_LANG]["voice_tab"]]
tabs = st.tabs(tab_titles)

# --- Text Input Tab ---
with tabs[0]:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    query = st.text_input(UI_TEXT[CURRENT_LANG]["question_placeholder"], key="text_query_input")
    if query:
        with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
            answer = rag_pipeline(query, lang=CURRENT_LANG, k=3)
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.write(answer)
        st.markdown('</div>', unsafe_allow_html=True)
        if st.button(UI_TEXT[CURRENT_LANG]["listen_button"], key="text_listen_button"):
            with st.spinner(UI_TEXT[CURRENT_LANG]["generating_audio"]):
                audio_bytes = text_to_speech(answer, lang=CURRENT_LANG)
                if audio_bytes: st.audio(audio_bytes, format="audio/wav")
                else: st.error(UI_TEXT[CURRENT_LANG]['audio_playback_error'])
    st.markdown('</div>', unsafe_allow_html=True)


# --- Voice Input Tab ---
with tabs[1]:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.subheader(UI_TEXT[CURRENT_LANG]["voice_subtitle"])

    audio_player_container = st.container()
    text_results_container = st.container()
    audio_message_placeholder = st.empty() # Placeholder for messages

    st.markdown(f"<p style='text-align: center;'>{UI_TEXT[CURRENT_LANG]['record_instruction']}</p>", unsafe_allow_html=True)

    # ### NOUVEAU ### Utilisation de audiorecorder avec le mode visualiseur (pas de texte pour les boutons)
    # Vous pouvez aussi mettre du texte: audiorecorder("Démarrer", "Arrêter")
    # show_visualizer=True est la valeur par défaut quand les prompts sont vides.
    # Mettre show_visualizer=False afficherait le texte 'Recording...'
    recorded_audio_segment = audiorecorder("", "", key="audio_recorder_main")

    # ### MODIFIÉ ### Logique pour traiter la sortie de audiorecorder (AudioSegment)
    if recorded_audio_segment and len(recorded_audio_segment) > 0:
        # L'enregistrement a produit un segment audio non vide
        st.session_state['audio_permission_checked'] = True
        audio_message_placeholder.empty() # Efface les messages précédents

        transcription = None
        try:
            with st.spinner(UI_TEXT[CURRENT_LANG]["processing"]):
                # ### MODIFIÉ ### Exporter l'AudioSegment en bytes (format WAV recommandé)
                audio_bytes_to_transcribe = recorded_audio_segment.export(format="wav").read()
                transcription = transcribe_audio(audio_bytes_to_transcribe, lang=CURRENT_LANG)
        except Exception as e:
             audio_message_placeholder.error(f"{UI_TEXT[CURRENT_LANG]['processing_error']} {e}")
             transcription = None

        if transcription and isinstance(transcription, str) and transcription.strip():
            # Vider les anciens résultats si une nouvelle transcription est réussie
            audio_player_container.empty()
            with text_results_container: st.empty() # Assure que l'expander est recréé

            with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
                answer = rag_pipeline(transcription, lang=CURRENT_LANG, k=3)

            with text_results_container.expander(UI_TEXT[CURRENT_LANG]["show_details"], expanded=False): # Ouvre l'expander par défaut
                st.write(f"**{UI_TEXT[CURRENT_LANG]['your_question']}**")
                st.write(transcription)
                st.write(f"**{UI_TEXT[CURRENT_LANG]['response']}**")
                st.write(answer)

            with st.spinner(UI_TEXT[CURRENT_LANG]["generating_voice"]):
                audio_response = text_to_speech(answer, lang=CURRENT_LANG)
                if audio_response:
                    with audio_player_container:
                         st.markdown(f"**{UI_TEXT[CURRENT_LANG]['listen_button']}**") # Ajout titre pour clarté
                         st.audio(audio_response, format="audio/wav")
                else:
                     audio_message_placeholder.warning(UI_TEXT[CURRENT_LANG]['audio_playback_error'])

        # Gère les cas où la transcription a échoué APRES un enregistrement valide
        elif not transcription:
             audio_message_placeholder.error(UI_TEXT[CURRENT_LANG]["no_transcription"])
        elif not transcription.strip():
             audio_message_placeholder.warning(UI_TEXT[CURRENT_LANG]["no_transcription"] + " (Transcription was empty)")

    # ### MODIFIÉ ### Gérer le cas où l'enregistrement a échoué ou est vide
    # recorded_audio_segment sera None ou un AudioSegment de longueur 0
    elif recorded_audio_segment is not None: # Si le composant a retourné quelque chose (même vide)
        if not st.session_state['audio_permission_checked']:
            # Probablement la première interaction après demande de permission
            audio_message_placeholder.info(
                "🎤 Accordez l'accès au Microphone ! Cliquez une premiere fois pour donner l'accès et une seconde fois pour commencer l'enregistrement"
                if CURRENT_LANG == 'FR' else
                "🎤 Give access to the microphone! Click a first time to give access and a second time to start recording"
            )
            st.session_state['audio_permission_checked'] = True
        else:
            # Ce n'est pas la première tentative OU l'utilisateur a arrêté très vite
            # On ne peut pas distinguer facilement un vrai échec d'un arrêt immédiat.
            # On peut juste indiquer que l'audio reçu est vide.
            # audio_message_placeholder.warning(UI_TEXT[CURRENT_LANG]["recording_failed"] + " (Audio data is empty)")
            # Optionnellement: ne rien afficher si l'audio est juste vide (l'utilisateur a peut-être annulé)
            pass # Ne rien afficher explicitement pour un enregistrement vide après la première tentative

    # --- Upload Option (inchangé mais utilise k=3 dans rag_pipeline) ---
    st.markdown("---")
    uploaded_file = st.file_uploader(
        UI_TEXT[CURRENT_LANG]["upload_audio"],
        type=["mp3", "wav", "m4a", "ogg", "flac"],
        key="audio_uploader"
    )
    if uploaded_file is not None:
        # Vider les résultats précédents (enregistrement ou upload)
        audio_player_container.empty()
        with text_results_container: st.empty()
        audio_message_placeholder.empty()

        transcription = None
        try:
            uploaded_bytes = uploaded_file.getvalue()
            with st.spinner(UI_TEXT[CURRENT_LANG]["processing_upload"]): # Message différent
                transcription = transcribe_audio(uploaded_bytes, lang=CURRENT_LANG)
        except Exception as e:
            st.error(f"{UI_TEXT[CURRENT_LANG]['upload_process_error']} {e}")
            transcription = None

        if transcription and isinstance(transcription, str) and transcription.strip():
            with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
                answer = rag_pipeline(transcription, lang=CURRENT_LANG, k=3)

            with text_results_container.expander(UI_TEXT[CURRENT_LANG]["show_details"], expanded=True):
                st.write(f"**{UI_TEXT[CURRENT_LANG]['your_question']}**")
                st.write(transcription)
                st.write(f"**{UI_TEXT[CURRENT_LANG]['response']}**")
                st.write(answer)

            with st.spinner(UI_TEXT[CURRENT_LANG]["generating_voice"]):
                audio_response = text_to_speech(answer, lang=CURRENT_LANG)
                if audio_response:
                     with audio_player_container:
                         st.markdown(f"**{UI_TEXT[CURRENT_LANG]['listen_button']}**")
                         st.audio(audio_response, format="audio/wav")
                else:
                     st.warning(UI_TEXT[CURRENT_LANG]['audio_playback_error']) # Afficher l'erreur dans le main container ici

        elif transcription is None:
            st.error(UI_TEXT[CURRENT_LANG]["upload_error"]) # Afficher l'erreur dans le main container ici
        else:
            st.warning(UI_TEXT[CURRENT_LANG]["upload_error"] + " (Transcription was empty)") # Afficher l'erreur ici

    st.markdown('</div>', unsafe_allow_html=True)
