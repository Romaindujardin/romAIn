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
# import pyaudio # Removed - No longer needed for recording
import wave
import tempfile
from streamlit_audiorec import st_audiorec # Added for client-side recording

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
    /* Styling for streamlit-audiorec is handled by the component itself */
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

# Language selection in the top-right corner
# Note: The JavaScript approach for language switching might be less reliable
# in Streamlit Cloud compared to standard Streamlit components.
# The st.radio approach below is generally preferred.

# Handle language selection from JavaScript (Keep if needed, but st.radio is more robust)
if 'selected_language' in st.session_state:
    st.session_state['language'] = st.session_state['selected_language']

# Alternative language selector using Streamlit components
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    lang_options = ["🇫🇷 FR", "🇬🇧 EN"]
    selected_lang_option = st.radio(
        "Language / Langue",
        lang_options,
        index=1 if st.session_state['language'] == 'EN' else 0,
        horizontal=True,
        label_visibility="collapsed" # Hide the label if desired
    )
    # Update session state only if the selection changed
    new_lang = 'EN' if selected_lang_option == "🇬🇧 EN" else 'FR'
    if st.session_state['language'] != new_lang:
        st.session_state['language'] = new_lang
        st.experimental_rerun() # Rerun to apply language change immediately

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
        'recording': 'Recording... Click stop when done.', # Updated message
        'processing': 'Processing audio...', # Updated message
        'generating_voice': 'Generating voice response...',
        'show_details': 'Show text details',
        'your_question': 'Your question:',
        'response': 'Response:',
        'no_transcription': 'No transcription was generated. Please try again.',
        'recording_failed': 'Audio recording failed or no audio received.', # Updated message
        'upload_audio': 'Or upload an audio file',
        'processing_upload': 'Processing uploaded audio...',
        'upload_error': 'No transcription was generated from the uploaded file.',
        'record_instruction': 'Click the microphone icon below to start/stop recording:' # New instruction
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
        'recording': 'Enregistrement en cours... Cliquez sur stop une fois terminé.', # Updated message
        'processing': 'Traitement de l\'audio...', # Updated message
        'generating_voice': 'Génération de la réponse vocale...',
        'show_details': 'Afficher les détails du texte',
        'your_question': 'Votre question:',
        'response': 'Réponse:',
        'no_transcription': 'Aucune transcription n\'a été générée. Veuillez réessayer.',
        'recording_failed': 'L\'enregistrement audio a échoué ou aucun audio n\'a été reçu.', # Updated message
        'upload_audio': 'Ou téléchargez un fichier audio',
        'processing_upload': 'Traitement de l\'audio téléchargé...',
        'upload_error': 'Aucune transcription n\'a été générée à partir du fichier téléchargé.',
        'record_instruction': 'Cliquez sur l\'icône micro ci-dessous pour démarrer/arrêter l\'enregistrement :' # New instruction
    }
}

# Initialisation du modèle d'embedding et de l'index FAISS
@st.cache_resource # Cache the model and index
def load_models_and_index(lang):
    print(f"Loading models and index for language: {lang}")
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    # Define language-specific documents
    documents_en = [
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
    documents_fr = [
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

    documents = documents_fr if lang == 'FR' else documents_en
    doc_embeddings = embedding_model.encode(documents)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)
    return embedding_model, index, documents

embedding_model, index, documents = load_models_and_index(CURRENT_LANG)

def find_relevant_docs(query, k=2):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k)

    threshold = 1.41
    if not indices.size or distances[0][0] > threshold: # Check if indices is empty
        return [], []

    relevant_indices = [idx for i, idx in enumerate(indices[0]) if distances[0][i] <= threshold]
    relevant_distances = [dist for dist in distances[0] if dist <= threshold]

    if not relevant_indices:
        return [], []

    return [documents[idx] for idx in relevant_indices], relevant_distances

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
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=45) # Added timeout
        response.raise_for_status() # Raise exception for bad status codes
        return response.json()[0]["generated_text"]
    except requests.exceptions.RequestException as e:
        error_msg = "Error calling Mistral API: " if lang == 'EN' else "Erreur lors de l'appel à l'API Mistral: "
        st.error(f"{error_msg} {e}")
        print(f"Mistral API Error: {e}") # Log detailed error
        if hasattr(e, 'response') and e.response is not None:
             print(f"Mistral API Response: {e.response.text}")
        return None # Return None on error

# Improved RAG pipeline with first-person response and language support
def rag_pipeline(query, k=2, lang='EN'):
    relevant_docs, distances = find_relevant_docs(query, k)

    if not relevant_docs:
        if lang == 'FR':
            return "Je suis désolé, je n'ai pas assez d'informations pertinentes pour répondre précisément à cette question."
        else:
            return "I'm sorry, I don't have enough relevant information to answer that question precisely."

    context = "\n".join(relevant_docs)

    if lang == 'FR':
        prompt = f"""Contexte: {context}

Question: {query}

Tu es Romain Dujardin. Réponds à la question à la première personne ('Je', 'mon', 'mes'). Utilise uniquement les informations fournies dans le contexte ci-dessus. Ne mentionne jamais "le contexte". Sois direct et conversationnel. Si la réponse n'est pas dans le contexte, dis que tu n'as pas l'information.

Réponse:"""
    else:
        prompt = f"""Context: {context}

Question: {query}

You are Romain Dujardin. Answer the question in the first person ('I', 'my', 'me'). Use only the information provided in the context above. Never mention "the context". Be direct and conversational. If the answer is not in the context, state that you don't have the information.

Answer:"""

    response_text = mistral_via_api(prompt, lang)

    if response_text is None: # Handle API errors
         return "Sorry, I encountered an error while generating the response." if lang == 'EN' else "Désolé, j'ai rencontré une erreur en générant la réponse."


    # Extract the answer after "Answer:" or "Réponse:"
    search_pattern = r"Réponse:\s*(.*)" if lang == 'FR' else r"Answer:\s*(.*)"
    answer_match = re.search(search_pattern, response_text, re.DOTALL | re.IGNORECASE) # Added IGNORECASE

    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback: Take text after the prompt if pattern fails
        parts = response_text.split(prompt)
        if len(parts) > 1:
            answer = parts[1].strip()
        else:
            # If even splitting fails, use the whole response (might contain the prompt)
            answer = response_text.strip()
            # Try to remove the prompt manually if it's still there
            if answer.startswith(prompt):
                 answer = answer[len(prompt):].strip()


    # Additional cleaning
    common_unwanted_start_phrases = [
        "En tant que Romain Dujardin,", "As Romain Dujardin,",
        "En tant que Romain,", "As Romain,",
        "Basé sur le contexte,", "Based on the context,",
        "Selon les informations,", "According to the information,",
        "D'après les informations fournies,", "Based on the information provided,",
    ]
    # Clean start phrases
    for phrase in common_unwanted_start_phrases:
        if answer.lower().startswith(phrase.lower()):
            answer = answer[len(phrase):].strip()

    # Correct common third-person references (more robustly)
    if lang == 'FR':
        answer = re.sub(r'\bRomain Dujardin est\b', 'Je suis', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\bRomain est\b', 'Je suis', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\bRomain a\b', 'J\'ai', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\bde Romain\b', 'mon', answer, flags=re.IGNORECASE) # Simple replacement for possessive
    else:
        answer = re.sub(r'\bRomain Dujardin is\b', 'I am', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\bRomain is\b', 'I am', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\bRomain has\b', 'I have', answer, flags=re.IGNORECASE)
        answer = re.sub(r'\bRomain\'s\b', 'my', answer, flags=re.IGNORECASE)

    # Check for refusal phrases if no relevant docs were initially found (redundancy check)
    no_info_phrases_en = ["I don't have information about", "There is no information about", "The context doesn't mention"]
    no_info_phrases_fr = ["Je n'ai pas d'informations sur", "Il n'y a pas d'informations sur", "Le contexte ne mentionne pas"]
    no_info_phrases = no_info_phrases_fr if lang == 'FR' else no_info_phrases_en

    for phrase in no_info_phrases:
         if phrase.lower() in answer.lower():
             # Return a standard 'no info' message if the LLM indicates lack of context
             return "Je suis désolé, je n'ai pas l'information nécessaire pour répondre à cela." if lang == 'FR' else "I'm sorry, I don't have the necessary information to answer that."


    # Final trim
    answer = answer.strip()

    return answer

# Function to transcribe audio with Whisper with language support
def transcribe_audio(audio_bytes, lang='EN'): # Modified to accept bytes
    try:
        # Choose the appropriate model based on language - large-v3 is good
        # Consider smaller models like base or medium if latency is an issue,
        # but large-v3 provides better accuracy.
        model = "openai/whisper-large-v3" # Changed from turbo, check available models if needed

        # The InferenceClient expects bytes or a file path. Pass bytes directly.
        transcription_result = client.automatic_speech_recognition(
            audio=audio_bytes, # Pass the bytes directly
            model=model
        )
        # Check the structure of the result, usually it's in a 'text' field
        if isinstance(transcription_result, dict) and 'text' in transcription_result:
            return transcription_result['text']
        elif isinstance(transcription_result, str): # Some API versions might return string directly
             return transcription_result
        else:
            print(f"Unexpected transcription result format: {transcription_result}")
            return None

    except Exception as e:
        print(f"Erreur lors de la transcription: {str(e)}")
        st.error(f"Transcription Error: {str(e)}")
        return None


# Function to generate audio from text with language support
def text_to_speech(text, lang='EN'):
    try:
        # Choose the appropriate voice model based on language
        voice = "facebook/mms-tts-fra" if lang == 'FR' else "facebook/mms-tts-eng"

        # Use the Hugging Face API for speech synthesis
        audio_content = client.text_to_speech(text, model=voice) # Returns bytes
        return audio_content # Return the bytes directly

    except Exception as e:
        error_msg = "Error during speech synthesis: " if lang == 'EN' else "Erreur lors de la synthèse vocale: "
        st.error(f"{error_msg} {str(e)}")
        print(f"TTS Error: {e}") # Log detailed error
        return None

# Removed the pyaudio record_audio function

# Improved audio player function with unique ID generation
# Ensure this function handles bytes correctly
def get_audio_player_html(audio_bytes):
    if audio_bytes is None:
        return None

    # Generate a truly unique ID using timestamp and maybe a random element
    unique_id = f"audio_{int(time.time() * 1000)}_{np.random.randint(1000)}"

    # Encode to base64 for HTML embedding
    b64 = base64.b64encode(audio_bytes).decode()

    # Create HTML audio player with unique ID and autoplay
    # The 'type' depends on the output format of the TTS API.
    # Hugging Face TTS often outputs WAV or MP3. Assuming WAV here.
    # If it's MP3, change type="audio/wav" to type="audio/mpeg".
    audio_player = f"""
    <audio id="{unique_id}" controls autoplay>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    <script>
        var audio = document.getElementById("{unique_id}");
        if (audio) {{
            // Attempt to play, catching potential errors if autoplay is blocked
            var playPromise = audio.play();
            if (playPromise !== undefined) {{
                playPromise.catch(error => {{
                    console.log("Autoplay prevented: ", error);
                    // Optionally, show a play button or message to the user
                }});
            }}
        }} else {{
            console.error("Audio element not found: {unique_id}");
        }}
    </script>
    """

    return audio_player

# Display the main page content
st.markdown(f'<h1 class="centered h1">{UI_TEXT[CURRENT_LANG]["title"]}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="centered p">{UI_TEXT[CURRENT_LANG]["subtitle"]}</p>', unsafe_allow_html=True)

# Create tabs for different interaction methods
tabs = st.tabs([UI_TEXT[CURRENT_LANG]["text_tab"], UI_TEXT[CURRENT_LANG]["voice_tab"]])

# --- Text Input Tab ---
with tabs[0]:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    query = st.text_input(UI_TEXT[CURRENT_LANG]["question_placeholder"], key="text_query_input")

    if query:
        with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
            answer = rag_pipeline(query, lang=CURRENT_LANG)

        # Display the text response
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.write(answer)
        st.markdown('</div>', unsafe_allow_html=True)

        # Option to listen to the answer
        if st.button(UI_TEXT[CURRENT_LANG]["listen_button"], key="text_listen_button"):
            with st.spinner(UI_TEXT[CURRENT_LANG]["generating_audio"]):
                audio_bytes_response = text_to_speech(answer, lang=CURRENT_LANG)
                if audio_bytes_response:
                    st.markdown(get_audio_player_html(audio_bytes_response), unsafe_allow_html=True)
                else:
                    st.warning("Could not generate audio for the answer.") # User feedback
    st.markdown('</div>', unsafe_allow_html=True)

# --- Voice Input Tab ---
with tabs[1]:
    st.markdown('<div class="tab-container">', unsafe_allow_html=True)
    st.subheader(UI_TEXT[CURRENT_LANG]["voice_subtitle"])
    st.markdown(UI_TEXT[CURRENT_LANG]["record_instruction"]) # Add instruction

    # Use streamlit-audiorec for recording
    # This function returns audio bytes when the user stops recording
    audio_bytes_recorded = st_audiorec()

    # Create containers for results that persist across reruns
    if 'audio_player_container_voice' not in st.session_state:
        st.session_state.audio_player_container_voice = st.empty()
    if 'text_results_container_voice' not in st.session_state:
        st.session_state.text_results_container_voice = st.empty()


    # Process recorded audio if bytes are received
    if audio_bytes_recorded:
        st.info(UI_TEXT[CURRENT_LANG]["processing"]) # Indicate processing starts

        # No need for sample rate here, as transcribe_audio handles bytes directly
        # No need to save to a temporary file if transcribe_audio accepts bytes
        with st.spinner(UI_TEXT[CURRENT_LANG]["processing"]):
            transcription = transcribe_audio(audio_bytes_recorded, lang=CURRENT_LANG)

        if transcription and transcription.strip(): # Check if transcription is not empty
            st.success(f"Transcription: {transcription}") # Show transcription briefly

            with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
                answer = rag_pipeline(transcription, lang=CURRENT_LANG)

            with st.spinner(UI_TEXT[CURRENT_LANG]["generating_voice"]):
                audio_response = text_to_speech(answer, lang=CURRENT_LANG)

                if audio_response:
                    # Update the audio player container
                    audio_player_html = get_audio_player_html(audio_response)
                    st.session_state.audio_player_container_voice.markdown(audio_player_html, unsafe_allow_html=True)

                    # Update the text container with expandable details
                    with st.session_state.text_results_container_voice.expander(UI_TEXT[CURRENT_LANG]["show_details"], expanded=True):
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['your_question']}**")
                        st.write(transcription)
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['response']}**")
                        st.write(answer)
                else:
                    # If TTS failed, still show text response
                    with st.session_state.text_results_container_voice.expander(UI_TEXT[CURRENT_LANG]["show_details"], expanded=True):
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['your_question']}**")
                        st.write(transcription)
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['response']}**")
                        st.write(answer)
                    st.warning("Could not generate audio response, showing text instead.")

        else:
            st.error(UI_TEXT[CURRENT_LANG]["no_transcription"])
            # Clear previous results if transcription fails
            st.session_state.audio_player_container_voice.empty()
            st.session_state.text_results_container_voice.empty()

    # --- File Upload Section (as an alternative) ---
    st.markdown("---") # Separator
    uploaded_file = st.file_uploader(UI_TEXT[CURRENT_LANG]["upload_audio"], type=["mp3", "wav", "m4a", "ogg"], key="voice_file_uploader")

    if uploaded_file is not None:
        st.info(UI_TEXT[CURRENT_LANG]["processing_upload"])
        # Read bytes from uploaded file
        uploaded_audio_bytes = uploaded_file.read()

        with st.spinner(UI_TEXT[CURRENT_LANG]["processing"]):
            transcription = transcribe_audio(uploaded_audio_bytes, lang=CURRENT_LANG)

        if transcription and transcription.strip():
            st.success(f"Transcription: {transcription}") # Show transcription

            with st.spinner(UI_TEXT[CURRENT_LANG]["thinking"]):
                answer = rag_pipeline(transcription, lang=CURRENT_LANG)

            with st.spinner(UI_TEXT[CURRENT_LANG]["generating_voice"]):
                audio_response = text_to_speech(answer, lang=CURRENT_LANG)

                if audio_response:
                    # Update the audio player container
                    audio_player_html = get_audio_player_html(audio_response)
                    st.session_state.audio_player_container_voice.markdown(audio_player_html, unsafe_allow_html=True)

                    # Update the text container
                    with st.session_state.text_results_container_voice.expander(UI_TEXT[CURRENT_LANG]["show_details"], expanded=True):
                         st.write(f"**{UI_TEXT[CURRENT_LANG]['your_question']} (uploaded)**")
                         st.write(transcription)
                         st.write(f"**{UI_TEXT[CURRENT_LANG]['response']}**")
                         st.write(answer)
                else:
                     # If TTS failed, still show text response
                    with st.session_state.text_results_container_voice.expander(UI_TEXT[CURRENT_LANG]["show_details"], expanded=True):
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['your_question']} (uploaded)**")
                        st.write(transcription)
                        st.write(f"**{UI_TEXT[CURRENT_LANG]['response']}**")
                        st.write(answer)
                    st.warning("Could not generate audio response, showing text instead.")

        else:
            st.error(UI_TEXT[CURRENT_LANG]["upload_error"])
            # Clear previous results if upload processing fails
            st.session_state.audio_player_container_voice.empty()
            st.session_state.text_results_container_voice.empty()


    st.markdown('</div>', unsafe_allow_html=True)

# Remove the JavaScript snippet for receiving audio data, as streamlit-audiorec handles this internally.

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