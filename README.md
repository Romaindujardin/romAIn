# romAIn
This document describes the process behind the RomAIn chatbot, an AI designed to answer questions based on personal information about me. The chatbot now supports interactions in both French and English, and can receive input via text or voice, and provide output as text or synthesized speech.

## Access

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://romain.streamlit.app)

or Direct link : https://romain.streamlit.app

## Explanation
![image](https://github.com/user-attachments/assets/a3faf551-cbf4-429d-9531-ff739b90a10c)

### 1. Data Preparation (Offline Process)

- Multilingual Data Collection: First, personal information about Romain is gathered and written down as distinct facts or sentences, separately in both French and English.
  - Example (EN): "My name is Romain Dujardin", "I am 22 years old".
  - Example (FR): "Je m'appelle Romain Dujardin", "J'ai 22 ans".
- Chunking: If some sentences are too long, they might be split into smaller, more focused sub-sentences to ensure the context retrieved later is concise and relevant.
- Embedding: Each of these sentences (in both languages) is then processed by a multilingual embedding model (like paraphrase-multilingual-mpnet-base-v2). This model converts the text sentences into numerical vectors (sequences of numbers, not just 0s and 1s, e.g., [0.12, -0.05, 0.88, ...]). These vectors capture the semantic meaning of the sentences in a way the computer can understand and compare.
- Vector Storage (FAISS Index): All the generated vectors for each language are stored in a dedicated, efficient vector database. Specifically, two separate FAISS indexes are created: one containing the embeddings of the English sentences, and another for the French sentences.
  
This data preparation is done offline. It only needs to be repeated when new information about Romain needs to be added or updated.

### 2. User Interaction & Input Processing (Runtime Process)

- Interface (Streamlit): The user interacts with the chatbot via a Streamlit web interface.
- Initial Choices: The user first selects:
  1. Their desired language (French or English).
  2. Their preferred mode of interaction:
      - Text-to-Text (T2T): Ask in text, get a text answer.
      - Text-to-Speech (T2S): Ask in text, get a spoken answer (and text).
      - Speech-to-Speech (S2S): Ask by voice, get a spoken answer (and text).
- Input Processing:
  - For T2T and T2S: The user types their question directly into the interface. This text question is then vectorized using the same multilingual embedding model used for the data preparation.
  - For S2S:
    - The user either records their voice directly in the interface or uploads an audio file.
    - This audio data is sent to an Automatic Speech Recognition (ASR) model (like Whisper).
    - Whisper transcribes the speech into text.
    - Then, this resulting text transcription is vectorized using the multilingual embedding model.
      
No matter the input method (typed text or transcribed speech), the result at this stage is a query vector representing the user's question.

### 3. Context Retrieval (Finding Relevant Information)

- Index Selection: Based on the language selected by the user at the start, the system chooses the corresponding FAISS index (either the English or the French one).
- Similarity Search: The user's query vector is compared against all the document vectors stored in the selected FAISS index. FAISS efficiently calculates the similarity (or distance) between the query vector and the document vectors.
- Selecting Top Documents: The system retrieves the documents whose vectors are most similar to the query vector. In this case, the top 3 most relevant documents are selected to serve as context for answering the question.
  - Example: For the question "What's your name?", the vector for "My name is Romain Dujardin" would likely be very similar, while the vector for "I am 22 years old" would be less similar, ensuring the most relevant fact is retrieved.

### 4. Generating the Answer (LLM Interaction)

- Prompt Construction: A prompt is carefully constructed to be sent to the Large Language Model (LLM). This prompt typically includes:
  - The retrieved relevant documents (the top 3 sentences identified in the previous step).
  - The user's original question (as text).
  - Custom instructions guiding the LLM on how to answer (e.g., "You are Romain Dujardin. Answer in the first person using only the provided context. Answer in [French/English].").
- LLM Call: This complete prompt is sent to the LLM (e.g., Mistral-7B-Instruct-v0.2) via the Hugging Face API.
- Receiving the Response: The LLM generates a text-based answer based on the provided context and instructions.

### 5. Output Processing & Display

Response Cleaning: The raw text response from the LLM is cleaned. This involves removing potential repetitions of the question or context, or boilerplate phrases (like "Based on the context..."). The goal is to have a natural, direct answer.
Output Delivery (Mode-Dependent):
If T2T: The cleaned text answer is displayed directly on the Streamlit interface.
If T2S or S2S:
The cleaned text answer is also sent to a Text-to-Speech (TTS) model (like Facebook MMS-TTS).
The appropriate language model is selected (e.g., mms-tts-fra for French, mms-tts-eng for English) based on the user's initial language choice.
The TTS model generates an audio waveform of the answer being spoken.
This audio output is then presented to the user for playback via the Streamlit interface (often alongside the displayed text answer).




