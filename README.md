# romAIn
My personal chatbot trained to answer professional questions in English for me.

## Access

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://romain.streamlit.app)

or Direct link : https://romain.streamlit.app

## Explanation
<img width="2607" alt="Architecture" src="https://github.com/user-attachments/assets/0d751d8c-e2d8-464d-9764-5b4bbc35c260" />

### Data 
So for the data, first, I list the data about me in the form of a sentence like “my name is romain”, I split these sentences into several sub-sentences so that I don't have just one long one, but several short ones for a more appropriate context.

and then I put each of these sentences into an embedding model, which means that instead of having sentences with letters like “my name is romain”, I'd have a vector, a sequence of 0s and 1s that allows for a better understanding of the sentence, so “my name is roman” becomes “011100101”.

then I store it all in a database

I repeat these steps as soon as I have new information about myself, so they don't have to be repeated every time.


### Input
then i'll do the same thing with the user's question. the question is put in vector form like my input data


### Best document for context
Once our input data and our question are in the right form, we need to determine which input data is more relevant to answer. For example, to answer the question “what's your name?” the input “I'm 22” isn't really of interest, so the aim is to filter the documents to select the most relevant ones. In my case, I decided to select 3 of the most relevant.

for this I use a FAISS index and a function to search for relevant documents according to the embedding of the question, the embedding of the documents and the index


### Sent to LLM
once we've determined which documents are most relevant to answering the question, we send the question + the documents in question + a personalized prompt to optimize the answer. 

As for LLM, I use Mistral-7B-Instruct-v0.2 via hugging face, I use the api for communication

### Show answer on web interface
Once the answer is returned, I filter it to prevent it from repeating the question or context and display it on the streamlite page.




