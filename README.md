# romAIn
My personal chatbot trained to answer professional questions in English for me.

## Access

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://romain.streamlit.app)

or Direct link : https://romain.streamlit.app

## Explanation
<img width="2607" alt="Architecture" src="https://github.com/user-attachments/assets/0d751d8c-e2d8-464d-9764-5b4bbc35c260" />

### Data 
So for the data, first, I list the data about me in the form of a sentence like “my name is roman”, I split these sentences into several sub-sentences so that I don't have just one long one, but several short ones for a more appropriate context.
and then I put each of these sentences into an embedding model, which means that instead of having sentences with letters like “my name is roman”, I'd have a vector, a sequence of 0s and 1s that allows for a better understanding of the sentence, so “my name is roman” becomes “011100101”.


