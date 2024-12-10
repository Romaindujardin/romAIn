import streamlit as st

def main():
    st.markdown(
        """
        <style>
        .centered {
            text-align: center;
        }
        .h1 {
            font-size: 7em;
        }
        .p {
            font-size: 1em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h1 class="centered h1">Welcome to, <span style="opacity: 0.5;">rom</span>A</span>I<span style="opacity: 0.5;">n</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="centered p">here is <span style="opacity: 0.5;">rom</span>A</span>I<span style="opacity: 0.5;">n</span>, an AI in the image of Romain Dujardin. Ask him questions in English and he will answer them as best he can.</p>', unsafe_allow_html=True)
    st.markdown('<p class="centered p">Can be made mistake</p>', unsafe_allow_html=True)

    
    if st.button("Go to romAIn", key="centered_button"):
        st.session_state.page = "main"

    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            display: block;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()