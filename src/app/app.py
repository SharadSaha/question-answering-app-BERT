import streamlit as st
import path
import os

st.set_page_config(page_title="question-answering-app",
                    page_icon=":book:",
                    layout="wide"
)

hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """

st.markdown(hide_st_style,unsafe_allow_html=True)

# ---- MAINPAGE ----
st.title("❓ QueryHELP")
st.markdown("##")


left_column, right_column = st.columns(2)
with left_column:
    st.markdown("* > ### **Question answering :books:**")
    st.markdown("#### ")
with right_column:
    st.markdown("* > ### **Text analysis :bar_chart:**")
    st.markdown("#### ")

st.markdown("""---""")


left_column, right_column = st.columns(2)
with left_column:
    directory = path.Path(__file__).abspath()
    img_path = os.path.join(directory.parent,'images','wordcloud.png')
    st.image(img_path)

with right_column:
    directory = path.Path(__file__).abspath()
    img_path = os.path.join(directory.parent.parent,'images','wordcloud.png')
    st.image(img_path)
