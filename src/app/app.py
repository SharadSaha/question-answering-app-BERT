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

from PIL import Image
directory = path.Path(__file__).abspath()
img_path1 = os.path.join(directory.parent.parent,'images','bert.png')
image1 = Image.open(img_path1)
img_path2 = os.path.join(directory.parent.parent,'images','text.png')
image2 = Image.open(img_path2)

left_column, right_column = st.columns(2)
with left_column:
    st.image(image1,width=350)

with right_column:
    st.image(image2,width=280)