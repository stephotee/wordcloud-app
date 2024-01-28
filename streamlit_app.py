import streamlit as st
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import random
import nltk

# Required for text processing in NLTK
nltk.download('punkt')

# Function to generate a color based on the "Colorful" option
def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = random_state.randint(0, 360)
    s = random_state.randint(70, 100)
    l = random_state.randint(40, 90)
    return f"hsl({h}, {s}%, {l}%)"

# Function to generate word cloud
def generate_word_cloud(text, max_words, color_scheme, text_case, additional_stop_words):
    # Tokenization
    tokens = word_tokenize(text)

    # Stop words removal
    stop_words = set(STOPWORDS).union(set(additional_stop_words))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Text case conversion
    tokens = [word.upper() if text_case == 'Upper case' else word.lower() for word in tokens]

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        max_words=max_words,
        color_func=random_color_func if color_scheme == 'Colourful text' else lambda *args, **kwargs: "black",
        background_color='white'
    ).generate(' '.join(tokens))

    # Display the generated image:
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    
    # Save the plot to a BytesIO object to display and download
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    st.image(buf)
    
    # Return the buffer to use for the download link
    return buf

# Streamlit app layout
st.title('Word Cloud Generator')

# Text input
text_input = st.text_area("Paste text here (use commas to separate words or phrases)")

# File upload
uploaded_file = st.file_uploader("Or upload your data (.csv or .txt)", type=['csv', 'txt'])

# Process uploaded file
if uploaded_file is not None:
    try:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            # Drop rows with NaN values to prevent 'NaN' in word cloud
            df.dropna(inplace=True)
            text_input = ' '.join(df.iloc[:, 0].astype(str).tolist())
        elif uploaded_file.type == "text/plain":
            text_input = uploaded_file.read().decode('utf-8')
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Sidebar for additional controls
max_words = st.sidebar.slider("Number of words", 5, 100, 50, 5)
color_scheme = st.sidebar.selectbox("Text color", options=['Black text', 'Colourful text'])
text_case = st.sidebar.radio("Text case", ('Upper case', 'Lower case'))
additional_stop_words_input = st.sidebar.text_input("Additional stop words", value='')

# Process the additional stop words, removing any spaces around the words
additional_stop_words = [word.strip() for word in additional_stop_words_input.split(',') if word.strip()]

# Button to generate word cloud
if st.button('Generate Word Cloud'):
    buf = generate_word_cloud(text_input, max_words, color_scheme, text_case, additional_stop_words)
    # Create a download link for the generated word cloud
    st.sidebar.download_button(
        label="Download PNG",
        data=buf,
        file_name="wordcloud.png",
        mime="image/png"
    )
