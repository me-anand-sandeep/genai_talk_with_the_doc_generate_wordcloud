# pip install wordcloud
# pip install PyPDF2 -- for manipulating PDF files
# pip install python-docx -- The python-docx library is used for creating, updating, and reading Word (.docx) files in Python.
# pip install 
# pip install contractions



import streamlit as st
import numpy as np
import pandas as pd

import seaborn as sns
import PyPDF2
from docx import Document

import base64
from io import BytesIO
from wordcloud import WordCloud
from collections import Counter

import nltk
from nltk.corpus import stopwords
import contractions
import re
import warnings
warnings.filterwarnings("ignore")

import anthropic


# Reads a text file and returns its content as a string
def read_txt(uploaded_file):
    """
    Reads a text file and returns its content as a string.
    
    Parameters
    ----------
    uploaded_file : file object
        The file object to read from.
    
    Returns
    -------
    str
        The content of the file as a string.
    """
    return uploaded_file.read().decode('utf-8')



# Reads a word file and returns its content as a string
def read_docx(uploaded_file):
    """
    Reads a Word (.docx) file and returns its content as a string.
    
    Parameters
    ----------
    uploaded_file : file object
        The file object to read from.
    
    Returns
    -------
    str
        The content of the file as a string.
    """
    # Open the .docx file
    doc = Document(uploaded_file)
    
    # Use list comprehension to extract text from all paragraphs
    full_text = [paragraph.text for paragraph in doc.paragraphs]
    
    # Join the list into a single string with newline characters
    return '\n'.join(full_text)



# Reads a pdf file and returns its content as a string
def read_pdf(uploaded_file):
    """
    Reads a PDF file and returns its content as a string.
    
    Parameters
    ----------
    uploaded_file : file object
        The file object to read from.
    
    Returns
    -------
    str
        The content of the file as a string.
    """
    # Open the PDF file
    reader = PyPDF2.PdfReader(uploaded_file)
    
    # Use list comprehension to extract text from all pages
    full_text = [reader.pages[page_num].extract_text() for page_num in range(len(reader.pages))]
    
    # Join the list into a single string with newline characters
    return '\n'.join(full_text)



# Download stopwords if not already downloaded
nltk.download('stopwords')

# Function to remove stopwords 
def remove_stopwords(text, additional_stopwords=None):
    """
    Removes stopwords from the given text.
    
    Parameters
    ----------
    text : str
        The text from which to remove stopwords.
    additional_stopwords : list, optional
        Additional stopwords to include in the removal process.
    
    Returns
    -------
    str
        The text with stopwords removed.
    """
    stop_words = set(stopwords.words('english'))
    
    if additional_stopwords:
        # Ensure additional stopwords are split correctly and converted to a set
        additional_stopwords_set = set(word.strip() for word in additional_stopwords)
        stop_words = stop_words.union(additional_stopwords_set)
    
    filtered_words = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(filtered_words)




# Function to handle file upload and call the appropriate function based on file type
def handle_file_upload(uploaded_file, remove_stopwords_option, additional_stopwords):
    """
    Handles the uploaded file and calls the appropriate function based on the file type.
    
    Parameters
    ----------
    uploaded_file : file object
        The file object to read from.
    remove_stopwords_option : bool
        Whether to remove stopwords from the text.
    additional_stopwords : list
        Additional stopwords to include in the removal process.
    
    Returns
    -------
    str
        The content of the file as a string.
    """
    # Get the file name and extension
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()

    if file_extension == 'txt':
        text = read_txt(uploaded_file)  # Corrected: Ensure text is returned from read_txt
    elif file_extension == 'docx':
        text = read_docx(uploaded_file)  # Corrected: Ensure text is returned from read_docx
    elif file_extension == 'pdf':
        text = read_pdf(uploaded_file)  # Corrected: Ensure text is returned from read_pdf
    else:
        return "Unsupported file type"
    
    # Clean text
    text = clean_text(text)

    if remove_stopwords_option:
        text = remove_stopwords(text, additional_stopwords)  # Corrected: Ensure stopwords are removed from text

    return text  # Corrected: Ensure text is always returned from the function



# Function to generate the word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud



# Function to convert image to byte stream and create download link
def get_image_download_link(img, filename, text, format):
    buffered = BytesIO()
    img.save(buffered, format=format.upper())
    buffered.seek(0)
    img_str = base64.b64encode(buffered.read()).decode()
    href = f'<a href="data:file/{format.lower()};base64,{img_str}" download="{filename}">Download {text}</a>'
    return href



# Function to get the top N most frequent words 
def get_top_frequent_words(text, top_n): 
    words = text.split() 
    word_counts = Counter(words) 
    most_common_words = word_counts.most_common(top_n) 
    return pd.DataFrame(most_common_words, columns = ['Word', 'Frequency'])



def clean_text(text):
    """
    Cleans the given text by expanding contractions, converting text to lowercase,
    removing punctuation, digits, and extra whitespace.
    
    Parameters
    ----------
    text : str
        The text to be cleaned.
    
    Returns
    -------
    str
        The cleaned text.
    """
    # Expand contractions
    text = contractions.fix(text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text



# **Function to create download link for dataframe as CSV**
def get_table_download_link(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download csv file</a>'
    return href



def chat_with_document(text, api_key, question, model, temperature, max_tokens):
    client = anthropic.Client(api_key=api_key)
    response = client.completions.create(
        model=model,
        prompt=f"Document: {text}\n\nQuestion: {question}\n\nAnswer:",
        max_tokens_to_sample=max_tokens,
        temperature = temperature)
    answer = response.completion.strip()
    return answer


# --------------------------------------------------------------------------
# ----------------------------------------------------------------------------


# Streamlit app
st.title('Machine Learning - NLP in Action')

st.write("")
st.write("")


# Sidebar for stopwords removal option 
remove_stopwords_option = st.sidebar.checkbox("Remove Stopwords") 
additional_stopwords_input = st.sidebar.text_area("Additional Stopwords " +  repr('\n') + " separated") 
apply_stopwords_button = st.sidebar.button("Apply additional Stopwords")

# Initialize additional stopwords as an empty list 
additional_stopwords = []

if remove_stopwords_option or (remove_stopwords_option and additional_stopwords_input and apply_stopwords_button):
    additional_stopwords = [word.strip() for word in additional_stopwords_input.split('\n')] 
else: 
    additional_stopwords = []

# Sidebar for selecting top N most frequent words 
top_n = st.sidebar.slider("Select top N most frequent words", min_value = 1, max_value = 50, value = 10) 

# File uploader
uploaded_file = st.file_uploader("Please upload a file", type = ["txt", "docx", "pdf"])

st.write("")
st.write("")

if uploaded_file is not None:
    # Handle file upload and display content
    text = handle_file_upload(uploaded_file, remove_stopwords_option, additional_stopwords)
    st.text_area("Uploaded File Content", text, height = 400)


    st.write("")
    st.write("")


    # Show top N most frequent words as a dataframe
    top_n_words_df = get_top_frequent_words(text, top_n)
    st.write("Top ", top_n, " Most Frequent Words:")
    st.dataframe(top_n_words_df)


    # **Create and show download link for the top N words dataframe**
    download_link_csv = get_table_download_link(top_n_words_df, "top_n_frequent_words.csv")
    st.markdown(download_link_csv, unsafe_allow_html = True)


    st.write("")
    st.write("")


    # Multiselect widget for additional stopwords 
    multiselect_stopwords = st.multiselect("Select additional stopwords from the top N frequent words", top_n_words_df["Word"].tolist())

    
    # Update additional stopwords and clean text if any words are selected
    if multiselect_stopwords:
        additional_stopwords.extend(multiselect_stopwords)
        text = remove_stopwords(text, additional_stopwords)

        st.write("")
        st.write("")

        st.text_area("Updated File Content after removing additional stopwords from the top N frequent words", text, height = 400)
        top_n_words_df = get_top_frequent_words(text, top_n)

        st.write("")
        st.write("")

        st.write("Updated Top ", top_n, " Most Frequent Words:")
        st.dataframe(top_n_words_df)


        # **Create and show download link for the updated top N words dataframe**
        download_link_csv = get_table_download_link(top_n_words_df, "updated_top_n_frequent_words.csv")
        st.markdown(download_link_csv, unsafe_allow_html = True)


    st.write("")
    st.write("")


    # Generate word cloud
    if st.button("Generate WordCloud"):
        wordcloud = generate_wordcloud(text)
        st.image(wordcloud.to_array(), use_column_width = True)
        
        st.write("")

        # Format selection
        format = st.selectbox("Select the format to save the WordCloud", ("jpeg", "png", "pdf"))

        # Download link
        download_link = get_image_download_link(wordcloud.to_image(), f"WordCloud.{format}", "your WordCloud", format)
        st.markdown(download_link, unsafe_allow_html = True)


    # Add a text input for the Anthropic API key
    api_key = st.sidebar.text_input("Paste your Anthropic API key here", type = "password")

    # Add a dropdown to select the model
    model = st.sidebar.selectbox("Select Model", ["Claude 3.5 Sonnet 2024-10-22", "Claude 3.5 Haiku", "Claude 3 Opus"])

    # Add a slider to control the temperature
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5)

    # Add a slider to control max tokens
    max_tokens = st.sidebar.slider("Max Tokens to Sample", min_value=50, max_value=500, value=150)

    if api_key:
        st.write("")
        st.write("")

        prompt = st.text_area("Ask a question about the uploaded document", "")

        if st.button("Chat with the document"):
            response = chat_with_document(text, api_key, prompt, model, temperature, max_tokens)
            st.write("Response from Claude:")
            st.write(response)
