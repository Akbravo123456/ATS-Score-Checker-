import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def calculate_similarity(job_description, resume_text):
    job_description = preprocess_text(job_description)
    resume_text = preprocess_text(resume_text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
    similarity_score = (tfidf_matrix * tfidf_matrix.T).A[0, 1]

    return similarity_score

st.title('ATS Score Checker')
st.subheader('Enter Job Description:')
job_description = st.text_area('Job Description')
st.subheader('Upload Resume:')
resume_upload = st.file_uploader('Upload Resume', type=['docx', 'pdf', 'txt'])

if job_description and resume_upload is not None:
    resume_text = resume_upload.read().decode("latin-1")
    ats_score = calculate_similarity(job_description, resume_text)
    st.subheader('ATS Score:')
    st.write(f'The ATS score of the resume is: {ats_score:.2%}')
