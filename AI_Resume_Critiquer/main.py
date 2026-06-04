import streamlit as st
import PyPDF2
import io
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

st.set_page_config(page_title="AI Resume Critiquer", page_icon="🐙", layout="centered")

st.title("AI Resume Critiquer")
st.markdown("Upload your resume to get AI-powered feedback tailored to your needs <3")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# these variables receive the data and maintain their state when the script gets rerun
uploaded_file = st.file_uploader("Upload your resume (PDF of TXT)", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role you are targetting (optional)")

analyze = st.button("Analyze Resume")   # true when button pressed

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        # st.text("pdf file received")
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))  # converting to bytes
    # st.text("non pdf file received")
    return uploaded_file.read().decode("utf-8")     # if it's not a pdf, assume that it's a text file and decode it as a UTF 8

if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)

        if not file_content.strip():
            st.error("File is empty")
            st.stop
            
        prompt = f"""Analyze this resume and provide constructive feedback.
        Focus on the following aspects:
        1. Content clarity and impact
        2. Skills presentation
        3. Experience descriptions
        4. Specific improvements for {job_role if job_role else 'general job applications'}

        Resume content:
        {file_content}

        Provide analysis in a clear, structured format with specific recommendations.
        """

        client = ChatGroq(
            api_key = GROQ_API_KEY,
            model="llama-3.3-70b-versatile",
            temperature = 0.5,
            max_tokens=1000)
        
        response = client.invoke([
                {"role": "system", "content": "You are an expert resume reviewer with years of experience in HR and recruitment."},
                {"role": "user", "content": prompt}
            ])

        st.markdown('### Analysis Results')
        st.markdown(response.content)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    