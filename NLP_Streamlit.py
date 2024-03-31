import spacy
from spacy import displacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm
import json
import subprocess
import matplotlib.pyplot as plt
from collections import Counter
import streamlit as st
import os
import pandas as pd

# Load the trained model
nlp_new_model = spacy.load("/Users/sanjay/Downloads/SpaCy_Python/spaCy_Model/output/model-best")

# Downloading the medium English model
command = "python -m spacy download en_core_web_sm"
process = subprocess.run(command.split(), check=True)

# Loading the dataset
path = os.path.join(os.path.dirname(__file__), 'Corona2.json')
with open(path, 'r') as f:
    data = json.load(f)

# Title and Header
st.title('Medicine Prediction for Covid using NER')
st.write('📌Presented by Sanjay SANJAY, Han ZHU and Hong Ngoc VU ')

# Function definitions for each tab
def data_introduction_tab():
    st.header("Model Introduction")
    # Using containers to create visual blocks
    # App Summary in a full-width rectangular block
    # Define custom CSS to style the containers
    st.markdown("""
        <style>
        .data-container {
            border: 2px solid #4CAF50;
            border-radius: 5px; /* Rounded corners */
            padding: 20px; /* Some padding */
            margin: 10px 0px; /* Some space around */
            background-color: #f0f2f6; /* Light grey background */
        }
        </style>
    """, unsafe_allow_html=True)

    # App Reason: Why and How we did it?
    st.markdown('<div class="data-container"><h3>Why and How we did it?</h3><p>'
                '<ul>'
                '<li><b>WHY: </b>Why: We are creating an app for healthcare professionals and companies to analyze data on <b>"Medical Conditions," "Medicine," and "Pathogens."</b> Users input text, and our model highlights relevant findings in these areas.</li>'
                '<li><b>HOW: </b>We trained our NLP model using COVID-19 data, chosen for its rich variety of health issues encountered during the pandemic. This enables our model to accurately identify and interpret mentions of medical conditions, medicines, and pathogens in text.</li>'
                '</ul></p></div>', unsafe_allow_html=True)

    # App Summary with styled background
    st.markdown('<div class="data-container"><h3>Summary</h3><p>'
                'This app leverages advanced Natural Language Processing (NLP) techniques to analyze textual data '
                'related to the COVID-19 pandemic. It focuses on extracting and interpreting three key entities:'
                '<ul>'
                '<li><b>Medical Conditions</b>: Identifying mentions of various medical conditions associated with COVID-19.</li>'
                '<li><b>Medicines</b>: Extracting names of medicines that are being used or researched for treating COVID-19.</li>'
                '<li><b>Pathogens</b>: Highlighting mentions of pathogens, especially SARS-CoV-2, responsible for the condition.</li>'
                '</ul></p></div>', unsafe_allow_html=True)

    # Columns for Purpose of the App and Target Audience with styled background
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="data-container"><h3>Purpose</h3><p>'
                    '<li>Quickly identify relevant entities within large datasets.</li>'
                    '<li>Understand the context in which medicines and medical conditions are mentioned.</li>'
                    '<li>Accelerate the process of medical research by providing an automated way to sift through vast amounts '
                    'of textual information.</li>'
                    '</ul></p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="data-container"><h3>Target Audience</h3><p>'
                    '<li><b>Medical Companies</b>: Pharmaceutical and biotechnology companies engaged in drug research and development.</li>'
                    '<li><b>Researchers and Academics</b>: Individuals in the academic and research sector focusing on medical research related to COVID-19.</li>'
                    '<li><b>Healthcare Professionals</b>: Medical professionals seeking information on treatment strategies and drug efficacy.</li>'
                    '</ul>'
                    '</p></div>', unsafe_allow_html=True)


def overview_tab():
    st.header("Overview")

    # Custom CSS to make the text smaller and align the containers
    
    st.markdown('<div class="data-container"><h5>An overview sample data used for training of model:</h5><p>'
                    '</ul>'
                    '</p></div>', unsafe_allow_html=True)

    st.markdown("""
        <style>
        .kpi-container {
            border: 2px solid #4CAF50;
            border-radius: 5px;
            padding: 20px;
            margin: 10px;
            background-color: #f0f2f6;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 200px; /* Adjust height as needed */
        }
        .kpi-container h2 {
            font-size: 1.75rem; /* Adjust title size as needed */
            margin: 0;
        }
        .kpi-container p {
            font-size: 2.5rem; /* Adjust number size as needed */
            margin: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # KPIs layout using columns
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
            <div class="kpi-container">
                <h2>Total Medicines</h2>
                <p>86</p>
            </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        st.markdown(f"""
            <div class="kpi-container">
                <h2>Total Medical Conditions</h2>
                <p>102</p>
            </div>
        """, unsafe_allow_html=True)
    
    with kpi3:
        st.markdown(f"""
            <div class="kpi-container">
                <h2>Total Pathogens Suggested</h2>
                <p>60</p>
            </div>
        """, unsafe_allow_html=True)

# Loading the trained model
def load_model():
    model_path = r"C:\Users\vuhon\Downloads\Data science with python - dataset\spaCy_Model\output\model-best"
    return spacy.load(model_path)

def analyze_and_visualize(text):
    doc = nlp_new_model(text)
    colors = {"PATHOGEN": "#DE3163", "MEDICINE": "#6495ED", "MEDICALCONDITION": "#FF7F50"}
    options = {"ents": ["PATHOGEN", "MEDICINE", "MEDICALCONDITION"], "colors": colors}
    html = displacy.render(doc, style="ent", options=options)
    st.write(html, unsafe_allow_html=True)
    entity_summary(doc)


def entity_summary(doc):
    medicines = [ent.text for ent in doc.ents if ent.label_ == "MEDICINE"]
    conditions = [ent.text for ent in doc.ents if ent.label_ == "MEDICALCONDITION"]
    pathogens = [ent.text for ent in doc.ents if ent.label_ == "PATHOGEN"]

    st.subheader("Summary")
    st.write(f"Medicines Mentioned: {len(medicines)}")
    st.write(f"Medical Conditions Mentioned: {len(conditions)}")
    st.write(f"Pathogens Mentioned: {len(pathogens)}")

# Main app function
def main():
    # Tabs for model introduction, NLP processing, and conclusion
    tabs = ["Model Introduction", "Overview", "NLP Processing"]
    selected_tab = st.sidebar.radio("📢 Select Tab", tabs)

    if selected_tab == "Model Introduction":
        data_introduction_tab()
    elif selected_tab == "Overview":
        overview_tab()
    elif selected_tab == "NLP Processing":
        st.header("NLP Processing")
        user_input = st.text_area("Enter text for NLP processing", " ", key="nlp_input")
        if st.button('Analyze Text'):
            analyze_and_visualize(user_input)

# Run the app
if __name__ == "__main__":
    main()
