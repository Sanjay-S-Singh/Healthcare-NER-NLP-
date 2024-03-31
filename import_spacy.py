import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm
import json
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



# Downloading the large English model
#!python -m spacy download en_core_web_lg

# Load the large English model
#nlp = spacy.load("en_core_web_lg")

# Title and Header
st.title('Medicine Prediction for Covid using NER')
st.write('📌Presented by Sanjay SANJAY, Han ZHU and Hong Ngoc VU ')

# Function definitions for each tab
def data_introduction_tab():
    st.header("Data Introduction")
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

    # App Summary with styled background
    st.markdown('<div class="data-container"><h2>Summary</h2><p>'
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
        st.markdown('<div class="data-container"><h2>Purpose</h2><p>'
                    '<li>Quickly identify relevant entities within large datasets.</li>'
                    '<li>Understand the context in which medicines and medical conditions are mentioned.</li>'
                    '<li>Accelerate the process of medical research by providing an automated way to sift through vast amounts '
                    'of textual information.</li>'
                    '</ul></p></div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="data-container"><h2>Target Audience</h2><p>'
                    '<li><b>Medical Companies</b>: Pharmaceutical and biotechnology companies engaged in drug research and development.</li>'
                    '<li><b>Researchers and Academics</b>: Individuals in the academic and research sector focusing on medical research related to COVID-19.</li>'
                    '<li><b>Healthcare Professionals</b>: Medical professionals seeking information on treatment strategies and drug efficacy.</li>'
                    '</ul>'
                    '</p></div>', unsafe_allow_html=True)


def overview_tab():
    st.header("Overview")

    # Custom CSS to make the text smaller and align the containers
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
nlp_new_model = spacy.load("/Users/sanjay/Downloads/SpaCy_Python/spaCy_Model/output/model-best")

def main():
    # Prompt the user to enter text
    text = input("Enter some text: ")

    # Print the entered text
    #print("You entered:", text)
    
    # Process the text using SpaCy
    doc = nlp_new_model(text)

    # Colors for entity visualization
    colors = {"PATHOGEN": "#DE3163", "MEDICINE": "#6495ED", "MEDICALCONDITION": "#FF7F50"}
    options = {"colors": colors}

    # Visualizing entities in the text
    spacy.displacy.render(doc, style="ent", options=options, jupyter=True)

if __name__ == "__main__":
    main()
