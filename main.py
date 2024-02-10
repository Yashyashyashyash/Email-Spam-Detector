import streamlit as st
from traitlets.utils.importstring import import_item
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to preprocess the text
def preprocess_text(text):
    # Add your preprocessing steps here
    return text

# Load your data or CSV file
@st.cache
def load_data():
    df = pd.read_csv('mail_data.csv')  # Replace with your CSV file
    return df

# Function to train the model
def train_model(df):
    # Preprocess data
    data = df.where((pd.notnull(df)), '')
    data.info()
    data.loc[data['Category'] == 'spam', 'Category'] = 0
    data.loc[data['Category'] == 'ham', 'Category'] = 1

    X = data['Message']
    y = data['Category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

    # Feature extraction
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_feature = feature_extraction.fit_transform(X_train)
    X_test_feature = feature_extraction.transform(X_test)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # Model training
    model = LogisticRegression()
    model.fit(X_train_feature, y_train)

    return model, feature_extraction

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Spam Detector", page_icon=":shield:", layout="wide")
    st.markdown("""
        <style>
            body {
                background-color: black
                color: white;
                
            }
            .reportview-container .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)
    st.title("Email/SMS Spam Detection App")
    st.write("Enter an email or SMS to check if it's spam or ham:")

    user_input = st.text_area("Input Text Here")

    if st.button("Check"):
        model, feature_extraction = train_model(load_data())
        input_data_features = feature_extraction.transform([preprocess_text(user_input)])
        prediction = model.predict(input_data_features)

        if prediction[0] == 1:
            st.write("Prediction: Ham")
        else:
            st.write("Prediction: Spam")

if __name__ == "__main__":
    main()
