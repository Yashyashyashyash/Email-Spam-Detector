import streamlit as st


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
