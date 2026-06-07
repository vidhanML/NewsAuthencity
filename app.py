import re
import math
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="News Checker AI", 
    page_icon="📰", 
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("🧭 Navigation")
    page_choice = st.selectbox(
        "Choose a page:",
        ["Fake News Detector", "Importance of Real News"]
    )
    st.markdown("---")
    st.subheader("⚙️ System Details")
    st.write("🤖 **Model:** Passive Aggressive Classifier")
    st.write("📚 **Dataset Size:** 45,757 Articles")


if page_choice == "Fake News Detector":
    st.title("📰 Fake News Detector")
    st.write("Paste a news article below to check if it is real or fake.")

    @st.cache_resource
    def load_models():
        clf = joblib.load('fake_news_classifier_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return clf, vectorizer

    try:
        clf, vectorizer = load_models()
    except Exception:
        st.error("Could not find the model files. Make sure your .pkl files are in this folder.")
        st.stop()

    def clean_leakage_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'\b(reuters|ap|associated press|afp)\b', '', text)
        text = re.sub(r'\b(washington|london|tokyo)\s+—?\s*', '', text) 
        text = re.sub(r'\b(featured image|image caption|image copyright|story highlights|video)\b', '', text)
        text = re.sub(r'\b(www|com|http|https)\b', '', text)
        text = re.sub(r'\b(getty images|getty|flickr|photo by|image via)\b', '', text)
        text = re.sub(r'\b(watch|read|breaking|just|click here)\b', '', text)
        text = re.sub(r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', '', text)
        text = re.sub(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    with st.form("classification_form"):
        user_input = st.text_area(
            label="Article Text", 
            height=250, 
            placeholder="Paste your news story here..."
        )
        
        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            submit_button = st.form_submit_button(label="CHECK ARTICLE", use_container_width=True)

    if submit_button:
        if user_input.strip() == "":
            st.warning("Please paste some text first!")
        else:
            cleaned_text = clean_leakage_text(user_input)
            vector_input = vectorizer.transform([cleaned_text])
            
            decision_score = float(clf.decision_function(vector_input)[0])
            probability = 1 / (1 + math.exp(-decision_score))
            
            prediction = clf.predict(vector_input)
            is_real = (prediction == 1 or str(prediction).lower() == 'real')
            
            confidence_score = probability if is_real else (1 - probability)
            confidence_percentage = int(confidence_score * 100)
            
            st.subheader("📊 Results")
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                with st.container(border=True):
                    if is_real:
                        st.metric(label="VERDICT", value="✅ REAL NEWS")
                        st.write("This text looks like a normal, factual news report.")
                    else:
                        st.metric(label="VERDICT", value="🚨 FAKE NEWS")
                        st.write("The writing style looks like a fake or misleading story.")
                
            with col2:
                with st.container(border=True):
                    st.metric(label="AI CONFIDENCE", value=f"{confidence_percentage}%")
                    st.progress(confidence_score)
                    st.write("How sure the AI is about this result.")

            st.markdown("---")
            st.subheader("🔍 Key Phrases Driving this Prediction")
            
            feature_names = vectorizer.get_feature_names_out()
            weights = clf.coef_[0]
            user_word_indices = vector_input.nonzero()[1]

            if len(user_word_indices) > 0:
                word_analysis = []
                for idx in user_word_indices:
                    word = feature_names[idx]
                    weight = weights[idx]
                    impact = vector_input[0, idx] * weight
                    word_analysis.append({"Word": word, "Impact": impact})
                    
                df_words = pd.DataFrame(word_analysis)
                df_words = df_words.sort_values(by="Impact", key=abs, ascending=False).head(5)
                
                def get_influence_label(val):
                    if val > 0.01: return "🟢 Strongly Signals Real News"
                    elif val > 0: return "🍏 Weakly Signals Real News"
                    elif val < -0.01: return "🔴 Strongly Signals Fake News"
                    else: return "🍎 Weakly Signals Fake News"
                    
                df_words["Influence"] = df_words["Impact"].apply(get_influence_label)
                df_display = df_words[["Word", "Influence"]].reset_index(drop=True)
                st.table(df_display)
            else:
                st.info("The text provided contains too many common filler words for specific keyword analysis.")

else:
    st.title("🌱 Why Real News Matters")
    st.write("In a world full of information, keeping news honest and factual keeps our society running safely.")
    st.write("---")
    
    col_left, col_right = st.columns(2, gap="large")
    
    with col_left:
        with st.container(border=True):
            st.subheader("🤝 1. It Builds Social Trust")
            st.write("When people can trust what they read, they feel safer talking with their neighbors. Honest news brings communities closer together.")
        
        st.write("")
        
        with st.container(border=True):
            st.subheader("🏥 2. It Keeps Communities Safe")
            st.write("During emergencies or weather warnings, accurate information tells people exactly what to do to stay out of danger.")

    with col_right:
        with st.container(border=True):
            st.subheader("🗳️ 3. It Helps Us Make Smart Choices")
            st.write("Whether deciding on local rules or choosing leaders, families need real facts to make the best possible daily choices.")
        
        st.write("")
        
        with st.container(border=True):
            st.subheader("🛡️ 4. It Protects True History")
            st.write("Real journalism acts as a diary for our world. By writing down what actually happened, it keeps historical records accurate.")

st.write("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ♡ by Vidhan</p>", unsafe_allow_html=True)
