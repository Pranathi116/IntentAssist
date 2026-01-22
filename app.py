import streamlit as st
import joblib
import re

# Load trained model and vectorizer
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="Intent Assist", layout="centered")

# ---------------- NORMALIZATION ----------------
def normalize_text(text):
    text = text.lower().strip()
    # Reduce repeated characters: hellooo -> hello
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        margin-top: 40px;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 40px;
    }
    .risk {
        color: red;
        font-weight: bold;
        font-size: 22px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ----------------
st.markdown('<div class="title">INTENT ASSIST</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Hybrid AI Intent Detection Chatbot</div>', unsafe_allow_html=True)

# ---------------- INPUT ----------------
user_input = st.text_input("Enter your message", placeholder="Type something here...")
analyze = st.button("Analyze")

# ---------------- MAIN LOGIC ----------------
if analyze:
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        text_lower = normalize_text(user_input)

        # Rule word lists
        greeting_words = ["hi", "hello", "hey", "good morning", "good evening", "hola", "hii", "heyy"]

        self_harm_words = [
            "depressed", "depression", "sad", "not happy", "hopeless",
            "tired of living", "want to die", "kill myself", "end my life",
            "no reason to live", "empty", "broken", "worthless"
        ]

        danger_words = ["follow", "chase", "attack", "unsafe", "scared", "threat", "hurt", "behind"]

        # -------- HYBRID DECISION LAYER --------

        # 1. Greeting override
        if any(word in text_lower for word in greeting_words):
            intent = "greeting"
            confidence = 1.0

        # 2. Self-harm override (CRITICAL)
        elif any(word in text_lower for word in self_harm_words):
            intent = "self_harm"
            confidence = 1.0

        # 3. Danger override
        elif any(word in text_lower for word in danger_words):
            intent = "danger"
            confidence = 1.0

        # 4. ML fallback
        else:
            vec = vectorizer.transform([user_input])
            probs = model.predict_proba(vec)[0]
            intent_index = probs.argmax()
            intent = model.classes_[intent_index]
            confidence = probs[intent_index]

        # Low confidence fallback (do NOT downgrade safety intents)
        if confidence < 0.4 and intent not in ["danger", "self_harm"]:
            intent = "general_chat"

        # ---------------- DISPLAY RESULT ----------------
        st.markdown("### 🔍 Detected Intent")

        if intent in ["danger", "harassment", "stalking", "self_harm"]:
            st.markdown(f'<div class="risk">🚨 {intent.upper()} (Confidence: {confidence*100:.2f}%)</div>', unsafe_allow_html=True)
        else:
            st.success(f"{intent.upper()}  (Confidence: {confidence*100:.2f}%)")

        # ---------------- SUGGESTIONS ----------------
        st.markdown("### 💡 Suggestions")

        if intent in ["danger", "harassment", "stalking", "self_harm"]:
            st.error("This appears to be a safety-related concern.")

            st.markdown("#### 📞 National Helpline Numbers (India)")
            st.write("• Emergency: **112**")
            st.write("• Women Helpline: **181**")
            st.write("• Mental Health Helpline: **9152987821**")

            st.markdown("#### 🛡️ General Safety Precautions")
            st.write("• Stay in a safe and public place.")
            st.write("• Share your location with a trusted person.")
            st.write("• Keep emergency contacts ready.")
            st.write("• Do not hesitate to seek professional help.")

        elif intent == "weather":
            st.info("You asked about the weather. Check a weather app or website for live updates.")

        elif intent == "boredom":
            st.info("You seem bored 😄")
            st.write("• Watch a short video or movie")
            st.write("• Go for a walk")
            st.write("• Try a new hobby")
            st.write("• Listen to music or a podcast")

        elif intent == "query":
            st.info("You asked a general question. Try asking something more specific.")

        elif intent == "greeting":
            st.info("Hello! 👋 How can I help you today?")

        else:
            st.write("I'm here to help! Ask me anything.")
