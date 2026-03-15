
# app.py — Emotion Detector using Gradio
# Model: Linear SVM (BoW) — Best Overall Model (Test Acc: 0.8966)
#
# HOW TO RUN:
#   1. Make sure these 3 files are in the SAME folder as this app.py:
#         linear_svm_bow_model.pkl
#         bow_vectorizer.pkl
#         label_encoder.pkl
#   2. Open terminal in VS Code
#   3. Run: python gradioapp.py
#   4. Open browser and go to: http://127.0.0.1:7860


# Imports 
import gradio as gr
import joblib
import numpy as np
import string
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords



model      = joblib.load('final_model.pkl')
vectorizer = joblib.load('bow_vectorizer.pkl')
le         = joblib.load('label_encoder.pkl')


# STEP 2: TEXT CLEANING FUNCTION
# Must be exactly the same as what we used during training


stop_words_set = set(stopwords.words('english'))

def full_clean(txt):
    txt = txt.lower()                                                # lowercase
    txt = txt.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    txt = re.sub(r'\d+', '', txt)                                   # remove numbers
    txt = txt.encode('ascii', 'ignore').decode('ascii')             # remove emojis
    txt = re.sub(r'\s+', ' ', txt).strip()                         # remove extra spaces
    txt = ' '.join([w for w in txt.split() if w not in stop_words_set])  # remove stopwords
    return txt



# Each emotion has an emoji, color and description

EMOTION_META = {
    'joy':      {'emoji': '😄', 'color': '#F9C74F', 'desc': 'Feeling happy, elated or content'},
    'sadness':  {'emoji': '😢', 'color': '#577590', 'desc': 'Feeling down, sorrowful or blue'},
    'anger':    {'emoji': '😠', 'color': '#F94144', 'desc': 'Feeling frustrated, irritated or furious'},
    'fear':     {'emoji': '😨', 'color': '#9B5DE5', 'desc': 'Feeling scared, anxious or worried'},
    'love':     {'emoji': '❤️',  'color': '#F3722C', 'desc': 'Feeling affectionate or romantic'},
    'surprise': {'emoji': '😲', 'color': '#90BE6D', 'desc': 'Feeling shocked or astonished'},
}



# This function runs every time user clicks the button


def predict_emotion(text):

    # Check if user entered anything
    if not text or not text.strip():
        return "<p style='color:gray;font-family:sans-serif'>Please enter some text.</p>"

    # Clean the text same way as training
    cleaned = full_clean(text)

    # Convert text to numbers using BoW vectorizer
    vectorized = vectorizer.transform([cleaned])

    # Predict the emotion (returns a number)
    pred_number = model.predict(vectorized)[0]

    # Convert number back to emotion name
    pred_emotion = le.classes_[pred_number]

    # Get confidence scores using decision function
    # LinearSVC does not have predict_proba so we use decision_function
    raw_scores = model.decision_function(vectorized)[0]

    # Convert raw scores to percentages using softmax
    e_x   = np.exp(raw_scores - raw_scores.max())
    probs = e_x / e_x.sum()

    # Get confidence of predicted emotion
    confidence = probs[pred_number] * 100

    # Get emotion details for display
    meta = EMOTION_META.get(pred_emotion, {'emoji': '❓', 'color': '#888', 'desc': ''})

    # Sort emotions by probability (highest first)
    sorted_emotions = sorted(
        zip(le.classes_, probs),
        key=lambda x: x[1],
        reverse=True
    )

    # Build probability bars for all emotions
    prob_bars = ''
    for emo, prob in sorted_emotions:
        emo_meta = EMOTION_META.get(emo, {})
        prob_bars += f"""
        <div style='margin-bottom:10px'>
            <div style='display:flex; justify-content:space-between; font-size:13px; margin-bottom:3px'>
                <span>{emo_meta.get('emoji','')} {emo}</span>
                <span><b>{prob*100:.1f}%</b></span>
            </div>
            <div style='background:#e0e0e0; border-radius:6px; height:12px'>
                <div style='background:{emo_meta.get("color","#888")};
                            height:12px; border-radius:6px;
                            width:{prob*100:.1f}%'></div>
            </div>
        </div>
        """

    # Build the final HTML result card
    html = f"""
    <div style='font-family:sans-serif; max-width:550px; margin:0 auto'>

        <!-- Top result card -->
        <div style='background:{meta["color"]}20;
                    border: 2px solid {meta["color"]};
                    border-radius:16px;
                    padding:28px;
                    text-align:center;
                    margin-bottom:20px'>

            <div style='font-size:60px'>{meta["emoji"]}</div>

            <div style='font-size:28px;
                        font-weight:bold;
                        color:{meta["color"]};
                        margin:10px 0 6px'>
                {pred_emotion.upper()}
            </div>

            <div style='color:#555; font-size:14px; margin-bottom:12px'>
                {meta["desc"]}
            </div>

            <div style='font-size:13px; color:#888'>
                Confidence: <b style='color:{meta["color"]}; font-size:16px'>{confidence:.1f}%</b>
                &nbsp;·&nbsp;
                Model: <b>Linear SVM (BoW)</b>
            </div>
        </div>

        <!-- Probability breakdown -->
        <div style='background:#f7f7f7; border-radius:12px; padding:18px'>
            <div style='font-weight:bold; font-size:14px; margin-bottom:14px; color:#333'>
                Probability for each emotion:
            </div>
            {prob_bars}
        </div>

        <!-- Show cleaned text -->
        <div style='margin-top:10px; font-size:11px; color:#aaa; text-align:center'>
            Cleaned text: "{cleaned[:80]}{'...' if len(cleaned) > 80 else ''}"
        </div>

    </div>
    """

    return html



# BUILD THE GRADIO APP


# Example sentences the user can click to try
examples = [
    ["I feel so happy and grateful today!"],
    ["I am so sad and nothing feels right anymore."],
    ["I am furious about what happened!"],
    ["I am terrified about the exam tomorrow."],
    ["I love spending time with you so much."],
    ["I never expected that to happen at all!"],
]

# Build the interface
with gr.Blocks(title='Emotion Detector') as app:

    # Title and description
    gr.Markdown("""
    # 🎭 Emotion Detector
    **Model:** Linear SVM with Bag of Words &nbsp;|&nbsp;
    **Test Accuracy:** 89.66% &nbsp;|&nbsp;
    **Emotions:** Joy · Sadness · Anger · Fear · Love · Surprise

    Type any sentence and the model will detect the emotion behind it.
    """)

    # Two columns layout
    with gr.Row():

        # Left side — input
        with gr.Column():
            input_box  = gr.Textbox(
                label='Enter your sentence here',
                placeholder='Example: I feel so happy today!',
                lines=4
            )
            detect_btn = gr.Button('Detect Emotion', variant='primary')
            clear_btn  = gr.Button('Clear')

            # Example sentences
            gr.Examples(
                examples=examples,
                inputs=input_box,
                label='Click any example to try'
            )

        # Right side — output
        with gr.Column():
            output_box = gr.HTML(label='Result')

    # Connect buttons to the prediction function
    detect_btn.click(fn=predict_emotion, inputs=input_box, outputs=output_box)
    input_box.submit(fn=predict_emotion, inputs=input_box, outputs=output_box)
    clear_btn.click(fn=lambda: ('', ''), outputs=[input_box, output_box])



# LAUNCH THE APP

# share=False  → only runs on your computer at http://127.0.0.1:7860
# share=True   → creates a public link (use for sharing with others)
app.launch(share = True )
