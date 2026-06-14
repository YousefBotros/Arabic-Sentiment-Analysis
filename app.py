"""
Arabic Sentiment Analysis - Gradio Web Application
"""

import gradio as gr
from predict import ArabicSentimentAnalyzer

# Initialize analyzer
analyzer = ArabicSentimentAnalyzer()

# Custom CSS
custom_css = """
.gradio-container {
    max-width: 800px;
    margin: auto;
}
h1 {
    text-align: center;
    color: #1d4ed8;
}
"""

# Sample examples
examples = [
    ["المنتج رائع جدا أنصح به الجميع"],
    ["خدمة سيئة للغاية لن اشتري مرة أخرى"],
    ["المنتج جيد لكن فيه بعض العيوب البسيطة"],
    ["شكرا على السرعة في التوصيل"],
    ["مش عاجبني خالص"],
    ["ana 3agabni el montag awy (I really liked the product)"],
    ["5alas mesh hat3raf tgeb tany (You won't be able to bring another)"],
]

def predict_sentiment(text):
    """Predict sentiment for input text"""
    if not text or not text.strip():
        return "⚠️ Please enter some text to analyze", ""
    
    result = analyzer.predict(text)
    
    # Emoji based on sentiment
    emoji_map = {
        'Negative 😡': '🔴',
        'Neutral 😐': '🟡',
        'Positive 😊': '🟢'
    }
    
    sentiment_display = f"""
    ### {emoji_map[result['sentiment']]} {result['sentiment']}
    
    **Confidence:** {result['confidence']:.1%}
    """
    
    return sentiment_display, text

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Arabic Sentiment Analysis") as demo:
    gr.Markdown("""
    # 🌟 تحليل المشاعر باللغة العربية
    ## Arabic Sentiment Analysis
    
    This model analyzes Arabic text and detects whether the sentiment is **Positive**, **Neutral**, or **Negative**.
    
    ### Features:
    - Supports **Arabic script** and **Franco-Arabic** (Arabic written with Latin letters + numbers)
    - Handles emojis, URLs, and mentions
    - Uses hybrid Word2Vec + TF-IDF embeddings
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="📝 Enter Arabic Text",
                placeholder="اكتب النص العربي هنا...\nExample: المنتج رائع جدا",
                lines=5
            )
            analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
        
        with gr.Column():
            output_sentiment = gr.Markdown(label="Predicted Sentiment")
            output_text = gr.Textbox(label="Original Text", visible=False)
    
    gr.Markdown("### 💡 Try these examples:")
    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=[output_sentiment, output_text],
        fn=predict_sentiment,
        cache_examples=True
    )
    
    gr.Markdown("""
    ---
    ### 📊 Model Information
    - **Architecture:** Neural Network (128 → 64 → 32 → 3)
    - **Embeddings:** Word2Vec (300d) + TF-IDF weighted
    - **Preprocessing:** Normalization, stemming, stopword removal
    - **Classes:** Negative 😡, Neutral 😐, Positive 😊
    """)
    
    analyze_btn.click(
        fn=predict_sentiment,
        inputs=input_text,
        outputs=[output_sentiment, output_text]
    )

if __name__ == "__main__":
    demo.launch(share=True)
