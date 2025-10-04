from transformers import pipeline
import gradio as gr


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    if len(text.strip()) == 0:
        return "Please enter text"
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']


iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, placeholder="Paste or type your paragraph here..."),
    outputs="text",
    title="Summarize Text using Gen AI",
    description="Enter a paragraph or article and get a short summary powered by Hugging Face BART model."
)

iface.launch()
