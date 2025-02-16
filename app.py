import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import gradio as gr
import base64  # Fix for base64 import

# Initialize the Groq client
GROQ_API_KEY = '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'
client = Groq(api_key=GROQ_API_KEY)

# Load the trained Random Forest model
model_path = "random_forest_quality_classifier.pkl"
clf = joblib.load(model_path)

# Feature extraction functions
def calculate_laplacian_variance(image):
    """Measure sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness(image):
    """Measure brightness by calculating mean pixel intensity."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def calculate_noise(image):
    """Measure noise level using standard deviation of pixel intensities."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def extract_features(image_path):
    """Extract features for a single image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")
    return [
        calculate_laplacian_variance(image),
        calculate_brightness(image),
        calculate_noise(image),
    ]

# Function to classify the quality of an image
def classify_image_quality(image_path):
    try:
        features = extract_features(image_path)
        prediction = clf.predict([features])[0]
        return "good" if prediction == 1 else "poor"
    except Exception as e:
        raise ValueError(f"Error during classification: {e}")

# Function to encode image as base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to describe the image using the Groq model
def describe_image(image_path, prompt):
    if image_path:
        base64_image = encode_image(image_path)
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
    else:
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
    return response.choices[0].message.content

# Function to generate a PDF report
def generate_pdf(report, explanation):
    pdf_output = "./final_report.pdf"
    doc = SimpleDocTemplate(pdf_output, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(name="Title", fontSize=18, leading=22, alignment=1, spaceAfter=20)
    heading_style = ParagraphStyle(name="Heading", fontSize=14, leading=18, textColor=colors.darkblue, spaceAfter=10)
    body_style = ParagraphStyle(name="Body", fontSize=12, leading=14, spaceAfter=12)

    elements = [
        Paragraph("Final Radiology Report", title_style),
        Spacer(1, 12),
        Paragraph("Provisional Report:", heading_style),
        Paragraph(report, body_style),
        Spacer(1, 20),
        PageBreak(),
        Paragraph("Explanation", title_style),
        Spacer(1, 12),
        Paragraph(explanation, body_style)
    ]

    doc.build(elements)
    return pdf_output

# Function to process the uploaded image
def process_image(image_path, report_format):
    try:
        quality = classify_image_quality(image_path)
        if quality == "poor":
            raise gr.Error("Poor quality scan detected. Please upload a better-quality image.")

        prompt = f"You are a virtual radiology technician. Analyze the provided medical imaging and generate a detailed provisional report. Identify the scan type (e.g., X-ray, CT, MRI) and describe any abnormalities, patterns, or relevant observations. If patient info is missing, indicate that explicitly. Do not make a diagnosis; only assist the radiologist. {report_format}"
        description = describe_image(image_path, prompt)
        return description

    except gr.Error as e:
        raise e
    except Exception as e:
        raise gr.Error(f"Error: {e}")

# Function to generate the final report and explanation
def process_final_report(provisional_report, language):
    try:
        if language == "Hindi":
            explanation_prompt = f"Give explanation in Hindi. Based on this report: {provisional_report}, explain the findings to a layperson with empathy but your language should be professional and easy to understand also dont write in first person it should be written aa a normal report is written. Provide reassurance and advice for further action, highlighting the criticality of the condition."
        else:
            explanation_prompt = f"Based on this report: {provisional_report}, explain the findings to a layperson with empathy but your language should be professional and easy to understand also dont write in first person it should be written aa a normal report is written. Provide reassurance and advice for further action, highlighting the criticality of the condition."

        explanation = describe_image(None, explanation_prompt)
        pdf_path = generate_pdf(provisional_report, explanation)
        return explanation, pdf_path

    except Exception as e:
        return f"Error: {e}", None

# Define Gradio application
def create_app():
    with gr.Blocks() as app:
        gr.Markdown("## Radiology Provisional Report Generator")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="filepath", label="Upload X-Ray or Scan Image")
                report_format = gr.Textbox(placeholder="Define the report format here...", label="Provisional Report Format")
                submit_button = gr.Button("Generate Provisional Report")

            with gr.Column():
                provisional_report = gr.Textbox(label="Provisional Report", placeholder="The report will appear here...", lines=10, interactive=True)
                language_toggle = gr.Radio(label="Select Explanation Language", choices=["English", "Hindi"], value="English")
                explanation_output = gr.Textbox(label="Explanation for Layman", placeholder="The explanation will appear here...", lines=5)
                download_trigger = gr.Button("Download PDF")
                download_file = gr.File(label="Download Final PDF")

        submit_button.click(
            process_image,
            inputs=[image_input, report_format],
            outputs=[provisional_report],
        )

        def prepare_pdf(provisional_report, language):
            explanation, pdf_path = process_final_report(provisional_report, language)
            return explanation, pdf_path

        download_trigger.click(
            prepare_pdf,
            inputs=[provisional_report, language_toggle],
            outputs=[explanation_output, download_file],
        )

    return app

if __name__ == "__main__":
    app = create_app()
    app.launch()
