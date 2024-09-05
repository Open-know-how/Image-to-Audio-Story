from dotenv import load_dotenv, find_dotenv
import requests
import os
import re
import streamlit as st


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}


# Image to text
def image_to_text(filename):
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"

    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()[0]["generated_text"]


# Text to History
def text_to_history(text):
    # API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small"
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1/v1/chat/completions"

    template = f"""
    You are a story teller.
    You can generate a short story based on the following simple narrative, the story should be no more than 200 words and minimum 100 words.
    
    "{text}"
    
    The text should start and end with $$
    """

    data = {
        "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "messages": [{"role": "user", "content": template}],
        "max_tokens": 500,
        "stream": False,
    }
    response = requests.post(API_URL, headers=headers, json=data)
    content = response.json()["choices"][0]["message"]["content"]

    # Math text between $$ and $$ using regex
    match = re.search(r"\$\$([\s\S]*?)\$\$", content)
    if match:
        content = match.group(1)
        return content

    return content


# History to Audio
def text_to_speech(history):
    API_URL = (
        "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    )

    data = {"inputs": history}
    response = requests.post(API_URL, headers=headers, json=data)

    # Save audio
    with open("audio.flac", "wb") as f:
        f.write(response.content)


def main():
    st.set_page_config(page_title="Image to audio story", page_icon="🖼️")
    st.header("Turn an Image into a audio story")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as f:
            f.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        scenario = image_to_text(uploaded_file.name)
        print("Scenario :", scenario)

        story = text_to_history(scenario)
        print("Story :", story)

        # Generate and save audio
        text_to_speech(story)

        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)

        st.audio("audio.flac")


if __name__ == "__main__":
    main()
