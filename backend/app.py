import math
from moviepy.editor import VideoFileClip
import whisper
import cv2
from ultralytics import YOLO
import numpy as np
import re
import pytesseract
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from translate import Translator
from flask_socketio import SocketIO, emit


import os
import logging

from langchain_openai import ChatOpenAI # pass the model to use
from langchain.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS # perform similarity search on vector embeddings
from langchain.embeddings import HuggingFaceEmbeddings # convert text into vector embeddings
from langchain.text_splitter import CharacterTextSplitter # split text or doc in to smaller chunks
from langchain.chains import RetrievalQA # retrieveing information from the embeddings

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# os.environ["OPENAI_API_KEY"]=""
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hashlib
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import json
import urllib.request
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SocketIOHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        socketio.emit('log', {'message': log_entry})

socketio_handler = SocketIOHandler()
socketio_handler.setLevel(logging.INFO)
logging.getLogger().addHandler(socketio_handler)


# Set upload folder paths relative to the current working directory
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define input and output folder paths for processing
input_folder = os.path.join(UPLOAD_FOLDER, 'board_images')
output_folder = os.path.join(UPLOAD_FOLDER, 'classified_images')

# Ensure the directories exist
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

@app.route('/members')
def members():
    return jsonify({'members': ["member1", "member2"]})  # Use jsonify for JSON responses

# Update the path below based on your Tesseract installation
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Route to handle video file uploads
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded video file
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)

    logging.info("Video uploaded successfully.")

    # Return relative path so frontend can access it via /uploads/filename
    return jsonify({'message': 'Video uploaded successfully!', 'video_path': f'/uploads/{file.filename}'})

# Video segmentation
def segment_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second
    segments = []
    last_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Segment every 100 frames
        if current_frame % 100 == 0:
            if last_frame != 0:
                start_time = last_frame / fps
                end_time = current_frame / fps
                segments.append((start_time, end_time))
            last_frame = current_frame

    cap.release()
    return segments

# Audio extraction
def extract_audio(input_file, output_file):
    try:
        video = VideoFileClip(input_file)
        audio = video.audio
        audio.write_audiofile(output_file, codec='libmp3lame')  # Specify MP3 format
    except Exception as e:
        return str(e)

# Create formatted timestamps for segments
def format_timestamp(seconds):
    minutes = math.floor(seconds / 60)
    seconds = math.floor(seconds % 60)
    return f"{minutes:02}:{seconds:02}"

@app.route('/upload_and_transcribe', methods=['POST'])
def upload_and_transcribe():
    video_path = request.json.get('video_path')
    if not video_path:
        return jsonify({'error': 'No video path provided'}), 400

    # Load Whisper model
    model = whisper.load_model("tiny")
    # Get the full file path based on the relative path provided
    video_full_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(video_path))
    audio_path = os.path.splitext(video_full_path)[0] + '.mp3'

    # Extract audio from the video
    error = extract_audio(video_full_path, audio_path)
    if error:
        return jsonify({'error': f"Audio extraction failed: {error}"}), 500

    # Transcribe the audio
    result = model.transcribe(audio=audio_path, task='transcription')

    # Generate segments and timestamps
    segments = segment_video(video_full_path)
    timestamps = [(format_timestamp(start), format_timestamp(end)) for start, end in segments]

    # Compile transcript with timestamps
    audio_transcript = ""
    for (segment, (start, end)) in zip(result['segments'], timestamps):
        audio_transcript += f"Timestamps ({start} - {end}): {segment['text']}\n"

    return jsonify({
        'transcript': audio_transcript,
        'audio_file': f'/uploads/{os.path.basename(audio_path)}'
    })



def extract_frames(video_path, start_time, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    frame_count = 0

    # Total frames to skip for 30-second intervals
    frame_skip = frame_interval * fps

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Start processing frames after the specified start time (in seconds)
        if frame_count >= start_time * fps and frame_count % frame_skip == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames

def detect_teacher_in_frame(model, frame):
    results = model(frame)
    for r in results:
        for box in r.boxes:
            label = box.cls[0]
            if label == 'person':  # Assuming 'person' class is indexed as the teacher
                return box.xyxy[0].tolist()  # Return bounding box of the teacher
    return None

def select_keyframes(frames, model):
    """
    Selects keyframes for each slide, picking frames where the teacher covers the least amount of text.

    Args:
    - frames: A list of frames extracted from the video.
    - model: YOLO model used for teacher detection.

    Returns:
    - keyframes: A list of selected keyframes, one for each detected slide.
    """
    keyframes = []
    current_slide_frames = []
    min_teacher_area = float('inf')
    best_frame = None

    for frame in frames:
        teacher_box = detect_teacher_in_frame(model, frame)

        if teacher_box:
            # Calculate teacher area in the frame
            x1, y1, x2, y2 = teacher_box
            area = (x2 - x1) * (y2 - y1)

            if area < min_teacher_area:
                min_teacher_area = area
                best_frame = frame

            # If the frame belongs to the same slide, accumulate it
            current_slide_frames.append(frame)
        else:
            # If no teacher detected, assume a slide change
            if best_frame:
                keyframes.append(best_frame)  # Save the best frame for the current slide
            else:
                keyframes.append(frame)  # Save the frame if no teacher detected

            # Reset for the next slide
            current_slide_frames = []
            min_teacher_area = float('inf')
            best_frame = None

    # Add the last slide's keyframe if not already added
    if best_frame:
        keyframes.append(best_frame)

    return keyframes

def clean_ocr_text(text):
    """
    Cleans the OCR text by removing unwanted characters, extra spaces,
    random numbers, and correcting any known OCR errors.
    """
    # Remove any standalone numbers, special characters, and unwanted symbols
    text = re.sub(r'\b\d+\b', '', text)  # Remove isolated numbers
    text = re.sub(r'[^\w\s.,!?-]', '', text)  # Keep only words, spaces, and basic punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces or newlines with a single space
    text = text.strip()  # Remove leading/trailing spaces

    return text

def format_text_with_bullets(text):
    """
    Formats the cleaned text by splitting it into sentences and adding bullet points,
    with multiple line spaces between each sentence.
    """
    # Clean the text before formatting
    text = clean_ocr_text(text)

    # Split text into sentences using punctuation as delimiters
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Add bullet points and separate sentences with three line breaks
    formatted_text = "\n\n\n".join([f"- {sentence.strip()}" for sentence in sentences if sentence.strip()])

    return formatted_text

def extract_text_from_frame(frame):
    """
    Extracts text from the given frame using pytesseract OCR, then cleans
    and formats it by adding bullet points and separating sentences.
    """

    if frame is None or not isinstance(frame, np.ndarray):
        print("Error: Invalid frame.")
        return ""

    # Convert the frame to grayscale for better OCR performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply OCR to extract text from the frame
    text = pytesseract.image_to_string(gray)

    # Format the OCR text by adding bullet points and multiple line spaces
    formatted_text = format_text_with_bullets(text)

    return formatted_text

def process_video(video_path, start_time=30, frame_interval=5):
    # Load the YOLO model
    model = YOLO("yolov8n.pt")

    # Step 1: Extract frames after 30 seconds
    frames = extract_frames(video_path, start_time, frame_interval)

    # Step 2: Detect and select keyframes with the least obstruction using the YOLO model
    keyframes = select_keyframes(frames, model)

    # Step 3: Extract and combine text from multiple keyframes using OCR
    extracted_texts = []

    for keyframe in keyframes:
        if keyframe is not None:
            text = extract_text_from_frame(keyframe)
            if text:
                extracted_texts.append(text)

    # Combine all the extracted texts
    combined_text = "\n\n\n".join(extracted_texts)

    return combined_text



nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    # Clean text of extra spaces
    text = " ".join(text.split())  # Normalize spaces
    return text

def extract_key_points(text):
    # Apply NLP processing
    doc = nlp(text)

    key_points = []

    # Loop through sentences
    for sent in doc.sents:
        # Check for entities or key words (nouns/verbs)
        entities = [ent.text for ent in sent.ents]
        key_words = [token.lemma_ for token in sent if token.pos_ in ['NOUN', 'VERB', 'PROPN']]

        # If the sentence contains entities or key words, consider it important
        if entities or key_words:
            key_points.append(sent.text.strip())

    return key_points

def remove_duplicate_sentences(sentences, threshold=0.8):
    # Vectorize the sentences
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # Calculate similarity between each pair of sentences
    similarity_matrix = cosine_similarity(vectors)

    # List to store unique sentences
    unique_sentences = []

    # Track whether a sentence is too similar to an earlier one
    for i, sentence in enumerate(sentences):
        if not any(similarity_matrix[i][j] > threshold for j in range(i)):
            unique_sentences.append(sentence)

    return unique_sentences

def format_output(points):
    # Format the extracted points as bullet points
    formatted_points = [f"- {point}" for point in points]
    return "\n".join(formatted_points)


def process_and_format_text(input_text):
    """
    This function takes an input text, splits it into manageable chunks,
    generates embeddings, stores them in a FAISS vector store, and finally
    uses an LLM to correct, deduplicate, and format the text.

    Args:
    - input_text (str): The text to be processed.

    Returns:
    - str: The processed and formatted text.
    """
    # Split the text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=20)
    documents = [Document(page_content=input_text)]
    texts = text_splitter.split_documents(documents)

    # Load vector embedding model and create FAISS vector store
    embeddings = HuggingFaceEmbeddings()
    kb = FAISS.from_documents(texts, embeddings)

    # Initialize OpenAI LLM for text correction and formatting
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Generate a prompt and process text with LLM
    prompt = (
        "You are given the following text extracted from an image. "
        "Please correct any spelling mistakes, remove duplicate points, and provide a neat and concise formatted version of the text:\n\n"
        f"{input_text}\n\n"
        "Return the corrected and formatted text."
    )
    response = llm([HumanMessage(content=prompt)])

    return response.content

def remove_duplicates(input_text):
    lines = input_text.split('\n')
    unique_lines = []
    seen = set()

    for line in lines:
        line_cleaned = line.strip()
        if line_cleaned and line_cleaned not in seen:
            unique_lines.append(line)
            seen.add(line_cleaned)

    return '\n'.join(unique_lines)

@app.route('/process_video', methods=['POST'])
def process_video_route():
    video_path = request.json.get('video_path')
    if not video_path:
        return jsonify({'error': 'No video path provided'}), 400

    # Get full path of video
    video_full_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(video_path))

    # Process the video to extract text
    extracted_text = process_video(video_full_path)  # Call the process_video function
    raw_text = extracted_text
    cleaned_text = clean_text(raw_text)
    key_points = extract_key_points(cleaned_text)
    unique_points = remove_duplicate_sentences(key_points, threshold=0.8)
    formatted_output = format_output(unique_points)
    formatted_result = process_and_format_text(formatted_output)
    formatted_res = remove_duplicates(formatted_result)


    return jsonify({'extracted_text':formatted_res })

# Function to extract keyframes from video
def extract_keyframes(video_path, frame_interval=60):
    """
    Extracts keyframes from a video at regular intervals.

    Args:
    - video_path: Path to the video file.
    - frame_interval: Interval (in seconds) at which to extract frames.

    Returns:
    - A list of extracted frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_count = 0
    interval = int(fps * frame_interval)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames


def upload_and_transcribe_video(video_path):
    # Load Whisper model
    model = whisper.load_model("tiny")
    audio_path = os.path.splitext(video_path)[0] + '.mp3'

    # Extract audio from the video
    error = extract_audio(video_path, audio_path)
    if error:
        return f"Audio extraction failed: {error}"

    # Transcribe the audio
    result = model.transcribe(audio=audio_path, task='transcription')
    segments = segment_video(video_path)
    timestamps = [(format_timestamp(start), format_timestamp(end)) for start, end in segments]

    # Compile transcript with timestamps
    audio_transcript = ""
    for (segment, (start, end)) in zip(result['segments'], timestamps):
        audio_transcript += f"Timestamps ({start} - {end}): {segment['text']}\n"

    return audio_transcript

def process_video_root(video_path):
    # Process the video to extract text
    extracted_text = process_video(video_path)
    cleaned_text = clean_text(extracted_text)
    key_points = extract_key_points(cleaned_text)
    unique_points = remove_duplicate_sentences(key_points, threshold=0.8)
    formatted_output = format_output(unique_points)
    final_result = process_and_format_text(formatted_output)
    return final_result


def summarize_transcript(combined_transcript, vector_store_path="faiss_store"):
    """
    Creates a FAISS vector store from the combined transcript and stores it for future queries.

    Args:
    - combined_transcript (str): The transcript text to process.
    - vector_store_path (str): Path to save the FAISS vector store.

    Returns:
    - str: The processed summary.
    """
    # Splitting the combined transcript into smaller chunks
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_text(combined_transcript)

    # Creating a FAISS vector store for retrieval
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_texts(documents, embeddings)

    # Save the vector store to disk
    vector_store.save_local(vector_store_path)

    # Query to summarize the transcript
    retriever = vector_store.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    query = "Provide a summary with appropriate headers and detailed content under each header."
    response = rag_chain.invoke({'query': query})['result']

    query1 = response + " add starting timestamps to headers without changing the content"
    response1 = rag_chain.invoke({'query': query1})['result']

    return response1


def translate_text(response,tgt_lang):
    try:
        sections = response.split('\n\n')

        # Translate each section
        translated_sections = [translate_section(section,tgt_lang) for section in sections]

        # Join the translated sections to preserve formatting
        final_translation = "\n\n".join(translated_sections)
        return final_translation

    except Exception as e:
        print(f"Error during translation: {e}")
        return f"Error during translation: {e}"


def translate_section(section,tgt_lang):
    translator = Translator(to_lang=tgt_lang)
    return translator.translate(section)

@app.route('/summarize_transcript', methods=['POST'])
def summarize_transcript_route():
    data = request.json
    video_path = data.get('video_path')
    target_lang = data.get('language', 'en')  # Default to English if no language is provided

    if not video_path:
        return jsonify({'error': 'No video path provided'}), 400

    video_full_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(video_path))

    # Call functions for audio and image transcripts
    audio_transcript = upload_and_transcribe_video(video_full_path)
    image_transcript = process_video_root(video_full_path)

    # Clean up transcripts
    audio_transcript_clean = re.sub(r'\s+', ' ', audio_transcript).strip()
    image_transcript_clean = re.sub(r'\s+', ' ', image_transcript).strip()

    # Combine both transcripts
    combined_transcript = f"Audio Transcript: {audio_transcript_clean}\n\nImage Transcript: {image_transcript_clean}"

    # Summarize the combined transcript
    summarized_transcript = summarize_transcript(combined_transcript)

    tgt_lang = 'en'

    # Translate the summarized transcript
    if target_lang=='Hindi':
        tgt_lang='hi'
    elif target_lang=='French':
        tgt_lang='fr'
    translated_text = translate_text(summarized_transcript, tgt_lang)

    # Return both summarized and translated transcripts
    return jsonify({
        'summarized_transcript': summarized_transcript,
        'translated_text': translated_text
    })


def summarize_query(query, vector_store_path="faiss_store"):
    """
    Accesses the FAISS vector store and retrieves information based on the input query.

    Args:
    - query (str): The input query to retrieve and summarize information.
    - vector_store_path (str): Path to the saved FAISS vector store.

    Returns:
    - str: The processed output for the input query.
    """
    # Load the FAISS vector store from disk
    if not os.path.exists(vector_store_path):
        raise FileNotFoundError(f"Vector store not found at {vector_store_path}. Ensure it has been created.")

    vector_store = FAISS.load_local(vector_store_path, HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)

    retriever = vector_store.as_retriever()

    # Define the LLM and create the RetrievalQA chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False  # Set to True if you also want source documents
    )

    # Process the query
    response = rag_chain.invoke({'query': query})['result']
    return response

@app.route('/summarize_query', methods=['POST'])
def summarize_query_route():
    try:
        data = request.get_json()
        query = data.get('query', '')  # Ensure the key matches 'query'
        if not query:
            return jsonify({'response': 'No query provided'}), 400
        target_lang = data.get('language', 'en')
        
        # Process the query (Replace this with your logic)
        summarized_response = summarize_query(query)  # Your logic here

        # Translate the summarized transcript
        
        if target_lang=='Hindi':
            tgt_lang='hi'
        elif target_lang=='French':
            tgt_lang='fr'
        translated_response = translate_text(summarized_response, tgt_lang)


        # Return both the summarized and translated responses
        return jsonify({
            'summarized_response': summarized_response,
            'translated_response': translated_response
        }), 200
    except Exception as e:
        print(f"Error: {e}")  # Log the error
        return jsonify({'response': 'An error occurred while processing your query'}), 500

if __name__ == '__main__':
    logging.info("Starting Flask application.")
    app.run(debug=True)
