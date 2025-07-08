import os
import datetime
from flask import Flask, request, render_template, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import whisper
from transformers import pipeline
import sqlite3
import json
from pathlib import Path
import moviepy.editor as mp
import tempfile
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Configuration
UPLOAD_FOLDER = 'uploads'
TRANSCRIPT_FOLDER = 'transcripts'
SUMMARY_FOLDER = 'summaries'
INDEX_FOLDER = 'index'
ALLOWED_EXTENSIONS = {'mp4', 'wav', 'mp3', 'avi', 'mov', 'txt'}

# Create directories
for folder in [UPLOAD_FOLDER, TRANSCRIPT_FOLDER, SUMMARY_FOLDER, INDEX_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Global variables for models (lazy loading)
whisper_model = None
summarizer = None
sentence_model = None
model_lock = threading.Lock()

def load_models():
    """Load models with error handling and smaller variants"""
    global whisper_model, summarizer, sentence_model
    
    try:
        if whisper_model is None:
            logger.info("Loading Whisper model...")
            whisper_model = whisper.load_model("tiny")  # Smallest model
            logger.info("Whisper model loaded successfully")
        
        if summarizer is None:
            logger.info("Loading summarization model...")
            summarizer = pipeline("summarization", 
                                model="sshleifer/distilbart-cnn-12-6",
                                device=-1)  # Force CPU
            logger.info("Summarization model loaded successfully")
        
        if sentence_model is None:
            logger.info("Loading sentence transformer...")
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def get_models():
    """Thread-safe model getter"""
    with model_lock:
        if whisper_model is None or summarizer is None or sentence_model is None:
            load_models()
    return whisper_model, summarizer, sentence_model

# Database setup
def init_db():
    conn = sqlite3.connect('meeting_assistant.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            transcript_path TEXT NOT NULL,
            summary_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            indexed BOOLEAN DEFAULT FALSE,
            processing_status TEXT DEFAULT 'completed'
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            error_message TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_datetime_path():
    now = datetime.datetime.now()
    return now.strftime("%Y/%m/%d"), now.strftime("%Y-%m-%d-%H-%M")

def extract_audio_from_video(video_path):
    """Extract audio from video file with error handling"""
    try:
        video = mp.VideoFileClip(video_path)
        audio_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        return audio_path
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper with error handling"""
    try:
        whisper_model, _, _ = get_models()
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        return None

def save_transcript(transcript, date_path, filename):
    """Save transcript to organized folder structure"""
    folder_path = os.path.join(TRANSCRIPT_FOLDER, date_path)
    os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, f"{filename}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    return file_path

def summarize_text(text):
    """Summarize text using BART model with chunking"""
    try:
        _, summarizer, _ = get_models()
        
        # Split text into smaller chunks to avoid memory issues
        max_chunk_length = 800  # Reduced for lighter model
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_chunk_length):
            chunk = ' '.join(words[i:i + max_chunk_length])
            if len(chunk.strip()) > 100:
                chunks.append(chunk)
        
        summaries = []
        for chunk in chunks:
            try:
                summary = summarizer(chunk, 
                                   max_length=100,  # Reduced
                                   min_length=30,   # Reduced
                                   do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                logger.warning(f"Error summarizing chunk: {e}")
                continue
        
        return ' '.join(summaries) if summaries else "Summary generation failed for this content."
        
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        return "Summary generation failed."

def save_summary(summary, date_path, filename):
    """Save summary to organized folder structure"""
    folder_path = os.path.join(SUMMARY_FOLDER, date_path)
    os.makedirs(folder_path, exist_ok=True)
    
    file_path = os.path.join(folder_path, f"{filename}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    return file_path

def index_transcript(transcript_path, transcript_id):
    """Create embeddings for transcript and store in index"""
    try:
        _, _, sentence_model = get_models()
        
        with open(transcript_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into sentences for better search
        sentences = text.split('. ')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Limit sentences to avoid memory issues
        if len(sentences) > 100:
            sentences = sentences[:100]
        
        # Create embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            embeddings = sentence_model.encode(batch)
            all_embeddings.extend(embeddings)
        
        # Save to index
        index_data = {
            'transcript_id': transcript_id,
            'sentences': sentences,
            'embeddings': [emb.tolist() for emb in all_embeddings]
        }
        
        index_path = os.path.join(INDEX_FOLDER, f"transcript_{transcript_id}.pkl")
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        return True
    except Exception as e:
        logger.error(f"Error indexing transcript: {e}")
        return False

def search_transcripts(query, top_k=5):
    """Search through indexed transcripts with error handling"""
    try:
        _, _, sentence_model = get_models()
        query_embedding = sentence_model.encode([query])
        results = []
        
        # Load all indexes
        for index_file in os.listdir(INDEX_FOLDER):
            if index_file.endswith('.pkl'):
                try:
                    index_path = os.path.join(INDEX_FOLDER, index_file)
                    with open(index_path, 'rb') as f:
                        index_data = pickle.load(f)
                    
                    # Calculate similarities
                    embeddings = np.array(index_data['embeddings'])
                    similarities = cosine_similarity(query_embedding, embeddings)[0]
                    
                    # Get top matches
                    top_indices = np.argsort(similarities)[::-1][:top_k]
                    
                    for idx in top_indices:
                        if similarities[idx] > 0.2:  # Lower threshold
                            results.append({
                                'transcript_id': index_data['transcript_id'],
                                'sentence': index_data['sentences'][idx],
                                'similarity': float(similarities[idx])
                            })
                except Exception as e:
                    logger.warning(f"Error processing index file {index_file}: {e}")
                    continue
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    except Exception as e:
        logger.error(f"Error searching transcripts: {e}")
        return []

def process_file_background(file_path, original_filename):
    """Process file in background to avoid timeout"""
    try:
        date_path, time_filename = get_datetime_path()
        
        # Check if it's a video file
        if original_filename.lower().endswith(('.mp4', '.avi', '.mov')):
            audio_path = extract_audio_from_video(file_path)
            if not audio_path:
                raise Exception("Failed to extract audio from video")
            transcript = transcribe_audio(audio_path)
            os.remove(audio_path)  # Clean up
        else:
            # Audio file
            transcript = transcribe_audio(file_path)
        
        if not transcript:
            raise Exception("Failed to transcribe audio")
        
        # Save transcript
        transcript_path = save_transcript(transcript, date_path, time_filename)
        
        # Save to database
        conn = sqlite3.connect('meeting_assistant.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transcripts (filename, original_filename, transcript_path, processing_status)
            VALUES (?, ?, ?, ?)
        ''', (time_filename, original_filename, transcript_path, 'completed'))
        transcript_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Index the transcript
        if index_transcript(transcript_path, transcript_id):
            conn = sqlite3.connect('meeting_assistant.db')
            cursor = conn.cursor()
            cursor.execute('UPDATE transcripts SET indexed = TRUE WHERE id = ?', (transcript_id,))
            conn.commit()
            conn.close()
        
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        logger.info(f"Successfully processed file: {original_filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing file {original_filename}: {e}")
        # Update database with error
        conn = sqlite3.connect('meeting_assistant.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO transcripts (filename, original_filename, transcript_path, processing_status)
            VALUES (?, ?, ?, ?)
        ''', (original_filename, original_filename, '', f'failed: {str(e)}'))
        conn.commit()
        conn.close()
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process file in background thread
        thread = threading.Thread(target=process_file_background, args=(file_path, filename))
        thread.start()
        
        flash('File uploaded successfully! Processing in background...')
        return redirect(url_for('transcripts'))
    
    flash('Invalid file type')
    return redirect(url_for('index'))

@app.route('/transcripts')
def transcripts():
    conn = sqlite3.connect('meeting_assistant.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM transcripts ORDER BY created_at DESC')
    transcripts = cursor.fetchall()
    conn.close()
    
    return render_template('transcripts.html', transcripts=transcripts)

@app.route('/summarize/<int:transcript_id>')
def summarize_transcript(transcript_id):
    conn = sqlite3.connect('meeting_assistant.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM transcripts WHERE id = ?', (transcript_id,))
    transcript_record = cursor.fetchone()
    
    if not transcript_record:
        flash('Transcript not found')
        return redirect(url_for('transcripts'))
    
    try:
        # Read transcript
        with open(transcript_record[3], 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        # Generate summary
        summary = summarize_text(transcript_text)
        
        # Save summary
        date_path, time_filename = get_datetime_path()
        summary_path = save_summary(summary, date_path, time_filename)
        
        # Update database
        cursor.execute('UPDATE transcripts SET summary_path = ? WHERE id = ?', 
                      (summary_path, transcript_id))
        conn.commit()
        conn.close()
        
        flash('Summary generated successfully!')
        return redirect(url_for('transcripts'))
        
    except Exception as e:
        conn.close()
        flash(f'Error generating summary: {str(e)}')
        return redirect(url_for('transcripts'))

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/search', methods=['POST'])
def search_meetings():
    query = request.form.get('query', '').strip()
    
    if not query:
        flash('Please enter a search query')
        return redirect(url_for('search'))
    
    results = search_transcripts(query)
    
    # Get transcript details
    if results:
        transcript_ids = [r['transcript_id'] for r in results]
        placeholders = ','.join(['?'] * len(transcript_ids))
        
        conn = sqlite3.connect('meeting_assistant.db')
        cursor = conn.cursor()
        cursor.execute(f'SELECT * FROM transcripts WHERE id IN ({placeholders})', 
                      transcript_ids)
        transcripts = {row[0]: row for row in cursor.fetchall()}
        conn.close()
        
        # Combine results with transcript info
        for result in results:
            result['transcript_info'] = transcripts.get(result['transcript_id'])
    
    return render_template('search_results.html', results=results, query=query)

@app.route('/view/<int:transcript_id>')
def view_transcript(transcript_id):
    conn = sqlite3.connect('meeting_assistant.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM transcripts WHERE id = ?', (transcript_id,))
    transcript_record = cursor.fetchone()
    conn.close()
    
    if not transcript_record:
        flash('Transcript not found')
        return redirect(url_for('transcripts'))
    
    # Check if processing is complete
    if transcript_record[7] != 'completed':  # processing_status
        flash(f'Transcript processing status: {transcript_record[7]}')
        return redirect(url_for('transcripts'))
    
    # Read transcript
    try:
        with open(transcript_record[3], 'r', encoding='utf-8') as f:
            transcript_text = f.read()
    except FileNotFoundError:
        flash('Transcript file not found')
        return redirect(url_for('transcripts'))
    
    # Read summary if exists
    summary_text = None
    if transcript_record[4]:  # summary_path
        try:
            with open(transcript_record[4], 'r', encoding='utf-8') as f:
                summary_text = f.read()
        except FileNotFoundError:
            pass
    
    return render_template('view_transcript.html', 
                         transcript=transcript_record,
                         transcript_text=transcript_text,
                         summary_text=summary_text)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'models_loaded': whisper_model is not None and summarizer is not None and sentence_model is not None
    }

@app.errorhandler(413)
def too_large(e):
    flash('File is too large. Maximum size is 100MB.')
    return redirect(url_for('index'))

@app.errorhandler(500)
def internal_server_error(e):
    flash('An internal error occurred. Please try again.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    init_db()
    
    # Load models on startup in development
    if os.environ.get('FLASK_ENV') != 'production':
        try:
            load_models()
        except Exception as e:
            logger.warning(f"Could not load models on startup: {e}")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
