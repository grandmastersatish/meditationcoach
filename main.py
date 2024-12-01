from flask import Flask, request, jsonify
import logging
import time
import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import threading
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

app = Flask(__name__)

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(log_dir / "mindful.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
if not os.getenv('OPENAI_API_KEY'):
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise ValueError("OPENAI_API_KEY environment variable is required")

class MessageBuffer:
    def __init__(self):
        self.buffers = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self.silence_threshold = 120  # 2 minutes silence threshold
        self.min_words_after_silence = 5

    def get_buffer(self, session_id):
        current_time = time.time()
        
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_old_sessions()
        
        with self.lock:
            if session_id not in self.buffers:
                self.buffers[session_id] = {
                    'messages': [],
                    'last_analysis_time': time.time(),
                    'last_activity': current_time,
                    'meditation_state': 'beginning',
                    'session_length': 0,
                    'experience_level': 'beginner'
                }
            else:
                self.buffers[session_id]['last_activity'] = current_time
                
        return self.buffers[session_id]

    def cleanup_old_sessions(self):
        current_time = time.time()
        with self.lock:
            expired_sessions = [
                session_id for session_id, data in self.buffers.items()
                if current_time - data['last_activity'] > 3600
            ]
            for session_id in expired_sessions:
                del self.buffers[session_id]
            self.last_cleanup = current_time

message_buffer = MessageBuffer()

def generate_meditation_guidance(state: str, experience: str, length: int) -> str:
    """Generate meditation guidance using OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a calming meditation guide. Provide gentle, peaceful guidance appropriate for the current state and experience level."},
                {"role": "user", "content": f"Generate meditation guidance for: State: {state}, Experience: {experience}, Length: {length} minutes"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        guidance = response.choices[0].message.content.strip()
        logger.info("Meditation guidance generated successfully")
        return guidance
    except Exception as e:
        logger.error(f"Error generating guidance: {str(e)}")
        return "Take a deep breath and focus on your presence in this moment."

def create_meditation_response(session_data: dict) -> dict:
    """Create meditation guidance with template"""
    meditation_state = session_data.get('meditation_state', 'beginning')
    experience_level = session_data.get('experience_level', 'beginner')
    session_length = session_data.get('session_length', 10)
    
    guidance = generate_meditation_guidance(meditation_state, experience_level, session_length)
    
    return {
        "notification": {
            "guidance": guidance,
            "state": meditation_state,
            "remaining_time": session_length
        }
    }

@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        data = request.json
        session_id = data.get('session_id')
        state = data.get('state', 'beginning')
        
        if not session_id:
            logger.error("No session_id provided in request")
            return jsonify({"message": "No session_id provided"}), 400

        buffer_data = message_buffer.get_buffer(session_id)
        buffer_data['meditation_state'] = state

        response = create_meditation_response(buffer_data)
        
        return jsonify(response), 200

    return jsonify({}), 202

@app.route('/webhook/setup-status', methods=['GET'])
def setup_status():
    return jsonify({"is_setup_completed": True}), 200

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "active_sessions": len(message_buffer.buffers),
        "uptime": time.time() - start_time
    })

start_time = time.time()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
