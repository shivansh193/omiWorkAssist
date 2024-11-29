from flask import Flask, request, jsonify
import logging
import time
import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import threading
import google.generativeai as genai
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(log_dir / "mentor.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")  # Or set it directly if not using .env

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
generation_config = {
    "temperature": 0.75,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 2048,
}
model = genai.GenerativeModel(model_name="gemini-1.5-pro", generation_config=generation_config) # specify your model here

class MessageBuffer:
    def __init__(self):
        self.buffers = {}
        self.lock = threading.Lock()
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        self.silence_threshold = 120  # 2 minutes silence threshold
        self.min_words_after_silence = 5  # minimum words needed after silence

    def get_buffer(self, session_id):
        current_time = time.time()
        
        # Cleanup old sessions periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_old_sessions()
        
        with self.lock:
            if session_id not in self.buffers:
                self.buffers[session_id] = {
                    'messages': [],
                    'last_analysis_time': time.time(),
                    'last_activity': current_time,
                    'words_after_silence': 0,
                    'silence_detected': False
                }
            else:
                # Check for silence period
                time_since_activity = current_time - self.buffers[session_id]['last_activity']
                if time_since_activity > self.silence_threshold:
                    self.buffers[session_id]['silence_detected'] = True
                    self.buffers[session_id]['words_after_silence'] = 0
                    self.buffers[session_id]['messages'] = []  # Clear old messages after silence
                
                self.buffers[session_id]['last_activity'] = current_time
                
        return self.buffers[session_id]

    def cleanup_old_sessions(self):
        current_time = time.time()
        with self.lock:
            expired_sessions = [
                session_id for session_id, data in self.buffers.items()
                if current_time - data['last_activity'] > 3600  # Remove sessions older than 1 hour
            ]
            for session_id in expired_sessions:
                del self.buffers[session_id]
            self.last_cleanup = current_time

# Initialize message buffer
message_buffer = MessageBuffer()

ANALYSIS_INTERVAL = 120  # 2 minutes between analyses

user_facts = defaultdict(dict)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_gemini(prompt):
    try:
        response = model.generate_text(prompt=prompt)
        return response.text.strip()  # Access the text directly
    except Exception as e:
        logger.error(f"Error generating text with Gemini: {e}")
        raise e




def extract_topics_and_profession(discussion_text: str) -> dict:
    prompt = f"""
    You will analyze the provided conversation transcript and extract key topics discussed and infer the user's profession/role.
    Conversation Transcript:
    ```
    {discussion_text}
    ```
    Output a JSON object with the following structure:
    ```json
    {{
        "topics": ["topic1", "topic2", ...], // Array of strings representing the discussed topics.
        "profession": "profession" // String representing the inferred profession/role. If no profession is clearly inferable, output "unknown".
    }}
    ```
    """
    try:
        result = query_gemini(prompt)
        return json.loads(result)
    except Exception as e:
        logger.error(f"Error extracting topics and profession: {e}")
        return {"topics": [], "profession": "unknown"}


def create_notification_prompt(messages: list, session_id: str) -> dict:

    formatted_discussion = []
    for msg in messages:
        speaker = "user" if msg.get('is_user') else "other"
        formatted_discussion.append(f"{speaker}: {msg['text']}")

    discussion_text = "\n".join(formatted_discussion)

    analysis_results = extract_topics_and_profession(discussion_text)
    topics = analysis_results.get("topics", [])
    profession = analysis_results.get("profession", "unknown")

    user_facts[session_id]['profession'] = profession  # Update user facts
    user_facts[session_id]['topics'] = topics

    # Gemini Prompt with examples and targeted instructions
    system_prompt = f"""
    You are a career mentor AI assistant. Your goal is to help users maximize their career potential based on their daily activities.
    Analyze the conversation transcript below and provide targeted questions or insights relevant to the user's inferred profession.
    User's Inferred Profession: {profession}
    Conversation Transcript:
    ```
    {discussion_text}
    ```
    Instructions:
    1. If the user has expressed a problem, challenge, or question related to their profession, provide actionable advice focusing on concise, targeted solutions. If no clear problem/challenge is expressed, generate insightful questions relevant to the user's profession and daily activities that can stimulate reflection and help them maximize their potential. Aim for no more than 3 questions.
    2. Use specific examples and references from the conversation to tailor your advice or questions.
    3. Maintain a conversational, supportive tone. Encourage the user to reflect and take action.
    Example Interactions:
    Scenario 1 (Problem/Challenge):
    User (Software Engineer): I'm struggling to meet the deadline for my coding project. I'm feeling overwhelmed.
    AI: It sounds like you're facing a common challenge in software development. Given your tight deadline, I recommend focusing on the core features first. Can you create a prioritized list of tasks based on their impact and estimated time to complete? This can help you break down the project into more manageable steps. Also, consider communicating with your team lead about potential roadblocks and explore the possibility of extending the deadline if necessary.
    Scenario 2 (No Clear Problem/Challenge):
    User (Marketing Manager): Had a productive meeting with the sales team today. Discussed new strategies for customer acquisition.
    AI: That's great to hear! As a marketing manager, staying updated on customer acquisition trends is essential. Did you discuss any innovative channels for customer outreach? Also, are there any specific metrics you'll be tracking to measure the effectiveness of the new strategies? Considering today's discussion, are there any professional development resources that could enhance your team's skills in customer acquisition?
    Your Response:
    """
    return {"prompt": system_prompt, "params": [], "context": {}} # Updated return statement



@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        data = request.json
        session_id = data.get('session_id')
        segments = data.get('segments', [])
        
        if not session_id:
            logger.error("No session_id provided in request")
            return jsonify({"message": "No session_id provided"}), 400

        current_time = time.time()
        buffer_data = message_buffer.get_buffer(session_id)

        # Process new messages
        for segment in segments:
            if not segment.get('text'):
                continue

            text = segment['text'].strip()
            if text:
                timestamp = segment.get('start', 0) or current_time
                is_user = segment.get('is_user', False)

                # Count words after silence
                if buffer_data['silence_detected']:
                    words_in_segment = len(text.split())
                    buffer_data['words_after_silence'] += words_in_segment
                    
                    # If we have enough words, start fresh conversation
                    if buffer_data['words_after_silence'] >= message_buffer.min_words_after_silence:
                        buffer_data['silence_detected'] = False
                        buffer_data['last_analysis_time'] = current_time  # Reset analysis timer
                        logger.info(f"Silence period ended for session {session_id}, starting fresh conversation")

                can_append = (
                    buffer_data['messages'] and 
                    abs(buffer_data['messages'][-1]['timestamp'] - timestamp) < 2.0 and
                    buffer_data['messages'][-1].get('is_user') == is_user
                )

                if can_append:
                    buffer_data['messages'][-1]['text'] += ' ' + text
                else:
                    buffer_data['messages'].append({
                        'text': text,
                        'timestamp': timestamp,
                        'is_user': is_user
                    })

        # Check if it's time to analyze
        time_since_last_analysis = current_time - buffer_data['last_analysis_time']

        if (time_since_last_analysis >= ANALYSIS_INTERVAL and
            buffer_data['messages'] and
            not buffer_data['silence_detected']):

            sorted_messages = sorted(buffer_data['messages'], key=lambda x: x['timestamp'])

            notification_prompt_data = create_notification_prompt(sorted_messages, session_id)

            notification = query_gemini(notification_prompt_data['prompt'])

            buffer_data['last_analysis_time'] = current_time
            buffer_data['messages'] = []  # Clear the buffer


            logger.info(f"Sending notification: {notification}")
            return jsonify({"notification": notification}), 200

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

# Add start time tracking
start_time = time.time()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)