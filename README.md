# Chatbot
This project is a voice-based chatbot that uses OpenAI's API to interact with users. The chatbot listens to user input via voice, processes the input using OpenAI's `whisper` model for transcription, generates a response using the `GPT-3.5-turbo` model, and replies using text-to-speech model `tts-1`.

---

## Features
- **Voice Input**: Users interact with the chatbot using their voice.
- **Real-Time Transcription**: Converts voice input into text using OpenAI's `whisper` model.
- **Intelligent Responses**: Generates context-aware and empathetic responses using OpenAI's `GPT-3.5-turbo` model.
- **Voice Output**: Uses OpenAI's `tts-1` model to respond back in audio form.
- **Continuous Conversation**: Maintains conversation history for context-aware interactions.
- **Language support**: `whisper` can detect any language. **Your accent can affect language detection!**

---

## Prerequisites
- Python 3
- OpenAI API Key
- Virtual environment (optional but recommended)
- A working microphone and speaker
  
---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/EncodedMind/Chatbot.git
   cd Chatbot
   ```
2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Set your OpenAI API key
   Replace `YOUR_OPENAI_API_KEY` in the script with your actual OpenAI API key.

---

## Usage
1. Run the chatbot
```bash
python chatbot.py
```
2. Interact with the chatbot
- Speak into your microphone when prompted.
- Listen to the chatbot's response.
- Say "Goodbye" to end the session and exit the program.

---

## Project Structure
```bash
├── chatbot.py       # Main script for the chatbot
├── requirements.txt # List of dependencies
├── README.md        # Project documentation
```

---

## File Structure

- `chatbot.py`: Main script to run the chatbot.
- `requirements.txt`: List of dependencies required for the project.
- `Recording.wav`: Temporary file for storing audio recordings.

---

## Key Components
### Functions
- `record_audio(duration, fs, device)`: Captures audio input from the user.
- `get_chatgpt_response(conversation_history)`: Sends user messages to the `GPT-3.5-turbo` model and retrieves responses.
- `main()`: Orchestrates the chatbot's voice input/output loop and conversation logic.

### Libraries Used
- **OpenAI**: For GPT and Whisper APIs.
- **Sounddevice**: For capturing audio input.
- **Numpy**: For handling audio data.
- **Pydub**: For playing audio responses.
- **Scipy**: For saving audio files as `.wav`.
- **Requests**: For making HTTP requests to the OpenAI API.

---

## Future Improvements
- **Silence Detection**: Avoid the limitation of defining a `duration` parameter.

---

## Notes
**Ensure you have a stable internet connection for API calls.** The interaction pause time (or response speed) of the chatbot depends on the quality and speed of your internet connection, as it relies on real-time API calls to OpenAI servers.
The `requirements.txt` file includes all necessary libraries.
Modify the chatbot behavior by changing the initial conversation_history system message.

---

## Troubleshooting
Audio Issues: Ensure your microphone is properly configured and accessible to the program.
API Errors: Verify your API key and check for usage limits on your OpenAI account.
Dependency Issues: Ensure all required libraries are installed using `pip install -r requirements.txt`.

---

## Project Extension: Teddy Bear Voice Assistant

In addition to the core functionality of the voice chatbot, I extended the project by integrating it with an **Asus Tinker Board**. The board acts as the central controller for the chatbot's operations. I connected a **microphone** and **speaker** to the Tinker Board and embedded everything inside a **teddy bear** to create a kid-friendly interactive toy, similar to the **Furby**.

This project extension aims to provide a fun and engaging experience for children while maintaining the capabilities of the voice chatbot.

### Watch the Interaction in Action
You can see the teddy bear in action with the chatbot by watching this video on YouTube:

- [Video: Teddy Bear Voice Assistant Demo](https://www.youtube.com/watch?v=SeFYSADezcM)

---

## License
This project is open-source and available under the MIT License.

---

## Acknowledgements
- OpenAI for providing the models.
- The developers of `Sounddevice`, `Pydub` and `Scipy` for enabling audio processing.

---

Enjoy using the Chatbot!
