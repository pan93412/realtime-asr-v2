import torchaudio
import torch
import pyaudio
import numpy as np
import wave
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline

def record_audio_with_silence_detection(
    silence_threshold=0.01,
    silence_duration=2.0,
    sample_rate=16000,
    chunk_size=1024
):
    """
    Record audio from microphone with silence detection.
    
    Args:
        silence_threshold: Threshold for silence detection (0-1)
        silence_duration: Duration of silence to stop recording (seconds)
        sample_rate: Audio sample rate
        chunk_size: Number of frames per buffer
    
    Returns:
        numpy array of recorded audio
    """
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )
    
    print("Recording... Speak now!")
    print("Recording will stop after", silence_duration, "seconds of silence")
    
    frames = []
    silent_chunks = 0
    silence_threshold_chunks = int(silence_duration * sample_rate / chunk_size)
    
    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            
            # Check if this chunk is silent
            if np.max(np.abs(audio_data)) < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
            
            frames.append(audio_data)
            
            # Stop if we've had enough silence
            if silent_chunks >= silence_threshold_chunks:
                break
                
    except KeyboardInterrupt:
        print("\nRecording stopped by user")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    # Combine all frames
    audio = np.concatenate(frames)
    print(f"Recorded {len(audio) / sample_rate:.2f} seconds of audio")
    
    return audio

# 1. Record audio from microphone
print("Starting microphone recording...")
waveform = record_audio_with_silence_detection()
sample_rate = 16000

# 2. Preprocess (waveform is already mono and at correct sample rate)
waveform = waveform.squeeze()

# 3. Load Model
processor = WhisperProcessor.from_pretrained("MediaTek-Research/Breeze-ASR-25")
assert isinstance(processor, WhisperProcessor), "processor is not a WhisperProcessor"

model = WhisperForConditionalGeneration.from_pretrained("MediaTek-Research/Breeze-ASR-25")
model = model.to("mps" if torch.backends.mps.is_available() else "cpu")
model = model.eval()
assert isinstance(model, WhisperForConditionalGeneration), "model is not a WhisperForConditionalGeneration"

# 4. Build Pipeline
asr_pipeline = AutomaticSpeechRecognitionPipeline(
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=0
)

# 5. Inference
output = asr_pipeline(waveform, return_timestamps=True)  
assert isinstance(output, dict), "output is not a dict"
print("Result:", output["text"])
