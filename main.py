import torchaudio
import torch
import pyaudio
import numpy as np
import wave
import time
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline

def record_audio_with_silence_detection(
    silence_threshold=0.035,
    silence_duration=1.0,
    sample_rate=16000,
    chunk_size=512
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
    
    print("🎤 Recording... Speak now!")
    print(f"⏱️  Will stop after {silence_duration} seconds of silence")
    
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
        print("\n⏹️  Recording stopped by user")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    # Combine all frames
    audio = np.concatenate(frames)
    duration = len(audio) / sample_rate
    
    # Only process if we have meaningful audio (more than 0.5 seconds)
    if duration < 0.5:
        print("⚠️  Audio too short, skipping...")
        return None
    
    print(f"✅ Recorded {duration:.2f} seconds of audio")
    return audio

def main():
    print("🚀 Starting Real-Time ASR with Microphone Input")
    print("Press Ctrl+C to exit")
    print("-" * 50)
    
    # Load model and processor once (outside the loop for efficiency)
    print("📥 Loading ASR model...")
    processor = WhisperProcessor.from_pretrained("MediaTek-Research/Breeze-ASR-25")
    
    model = WhisperForConditionalGeneration.from_pretrained("MediaTek-Research/Breeze-ASR-25")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model = model.eval()
    
    # Build pipeline once using the processor directly
    asr_pipeline = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=0
    )
    
    print(f"✅ Model loaded on {device}")
    print("-" * 50)
    
    try:
        while True:
            # Record audio
            waveform = record_audio_with_silence_detection()
            
            if waveform is None:
                continue
            
            # Process audio
            print("🔄 Processing audio...")
            start_time = time.time()
            
            output = asr_pipeline(waveform, return_timestamps=False)
            
            processing_time = time.time() - start_time
            
            # Display results
            if isinstance(output, dict) and "text" in output:
                result_text = output["text"].strip()
                if result_text:
                    print(f"🎯 Result ({processing_time:.2f}s): {result_text}")
                else:
                    print("❌ No speech detected")
            else:
                print("❌ No speech detected")
            
            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
