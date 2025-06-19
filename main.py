import pyaudio
import numpy as np
import time
import torch
from faster_whisper import WhisperModel

def stream_audio_with_faster_whisper(
    silence_threshold=0.035,
    silence_duration=1.0,
    sample_rate=16000,
    chunk_size=1024
):
    """
    Stream audio from microphone with faster-whisper for real-time ASR.
    
    Args:
        silence_threshold: Threshold for silence detection (0-1)
        silence_duration: Duration of silence to stop recording (seconds)
        sample_rate: Audio sample rate
        chunk_size: Number of frames per buffer
    
    Returns:
        Generator yielding (audio_chunk, is_silent, should_stop)
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
    
    print("🎤 Streaming with Faster-Whisper... Speak now!")
    print(f"⏱️  Will process after {silence_duration} seconds of silence")
    
    silent_chunks = 0
    silence_threshold_chunks = int(silence_duration * sample_rate / chunk_size)
    
    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            
            # Check if this chunk is silent
            is_silent = np.max(np.abs(audio_data)) < silence_threshold
            
            if is_silent:
                silent_chunks += 1
            else:
                silent_chunks = 0
            
            should_stop = silent_chunks >= silence_threshold_chunks
            
            # Yield current state
            yield audio_data, is_silent, should_stop
            
            # Stop if we've had enough silence
            if should_stop:
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  Streaming stopped by user")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def main():
    print("🚀 Starting Real-Time ASR with Faster-Whisper")
    print("Press Ctrl+C to exit")
    print("-" * 50)
    
    # Load faster-whisper model (much faster than transformers)
    print("📥 Loading Faster-Whisper model...")
    model = WhisperModel(
        "base",  # or "small", "medium", "large" for different sizes
        device="mps" if torch.backends.mps.is_available() else "cpu",
        compute_type="float16"  # Use float16 for faster inference
    )
    
    print("✅ Model loaded")
    print("-" * 50)
    
    try:
        while True:
            print("🎤 Starting new recording session...")
            
            # Collect audio chunks
            audio_chunks = []
            
            # Stream audio
            for audio_chunk, is_silent, should_stop in stream_audio_with_faster_whisper():
                audio_chunks.append(audio_chunk)
                
                # Process when we detect silence
                if should_stop and len(audio_chunks) > 0:
                    print("🔄 Processing audio...")
                    start_time = time.time()
                    
                    # Combine all chunks
                    audio = np.concatenate(audio_chunks)
                    
                    # Process with faster-whisper (much faster!)
                    segments, info = model.transcribe(
                        audio,
                        beam_size=5,
                        language="zh",  # or None for auto-detection
                        vad_filter=True,  # Voice Activity Detection
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Display results
                    results = list(segments)
                    if results:
                        for segment in results:
                            print(f"🎯 Result ({processing_time:.2f}s): {segment.text.strip()}")
                    else:
                        print("❌ No speech detected")
                    
                    print("-" * 50)
                    break
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
