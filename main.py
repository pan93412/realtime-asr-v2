import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline

# 1. Load audio
audio_path = "./input_audio.wav"
waveform, sample_rate = torchaudio.load(audio_path)          

# 2. Preprocess
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)                         
waveform = waveform.squeeze().numpy()                        

if sample_rate != 16_000:
    resampler = torchaudio.transforms.Resample(sample_rate, 16_000)
    waveform = resampler(torch.tensor(waveform)).numpy()
    sample_rate = 16_000

# 3. Load Model
processor = WhisperProcessor.from_pretrained("MediaTek-Research/Breeze-ASR-25")
assert isinstance(processor, WhisperProcessor)

model = WhisperForConditionalGeneration.from_pretrained("MediaTek-Research/Breeze-ASR-25").to("mps").eval()

# 4. Build Pipeline
asr_pipeline = AutomaticSpeechRecognitionPipeline(
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=0
)

# 6. Inference
output = asr_pipeline(waveform, return_timestamps=True)  
print("Result:", output["text"])
