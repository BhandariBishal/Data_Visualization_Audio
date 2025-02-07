import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_audio_plots(audio_path, dpi=300):
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Generate and save waveform plot
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (sec)')  # Updated x-axis label
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig('waveform.png', dpi=dpi, bbox_inches='tight')
    plt.close()

    # Generate and save spectrogram plot
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (sec)')  # Updated x-axis label
    plt.ylabel('Frequency (Hz)')  # Updated y-axis label
    plt.tight_layout()
    plt.savefig('spectrogram.png', dpi=dpi, bbox_inches='tight')
    plt.close()

    print("Plots saved as 'waveform.png' and 'spectrogram.png'.")

# Usage
audio_file = "aahkohkimaki.mp3"  # Replace with your audio file path
generate_audio_plots(audio_file)
