import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_combined_plot(audio_path, dpi=300):
   y, sr = librosa.load(audio_path, sr=None)

   fig = plt.figure(figsize=(12, 8))
   gs = fig.add_gridspec(2, 2, width_ratios=[40, 1], height_ratios=[1, 2])
   ax1 = fig.add_subplot(gs[0, 0])
   ax2 = fig.add_subplot(gs[1, 0])
   cax = fig.add_subplot(gs[1, 1])

   librosa.display.waveshow(y, sr=sr, ax=ax1, color='blue')
   ax1.set_ylabel('Amplitude')
   ax1.set_title('Waveform')
   ax1.set_xlabel('Time (sec)')
   ax1.set_xlim(0, 2.5)
   ax2.set_xlim(0, 2.5)
   ax1.axvline(x=ax1.get_xlim()[1], color='black', linewidth=1)

   D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
   img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax2, cmap='magma')
   ax2.set_ylabel('Frequency(kHz)')
   
   # Convert Hz to kHz in y-axis ticks
   y_ticks = ax2.get_yticks()
   ax2.set_yticks(y_ticks)
   ax2.set_ylim(0, 10000)
   ax2.set_yticklabels([f'{int(y/1000)}' for y in y_ticks])
  
   ax2.set_xlabel('Time (sec)')
   ax2.set_title('Spectrogram')
   ax2.set_facecolor('black')

   cbar = plt.colorbar(img, cax=cax, format='%+2.0f dB')
   cbar.set_label('Intensity (dB)', labelpad=10)

   plt.subplots_adjust(wspace=0.02, hspace=0.3)
   plt.savefig('combined_audio_plot.jpg', dpi=dpi, bbox_inches='tight', facecolor='white')
   plt.close()

# Usage
audio_file = "aahkohkimaki.mp3"
generate_combined_plot(audio_file)