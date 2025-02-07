import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def generate_combined_plots(audio_paths, dpi=300):
    n_files = len(audio_paths)
    fig = plt.figure(figsize=(12*n_files, 8))
    
    # Define title font style
    title_font = {'fontname': 'DejaVu Sans', 'fontweight': 'semibold','fontsize': 14}
    
    time_ax = fig.add_subplot(111, frameon=False)
    time_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    time_ax.set_xlabel('Time (sec)', labelpad=20)
    
    for idx, audio_path in enumerate(audio_paths):
        y, sr = librosa.load(audio_path, sr=None)
        filename = audio_path.split("/")[-1].replace(".mp3", "")
        
        if idx < n_files - 1:
            gs = fig.add_gridspec(2, 1, 
                                height_ratios=[1, 2],
                                left=0.05+idx/n_files + 0.02, 
                                right=0.05+(idx+1)/n_files-0.02)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            
            librosa.display.waveshow(y, sr=sr, ax=ax1, color='blue')
            ax1.set_ylabel('Amplitude' if idx == 0 else '')
            ax1.set_title(f'{filename}', **title_font, pad=15)
            ax1.set_xlabel('')
            ax1.set_xlim(0, 2.5)
            
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', 
                                         ax=ax2, cmap='magma')
            ax2.set_ylabel('Frequency(kHz)' if idx == 0 else '')
            ax2.set_xlabel('')
            # ax2.set_title(f'{filename}', **title_font, pad=15)
            ax2.set_facecolor('black')
            ax2.set_xlim(0, 2.5)
            ax2.set_ylim(0, 10000)
            
            y_ticks = ax2.get_yticks()
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels([f'{int(y/1000)}' for y in y_ticks])
            
        else:
            gs = fig.add_gridspec(2, 2,
                                width_ratios=[40, 1],
                                height_ratios=[1, 2],
                                left=0.05+idx/n_files + 0.02,
                                right=0.95)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            cax = fig.add_subplot(gs[1, 1])
            
            librosa.display.waveshow(y, sr=sr, ax=ax1, color='blue')
            ax1.set_ylabel('Amplitude' if idx == 0 else '')
            ax1.set_title(f'{filename}', **title_font, pad=15)
            ax1.set_xlabel('')
            ax1.set_xlim(0, 2.5)
            
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz',
                                         ax=ax2, cmap='magma')
            ax2.set_ylabel('Frequency(kHz)' if idx == 0 else '')
            ax2.set_xlabel('')
            # ax2.set_title(f'{filename}', **title_font, pad=15)
            ax2.set_facecolor('black')
            ax2.set_xlim(0, 2.5)
            ax2.set_ylim(0, 10000)
            
            y_ticks = ax2.get_yticks()
            ax2.set_yticks(y_ticks)
            ax2.set_yticklabels([f'{int(y/1000)}' for y in y_ticks])
            
            cbar = plt.colorbar(img, cax=cax, format='%+2.0f dB')
            cbar.set_label('Intensity (dB)', labelpad=10)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.savefig('combined_audio_plots.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

# Usage
audio_files = ["aahkohkimaki.mp3", "peehki ceeliteeki noonki kaahkiihkwe.mp3"]
generate_combined_plots(audio_files)