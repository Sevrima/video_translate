"""
Audio comparison utility for voice resemblance analysis.
Compares two audio files and provides similarity score with visualizations.
"""
import argparse
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import soundfile as sf


def load_and_trim_audio(file1, file2, sr=22050):
    """
    Load two audio files and trim to the shorter duration.

    Args:
        file1 (str): Path to first audio file
        file2 (str): Path to second audio file
        sr (int): Target sampling rate

    Returns:
        tuple: (audio1, audio2, sr, duration)
    """
    # Load audio files
    audio1, sr1 = librosa.load(file1, sr=sr)
    audio2, sr2 = librosa.load(file2, sr=sr)

    # Get durations
    dur1 = len(audio1) / sr
    dur2 = len(audio2) / sr

    print(f"Audio 1 duration: {dur1:.2f}s")
    print(f"Audio 2 duration: {dur2:.2f}s")

    # Trim to shorter duration
    min_duration = min(dur1, dur2)
    min_samples = int(min_duration * sr)

    audio1 = audio1[:min_samples]
    audio2 = audio2[:min_samples]

    print(f"Comparing first {min_duration:.2f}s of both files")

    return audio1, audio2, sr, min_duration


def extract_features(audio, sr):
    """
    Extract audio features for comparison.

    Args:
        audio: Audio signal
        sr: Sample rate

    Returns:
        dict: Dictionary of features
    """
    features = {}

    # MFCC (Mel-frequency cepstral coefficients) - captures timbre
    features['mfcc'] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Mel spectrogram - frequency content
    features['mel_spec'] = librosa.feature.melspectrogram(y=audio, sr=sr)

    # Chroma features - pitch content
    features['chroma'] = librosa.feature.chroma_cqt(y=audio, sr=sr)

    # Spectral centroid - brightness
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)

    # Zero crossing rate - noisiness
    features['zcr'] = librosa.feature.zero_crossing_rate(audio)

    # Pitch (F0)
    features['pitch'] = librosa.yin(audio, fmin=50, fmax=500, sr=sr)

    return features


def compute_similarity_score(features1, features2):
    """
    Compute similarity score between two feature sets.

    Args:
        features1: Features from audio 1
        features2: Features from audio 2

    Returns:
        dict: Similarity scores for different metrics
    """
    scores = {}

    # MFCC similarity (most important for voice timbre)
    mfcc1_flat = features1['mfcc'].flatten()
    mfcc2_flat = features2['mfcc'].flatten()
    mfcc_similarity = 1 - cosine(mfcc1_flat, mfcc2_flat)
    scores['mfcc'] = max(0, min(1, mfcc_similarity))

    # Mel spectrogram similarity
    mel1_flat = features1['mel_spec'].flatten()
    mel2_flat = features2['mel_spec'].flatten()
    mel_similarity = 1 - cosine(mel1_flat, mel2_flat)
    scores['mel_spec'] = max(0, min(1, mel_similarity))

    # Chroma similarity (pitch content)
    chroma1_flat = features1['chroma'].flatten()
    chroma2_flat = features2['chroma'].flatten()
    chroma_similarity = 1 - cosine(chroma1_flat, chroma2_flat)
    scores['chroma'] = max(0, min(1, chroma_similarity))

    # Spectral centroid correlation
    sc1 = features1['spectral_centroid'].flatten()
    sc2 = features2['spectral_centroid'].flatten()
    sc_corr, _ = pearsonr(sc1, sc2)
    scores['spectral_centroid'] = max(0, min(1, (sc_corr + 1) / 2))

    # Pitch similarity
    pitch1 = features1['pitch'][~np.isnan(features1['pitch'])]
    pitch2 = features2['pitch'][~np.isnan(features2['pitch'])]
    if len(pitch1) > 0 and len(pitch2) > 0:
        min_len = min(len(pitch1), len(pitch2))
        pitch_corr, _ = pearsonr(pitch1[:min_len], pitch2[:min_len])
        scores['pitch'] = max(0, min(1, (pitch_corr + 1) / 2))
    else:
        scores['pitch'] = 0.5

    # Overall score (weighted average)
    weights = {
        'mfcc': 0.4,        # Most important for voice identity
        'mel_spec': 0.2,
        'chroma': 0.15,
        'spectral_centroid': 0.15,
        'pitch': 0.1
    }

    overall_score = sum(scores[key] * weights[key] for key in weights)
    scores['overall'] = overall_score

    return scores


def plot_comparison(audio1, audio2, features1, features2, sr, output_path='comparison.png'):
    """
    Create visualization comparing two audio files.

    Args:
        audio1: First audio signal
        audio2: Second audio signal
        features1: Features from audio 1
        features2: Features from audio 2
        sr: Sample rate
        output_path: Path to save the plot
    """
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))

    # Waveforms
    axes[0, 0].set_title('Audio 1 - Waveform')
    librosa.display.waveshow(audio1, sr=sr, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')

    axes[0, 1].set_title('Audio 2 - Waveform')
    librosa.display.waveshow(audio2, sr=sr, ax=axes[0, 1])
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')

    # Mel Spectrograms
    axes[1, 0].set_title('Audio 1 - Mel Spectrogram')
    img1 = librosa.display.specshow(
        librosa.power_to_db(features1['mel_spec'], ref=np.max),
        sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 0]
    )
    fig.colorbar(img1, ax=axes[1, 0], format='%+2.0f dB')

    axes[1, 1].set_title('Audio 2 - Mel Spectrogram')
    img2 = librosa.display.specshow(
        librosa.power_to_db(features2['mel_spec'], ref=np.max),
        sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 1]
    )
    fig.colorbar(img2, ax=axes[1, 1], format='%+2.0f dB')

    # MFCCs
    axes[2, 0].set_title('Audio 1 - MFCCs')
    img3 = librosa.display.specshow(
        features1['mfcc'], sr=sr, x_axis='time', ax=axes[2, 0]
    )
    fig.colorbar(img3, ax=axes[2, 0])

    axes[2, 1].set_title('Audio 2 - MFCCs')
    img4 = librosa.display.specshow(
        features2['mfcc'], sr=sr, x_axis='time', ax=axes[2, 1]
    )
    fig.colorbar(img4, ax=axes[2, 1])

    # Chroma
    axes[3, 0].set_title('Audio 1 - Chroma')
    img5 = librosa.display.specshow(
        features1['chroma'], sr=sr, x_axis='time', y_axis='chroma', ax=axes[3, 0]
    )
    fig.colorbar(img5, ax=axes[3, 0])

    axes[3, 1].set_title('Audio 2 - Chroma')
    img6 = librosa.display.specshow(
        features2['chroma'], sr=sr, x_axis='time', y_axis='chroma', ax=axes[3, 1]
    )
    fig.colorbar(img6, ax=axes[3, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Compare two audio files for voice resemblance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audio_compare.py --audio1 reference.wav --audio2 generated.wav
  python audio_compare.py --audio1 original.wav --audio2 cloned.wav --output comparison.png
        """
    )

    parser.add_argument('--audio1', type=str, required=True, help='Path to first audio file (reference)')
    parser.add_argument('--audio2', type=str, required=True, help='Path to second audio file (generated)')
    parser.add_argument('--output', type=str, default='comparison.png', help='Output path for comparison plot')
    parser.add_argument('--sr', type=int, default=22050, help='Sample rate for analysis')

    args = parser.parse_args()

    print("="*60)
    print("Audio Resemblance Comparison Tool")
    print("="*60)

    # Load and trim audio
    print("\nLoading audio files...")
    audio1, audio2, sr, duration = load_and_trim_audio(args.audio1, args.audio2, args.sr)

    # Extract features
    print("\nExtracting features...")
    features1 = extract_features(audio1, sr)
    features2 = extract_features(audio2, sr)

    # Compute similarity
    print("\nComputing similarity scores...")
    scores = compute_similarity_score(features1, features2)

    # Print results
    print("\n" + "="*60)
    print("SIMILARITY SCORES")
    print("="*60)
    print(f"MFCC Similarity (Voice Timbre):      {scores['mfcc']:.3f} ({scores['mfcc']*100:.1f}%)")
    print(f"Mel Spectrogram Similarity:           {scores['mel_spec']:.3f} ({scores['mel_spec']*100:.1f}%)")
    print(f"Chroma Similarity (Pitch Content):    {scores['chroma']:.3f} ({scores['chroma']*100:.1f}%)")
    print(f"Spectral Centroid Similarity:         {scores['spectral_centroid']:.3f} ({scores['spectral_centroid']*100:.1f}%)")
    print(f"Pitch Similarity:                     {scores['pitch']:.3f} ({scores['pitch']*100:.1f}%)")
    print("="*60)
    print(f"OVERALL RESEMBLANCE SCORE:            {scores['overall']:.3f} ({scores['overall']*100:.1f}%)")
    print("="*60)

    # Interpretation
    if scores['overall'] >= 0.8:
        quality = "Excellent"
    elif scores['overall'] >= 0.7:
        quality = "Good"
    elif scores['overall'] >= 0.6:
        quality = "Fair"
    elif scores['overall'] >= 0.5:
        quality = "Poor"
    else:
        quality = "Very Poor"

    print(f"\nQuality Assessment: {quality}")

    # Create visualization
    print("\nGenerating comparison plots...")
    plot_comparison(audio1, audio2, features1, features2, sr, args.output)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
