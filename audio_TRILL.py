"""
TRILLsson feature extraction for turn-based audio analysis.
Optimized for CPU with batch processing and efficient audio loading.
"""

import pickle
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import librosa

from path import get_text_audio_pairs

logger = logging.getLogger(__name__)

# --- CONFIGURATION REVISION ---
class TRILLssonConfig:
    """Configuration for TRILLsson feature extraction."""

    def __init__(self,
                 model_url: str = "https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3",
                 sample_rate: int = 16000,
                 batch_size: int = 4,
                 feature_dim: int = 512):

        self.model_url = model_url
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.feature_dim = feature_dim

# --- EXTRACTOR CLASS REVISION ---
class TRILLssonFeatureExtractor:
    """TRILLsson feature extractor optimized for CPU batch processing."""

    def __init__(self, config: Optional[TRILLssonConfig] = None):
        """Initialize the TRILLsson feature extractor with CPU optimizations."""
        self.config = config or TRILLssonConfig()

        # Check for GPU availability and configure accordingly
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Allow memory growth to avoid OOM errors
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s), enabling CUDA")
            except RuntimeError as e:
                logger.warning(f"GPU configuration failed: {e}")
        else:
            logger.info("No GPU found, using CPU")
            # Keep CPU threading config only if no GPU
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)

        try:
            # Download model from Kaggle using kagglehub
            import kagglehub

            # Extract model handle from URL
            # Example: "google/nonsemantic-speech-benchmark/tensorFlow2/trill"
            logger.info(f"Downloading model from Kaggle...")
            model_path = kagglehub.model_download(self.config.model_url)
            logger.info(f"Model downloaded to: {model_path}")

            # Load the downloaded model
            self.model = hub.load(model_path)
            logger.info(f"Initialized TRILLsson model")
            logger.info(f"CPU optimizations enabled: graph mode, auto-threading")

        except Exception as e:
            logger.error(f"Failed to initialize TRILLsson model: {e}")
            raise

    def _load_full_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Load entire audio file once into memory."""
        try:
            # Load complete audio file
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)

            # Normalize to [-1, 1]
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            logger.info(f"Loaded audio file: {audio_path} ({len(audio)/sr:.2f}s)")
            return audio

        except Exception as e:
            logger.error(f"Failed to load audio from {audio_path}: {e}")
            raise

    def _extract_audio_segments(self, full_audio: np.ndarray, turns_df: pd.DataFrame) -> list:
        """Extract audio segments for all turns from loaded audio."""
        segments = []

        for _, turn in turns_df.iterrows():
            start_sample = int(turn['start'] * self.config.sample_rate)
            end_sample = int(turn['end'] * self.config.sample_rate)

            # Extract segment from memory
            segment = full_audio[start_sample:end_sample]

            # Handle very short or empty segments
            if len(segment) < 1000:  # Less than ~60ms at 16kHz
                segment = np.zeros(1000, dtype=np.float32)

            segments.append(segment)

        return segments

    def _pad_segments_to_batch(self, segments: list) -> np.ndarray:
        """Pad variable-length audio segments to same length for batching."""
        # Find maximum length
        max_len = max(len(seg) for seg in segments)

        # Pad all segments to max length
        padded = np.zeros((len(segments), max_len), dtype=np.float32)
        for i, seg in enumerate(segments):
            padded[i, :len(seg)] = seg

        return padded

    def _extract_features_batch(self, audio_segments: list) -> np.ndarray:
        """
        Extract TRILLsson features for multiple audio segments using batch inference.

        Args:
            audio_segments: List of audio numpy arrays

        Returns:
            Feature matrix (num_segments, 1024) - one 1024-dim vector per segment
        """
        if not audio_segments:
            return np.array([])

        # Process in smaller batches to avoid memory issues
        batch_size = self.config.batch_size
        all_embeddings = []

        for i in range(0, len(audio_segments), batch_size):
            batch = audio_segments[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(audio_segments)-1)//batch_size + 1} ({len(batch)} segments)")

            try:
                # Pad segments to same length for batch processing
                padded_batch = self._pad_segments_to_batch(batch)

                # Convert to TensorFlow tensor: shape [batch_size, time]
                batch_tensor = tf.convert_to_tensor(padded_batch, dtype=tf.float32)

                # Run batch inference with sample_rate parameter
                outputs = self.model(batch_tensor, sample_rate=self.config.sample_rate)

                # Debug: Check output structure (only first batch)
                if i == 0:
                    logger.info(f"Output type: {type(outputs)}")
                    if isinstance(outputs, dict):
                        logger.info(f"Available keys: {list(outputs.keys())}")
                        logger.info(f"Output shape for 'embedding': {outputs['embedding'].shape}")

                # Extract frame-level embeddings: shape [batch_size, time_steps, feature_dim]
                frame_embeddings = outputs['embedding'].numpy()

                # Aggregate over time to get utterance-level embeddings
                # Using mean pooling: [batch_size, time_steps, 512] -> [batch_size, 512]
                embeddings = np.mean(frame_embeddings, axis=1)

                # Debug: Check embeddings (only first batch)
                if i == 0:
                    logger.info(f"Frame embeddings shape: {frame_embeddings.shape}")
                    logger.info(f"Utterance embeddings shape (after mean pooling): {embeddings.shape}")
                    logger.info(f"Embeddings stats - min: {embeddings.min():.4f}, max: {embeddings.max():.4f}, mean: {embeddings.mean():.4f}")

                all_embeddings.append(embeddings)

            except Exception as e:
                logger.error(f"Batch feature extraction failed for batch {i//batch_size + 1}: {e}")
                # Fallback: return zero vectors for this batch
                all_embeddings.append(np.zeros((len(batch), self.config.feature_dim)))

        return np.vstack(all_embeddings)

    def extract_turn_features(self, audio_path: Union[str, Path],
                            turns_df: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Extract TRILLsson features for all turns using optimized batch processing.

        Process:
        1. Load entire audio file once
        2. Extract all turn segments from memory
        3. Process all segments in batches
        4. Return 1024-dim numeric vector per turn
        """
        logger.info(f"Processing {len(turns_df)} turns from {audio_path}")

        # Step 1: Load full audio file once (CPU Optimization 3)
        full_audio = self._load_full_audio(audio_path)

        # Step 2: Extract all segments from memory (no repeated I/O)
        audio_segments = self._extract_audio_segments(full_audio, turns_df)

        # Step 3: Batch inference - process all segments together (CPU Optimization 4)
        all_features = self._extract_features_batch(audio_segments)

        # Step 4: Map features back to turn_ids
        turn_features = {}
        for idx, (_, turn) in enumerate(turns_df.iterrows()):
            turn_id = turn['turn_id']
            turn_features[turn_id] = all_features[idx]

        logger.info(f"Extracted {len(turn_features)} feature vectors")

        # Debug: Check sample features before returning
        if turn_features:
            sample_turn_id = list(turn_features.keys())[0]
            sample_features = turn_features[sample_turn_id]
            logger.info(f"Sample turn {sample_turn_id} - shape: {sample_features.shape}, stats: min={sample_features.min():.4f}, max={sample_features.max():.4f}, mean={sample_features.mean():.4f}")

        return turn_features

    def create_feature_dataframe(self, turn_features: Dict[int, np.ndarray],
                                turns_df: pd.DataFrame) -> pd.DataFrame:
        """Create DataFrame with turn metadata and 1024-dim feature vectors."""
        results = []

        for turn_id, features in turn_features.items():
            turn_info = turns_df[turns_df['turn_id'] == turn_id].iloc[0]

            result = {
                'turn_id': turn_id,
                'speaker': turn_info['speaker'],
                'text': turn_info.get('utterance', turn_info.get('clean_utterance', '')),
                'start': turn_info['start'],
                'end': turn_info['end'],
                'duration': turn_info['end'] - turn_info['start']
            }

            # Add 1024 numeric features (trillsson_feature_000 to trillsson_feature_1023)
            for i, feature_value in enumerate(features):
                result[f'trillsson_feature_{i:03d}'] = feature_value

            results.append(result)

        return pd.DataFrame(results)

def save_turn_features(turn_features: Dict[int, np.ndarray],
                      turns_df: pd.DataFrame,
                      output_path: Union[str, Path],
                      session_name: str) -> None:
    """Save extracted features to pickle file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Debug: Check features before saving
    if turn_features:
        sample_turn_id = list(turn_features.keys())[0]
        sample_features = turn_features[sample_turn_id]
        logger.info(f"Before saving - sample turn {sample_turn_id}: shape={sample_features.shape}, min={sample_features.min():.4f}, max={sample_features.max():.4f}")

    save_data = {
        'session_name': session_name,
        'turn_features': turn_features,
        'turn_metadata': turns_df.to_dict('records'),
        'feature_dim': 512,
        'num_turns': len(turn_features)
    }

    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)

    logger.info(f"Saved to {output_path}")

    # Debug: Verify saved file
    with open(output_path, 'rb') as f:
        loaded_data = pickle.load(f)
        loaded_features = loaded_data['turn_features']
        sample_loaded = loaded_features[sample_turn_id]
        logger.info(f"After loading - sample turn {sample_turn_id}: shape={sample_loaded.shape}, min={sample_loaded.min():.4f}, max={sample_loaded.max():.4f}")

def process_session(audio_path: Path, session_name: str, turns_df: pd.DataFrame,
                   output_dir: Path, extractor: TRILLssonFeatureExtractor) -> None:
    """Process single session: extract features and save."""
    logger.info(f"Processing: {session_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract features using batch processing
    turn_features = extractor.extract_turn_features(audio_path, turns_df)

    # Save as pickle (fast and efficient)
    output_path = output_dir / f"{session_name}_trillsson_features.pkl"
    save_turn_features(turn_features, turns_df, output_path, session_name)

    # Skip CSV to save time - pickle contains all the data
    # To load: with open('file.pkl', 'rb') as f: data = pickle.load(f)

    logger.info(f"Completed: {session_name}")


def main(transcription_file: str = "194_transcription_spID.pkl",
         audio_dir: str = "./audio",
         output_dir: str = "./trillsson_features_output",
         config: Optional[TRILLssonConfig] = None) -> None:
    """
    Main pipeline: Extract TRILLsson features from audio-text pairs.

    Output: 512-dim numeric vector per turn (saved as .pkl and .csv)
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting TRILLsson extraction (GPU/CPU auto-detect)")

    # Load audio-text pairs
    pairs = get_text_audio_pairs(transcription_file, audio_dir)
    logger.info(f"Loaded {len(pairs)} sessions")

    # Initialize extractor once (model loads once, reused for all sessions)
    extractor = TRILLssonFeatureExtractor(config)
    output_path = Path(output_dir)

    # Process each session
    for idx, (audio_path, session_name, turns_df) in enumerate(pairs, 1):
        logger.info(f"[{idx}/{len(pairs)}] {session_name}")

        try:
            process_session(audio_path, session_name, turns_df, output_path, extractor)
        except Exception as e:
            logger.error(f"Failed {session_name}: {e}")
            continue

    logger.info(f"Complete! Results in {output_path}")


if __name__ == "__main__":
    config = TRILLssonConfig(
        model_url="google/nonsemantic-speech-benchmark/tensorFlow2/trill",
        batch_size=4  # Increase batch size for faster processing
    )

    main(config=config)