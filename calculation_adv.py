"""
Advanced speech alignment metrics using TRILLsson features.

This module implements three approaches to measure speech alignment between speakers:
1. Local Similarity: Immediate adaptation to partner's previous turn
2. Convergence Over Time: Whether speakers become more similar as conversation progresses
3. Synchrony: Windowed average comparison between speakers
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    if vec1.size == 0 or vec2.size == 0:
        return 0.0

    try:
        # Handle NaN values
        vec1_clean = np.nan_to_num(vec1, nan=0.0, posinf=0.0, neginf=0.0)
        vec2_clean = np.nan_to_num(vec2, nan=0.0, posinf=0.0, neginf=0.0)

        # Calculate cosine similarity
        dot_product = np.dot(vec1_clean, vec2_clean)
        norm1 = np.linalg.norm(vec1_clean)
        norm2 = np.linalg.norm(vec2_clean)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = dot_product / (norm1 * norm2)

        # Ensure result is in valid range [-1, 1]
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

        return float(cosine_sim)

    except Exception as e:
        warnings.warn(f"Cosine similarity calculation failed: {e}")
        return 0.0


def get_speaker_features(df: pd.DataFrame, speaker: str,
                        feature_cols: List[str]) -> np.ndarray:
    """Extract feature matrix for a specific speaker."""
    speaker_df = df[df['speaker'] == speaker]
    if len(speaker_df) == 0:
        return np.array([])
    return speaker_df[feature_cols].values


def calculate_local_similarity(df: pd.DataFrame,
                               feature_cols: List[str]) -> Dict[str, float]:
    """
    Approach #1: Local Similarity (within-dyad comparison)

    Measures how much a speaker adapts to their partner's previous turn.
    For each turn i+1 by speaker A following turn i by speaker B:
        local_sim = cosine(A_turn_i+1, B_turn_i) - cosine(A_turn_i+1, A_turn_i-1)

    Positive values indicate adaptation toward partner.

    Args:
        df: DataFrame with turn features sorted by turn_id
        feature_cols: List of feature column names

    Returns:
        Dictionary with local similarity statistics
    """
    local_similarities = []
    speaker_specific = {}  # Track per-speaker adaptation

    for i in range(1, len(df) - 1):
        prev_turn = df.iloc[i - 1]
        curr_turn = df.iloc[i]
        next_turn = df.iloc[i + 1]

        # Check if next turn is by different speaker than current
        if next_turn['speaker'] != curr_turn['speaker']:
            speaker_A = next_turn['speaker']
            speaker_B = curr_turn['speaker']

            # Get feature vectors
            A_next = next_turn[feature_cols].values  # A's response
            B_curr = curr_turn[feature_cols].values  # B's previous turn

            # Find A's previous turn (for baseline)
            A_prev_turns = df[(df['speaker'] == speaker_A) & (df.index < i)]

            if len(A_prev_turns) > 0:
                A_prev = A_prev_turns.iloc[-1][feature_cols].values

                # Calculate adaptation
                sim_to_partner = calculate_cosine_similarity(A_next, B_curr)
                sim_to_self = calculate_cosine_similarity(A_next, A_prev)
                local_sim = sim_to_partner - sim_to_self

                local_similarities.append(local_sim)

                # Track per-speaker
                if speaker_A not in speaker_specific:
                    speaker_specific[speaker_A] = []
                speaker_specific[speaker_A].append(local_sim)

    if not local_similarities:
        return {
            'local_sim_mean': 0.0,
            'local_sim_std': 0.0,
            'local_sim_median': 0.0,
            'n_pairs': 0
        }

    results = {
        'local_sim_mean': np.mean(local_similarities),
        'local_sim_std': np.std(local_similarities),
        'local_sim_median': np.median(local_similarities),
        'n_pairs': len(local_similarities)
    }

    # Add per-speaker statistics
    for speaker, sims in speaker_specific.items():
        results[f'local_sim_{speaker}_mean'] = np.mean(sims)

    return results


def calculate_convergence(df: pd.DataFrame,
                         feature_cols: List[str],
                         early_ratio: float = 0.3,
                         late_ratio: float = 0.3) -> Dict[str, float]:
    """
    Approach #2: Convergence Over Time

    Measures whether speakers become more similar as conversation progresses.
    Compares cross-speaker similarity in early vs. late portions of conversation.

    Args:
        df: DataFrame with turn features sorted by turn_id
        feature_cols: List of feature column names
        early_ratio: Proportion of conversation to consider "early" (default: first 30%)
        late_ratio: Proportion of conversation to consider "late" (default: last 30%)

    Returns:
        Dictionary with convergence statistics
    """
    total_turns = len(df)
    early_n = int(total_turns * early_ratio)
    late_start = total_turns - int(total_turns * late_ratio)

    if early_n < 2 or late_start >= total_turns - 1 or late_start <= early_n:
        return {
            'convergence': 0.0,
            'early_similarity': 0.0,
            'late_similarity': 0.0,
            'early_n_pairs': 0,
            'late_n_pairs': 0
        }

    # Get early and late portions
    early_df = df.iloc[:early_n]
    late_df = df.iloc[late_start:]

    # Calculate cross-speaker similarities for early portion
    early_sims = []
    for i in range(len(early_df) - 1):
        if early_df.iloc[i]['speaker'] != early_df.iloc[i + 1]['speaker']:
            vec1 = early_df.iloc[i][feature_cols].values
            vec2 = early_df.iloc[i + 1][feature_cols].values
            early_sims.append(calculate_cosine_similarity(vec1, vec2))

    # Calculate cross-speaker similarities for late portion
    late_sims = []
    for i in range(len(late_df) - 1):
        if late_df.iloc[i]['speaker'] != late_df.iloc[i + 1]['speaker']:
            vec1 = late_df.iloc[i][feature_cols].values
            vec2 = late_df.iloc[i + 1][feature_cols].values
            late_sims.append(calculate_cosine_similarity(vec1, vec2))

    early_mean = np.mean(early_sims) if early_sims else 0.0
    late_mean = np.mean(late_sims) if late_sims else 0.0

    return {
        'convergence': late_mean - early_mean,  # Positive = increasing similarity
        'early_similarity': early_mean,
        'late_similarity': late_mean,
        'early_n_pairs': len(early_sims),
        'late_n_pairs': len(late_sims)
    }


def calculate_proximity(df: pd.DataFrame,
                       feature_cols: List[str]) -> Dict[str, float]:
    """
    Approach #3a: Proximity (Static Overall Similarity)

    Measures whether speakers are generally similar across the entire conversation.
    This is a static measure of overall acoustic similarity.

    Args:
        df: DataFrame with turn features sorted by turn_id
        feature_cols: List of feature column names

    Returns:
        Dictionary with proximity statistics
    """
    speakers = df['speaker'].unique()

    if len(speakers) < 2:
        return {
            'proximity_mean': 0.0,
            'proximity_std': 0.0,
            'n_comparisons': 0
        }

    # Calculate overall average features per speaker
    speaker_means = {}
    for speaker in speakers:
        speaker_df = df[df['speaker'] == speaker]
        if len(speaker_df) > 0:
            speaker_means[speaker] = speaker_df[feature_cols].mean().values

    # Compare all speaker pairs
    proximity_scores = []
    speaker_list = list(speaker_means.keys())

    if len(speaker_list) >= 2:
        for i in range(len(speaker_list)):
            for j in range(i + 1, len(speaker_list)):
                sim = calculate_cosine_similarity(
                    speaker_means[speaker_list[i]],
                    speaker_means[speaker_list[j]]
                )
                proximity_scores.append(sim)

    if not proximity_scores:
        return {
            'proximity_mean': 0.0,
            'proximity_std': 0.0,
            'n_comparisons': 0
        }

    return {
        'proximity_mean': np.mean(proximity_scores),
        'proximity_std': np.std(proximity_scores),
        'n_comparisons': len(proximity_scores)
    }


def calculate_local_similarity_by_bins(df: pd.DataFrame,
                                       feature_cols: List[str],
                                       n_bins: int = 5) -> Dict[str, float]:
    """
    Calculate local similarity separately for temporal bins.
    Shows how adaptation patterns change throughout the conversation.

    Args:
        df: DataFrame with turn features sorted by turn_id
        feature_cols: List of feature column names
        n_bins: Number of temporal bins to divide conversation into

    Returns:
        Dictionary with local similarity for each bin
    """
    total_turns = len(df)
    bin_size = total_turns // n_bins

    if bin_size < 3:  # Need at least 3 turns per bin for meaningful calculation
        return {f'bin_{i+1}_local_sim_mean': 0.0 for i in range(n_bins)}

    results = {}

    for bin_idx in range(n_bins):
        start_idx = bin_idx * bin_size
        end_idx = start_idx + bin_size if bin_idx < n_bins - 1 else total_turns
        bin_df = df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Calculate local similarity for this bin
        bin_results = calculate_local_similarity(bin_df, feature_cols)
        results[f'bin_{bin_idx+1}_local_sim_mean'] = bin_results['local_sim_mean']
        results[f'bin_{bin_idx+1}_local_sim_n_pairs'] = bin_results['n_pairs']

    return results


def calculate_asymmetric_adaptation(df: pd.DataFrame,
                                    feature_cols: List[str]) -> Dict[str, float]:
    """
    Measure asymmetric adaptation between speakers.
    Identifies if one speaker (e.g., student) adapts more to the other (e.g., teacher).

    For each speaker pair, calculates:
    - How much Speaker A adapts to Speaker B
    - How much Speaker B adapts to Speaker A
    - The asymmetry (difference between the two)

    Args:
        df: DataFrame with turn features sorted by turn_id
        feature_cols: List of feature column names

    Returns:
        Dictionary with asymmetric adaptation metrics
    """
    speakers = df['speaker'].unique()

    if len(speakers) < 2:
        return {
            'asymmetry_score': 0.0,
            'dominant_speaker': 'none',
            'adaptive_speaker': 'none'
        }

    # Track adaptation per speaker (how much they adapt TO others)
    speaker_adaptation = {}

    for i in range(1, len(df) - 1):
        prev_turn = df.iloc[i - 1]
        curr_turn = df.iloc[i]
        next_turn = df.iloc[i + 1]

        # Check if next turn is by different speaker than current
        if next_turn['speaker'] != curr_turn['speaker']:
            speaker_A = next_turn['speaker']
            speaker_B = curr_turn['speaker']

            # Get feature vectors
            A_next = next_turn[feature_cols].values
            B_curr = curr_turn[feature_cols].values

            # Find A's previous turn (for baseline)
            A_prev_turns = df[(df['speaker'] == speaker_A) & (df.index < i)]

            if len(A_prev_turns) > 0:
                A_prev = A_prev_turns.iloc[-1][feature_cols].values

                # Calculate adaptation of A toward B
                sim_to_partner = calculate_cosine_similarity(A_next, B_curr)
                sim_to_self = calculate_cosine_similarity(A_next, A_prev)
                adaptation_score = sim_to_partner - sim_to_self

                # Track adaptation by speaker
                if speaker_A not in speaker_adaptation:
                    speaker_adaptation[speaker_A] = []
                speaker_adaptation[speaker_A].append(adaptation_score)

    if not speaker_adaptation:
        return {
            'asymmetry_score': 0.0,
            'dominant_speaker': 'none',
            'adaptive_speaker': 'none'
        }

    # Calculate mean adaptation for each speaker
    speaker_means = {
        speaker: np.mean(scores)
        for speaker, scores in speaker_adaptation.items()
    }

    results = {}

    # Add per-speaker adaptation scores
    for speaker, mean_score in speaker_means.items():
        results[f'adaptation_{speaker}'] = mean_score

    # Find most adaptive (positive) and least adaptive (negative/dominant)
    sorted_speakers = sorted(speaker_means.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_speakers) >= 2:
        adaptive_speaker = sorted_speakers[0][0]  # Highest adaptation
        dominant_speaker = sorted_speakers[-1][0]  # Lowest adaptation
        asymmetry = sorted_speakers[0][1] - sorted_speakers[-1][1]

        results['asymmetry_score'] = asymmetry
        results['adaptive_speaker'] = adaptive_speaker
        results['dominant_speaker'] = dominant_speaker
    else:
        results['asymmetry_score'] = 0.0
        results['adaptive_speaker'] = 'none'
        results['dominant_speaker'] = 'none'

    return results


def calculate_weighted_local_similarity(df: pd.DataFrame,
                                        feature_cols: List[str],
                                        weight_by: str = 'length_seconds') -> Dict[str, float]:
    """
    Calculate local similarity weighted by turn importance (duration or word count).
    Filters out noise from short backchannels by giving more weight to substantial turns.

    Args:
        df: DataFrame with turn features sorted by turn_id
        feature_cols: List of feature column names
        weight_by: Column name to use for weighting ('length_seconds' or 'words')

    Returns:
        Dictionary with weighted local similarity statistics
    """
    local_similarities = []
    weights = []

    for i in range(1, len(df) - 1):
        prev_turn = df.iloc[i - 1]
        curr_turn = df.iloc[i]
        next_turn = df.iloc[i + 1]

        # Check if next turn is by different speaker than current
        if next_turn['speaker'] != curr_turn['speaker']:
            speaker_A = next_turn['speaker']
            speaker_B = curr_turn['speaker']

            # Get feature vectors
            A_next = next_turn[feature_cols].values
            B_curr = curr_turn[feature_cols].values

            # Find A's previous turn (for baseline)
            A_prev_turns = df[(df['speaker'] == speaker_A) & (df.index < i)]

            if len(A_prev_turns) > 0:
                A_prev = A_prev_turns.iloc[-1][feature_cols].values

                # Calculate adaptation
                sim_to_partner = calculate_cosine_similarity(A_next, B_curr)
                sim_to_self = calculate_cosine_similarity(A_next, A_prev)
                local_sim = sim_to_partner - sim_to_self

                # Weight by turn importance (duration or word count)
                if weight_by in next_turn and not pd.isna(next_turn[weight_by]):
                    weight = max(next_turn[weight_by], 0.1)  # Avoid zero weights
                else:
                    weight = 1.0

                local_similarities.append(local_sim)
                weights.append(weight)

    if not local_similarities:
        return {
            'weighted_local_sim_mean': 0.0,
            'weighted_local_sim_std': 0.0,
            'n_pairs': 0
        }

    # Calculate weighted mean and std
    local_similarities = np.array(local_similarities)
    weights = np.array(weights)

    weighted_mean = np.average(local_similarities, weights=weights)
    weighted_variance = np.average((local_similarities - weighted_mean)**2, weights=weights)
    weighted_std = np.sqrt(weighted_variance)

    return {
        'weighted_local_sim_mean': weighted_mean,
        'weighted_local_sim_std': weighted_std,
        'weighted_n_pairs': len(local_similarities)
    }


def calculate_synchrony(df: pd.DataFrame,
                       feature_cols: List[str]) -> Dict[str, float]:
    """
    Approach #3b: Synchrony (Dynamic Temporal Co-variation)

    Measures whether speakers change together moment-to-moment.
    Calculates correlation of feature changes over time between speakers.

    Args:
        df: DataFrame with turn features sorted by turn_id
        feature_cols: List of feature column names

    Returns:
        Dictionary with synchrony statistics
    """
    speakers = df['speaker'].unique()

    if len(speakers) < 2:
        return {
            'synchrony_mean': 0.0,
            'synchrony_median': 0.0,
            'n_dimensions': 0
        }

    # Get speaker-specific feature sequences
    speaker_sequences = {}
    for speaker in speakers:
        speaker_df = df[df['speaker'] == speaker].sort_values('turn_id')
        if len(speaker_df) >= 2:  # Need at least 2 turns to compute diff
            speaker_sequences[speaker] = speaker_df[feature_cols].values

    speaker_list = list(speaker_sequences.keys())

    if len(speaker_list) < 2:
        return {
            'synchrony_mean': 0.0,
            'synchrony_median': 0.0,
            'n_dimensions': 0
        }

    # Calculate synchrony for each speaker pair
    all_synchrony_scores = []

    for i in range(len(speaker_list)):
        for j in range(i + 1, len(speaker_list)):
            speaker_A = speaker_list[i]
            speaker_B = speaker_list[j]

            seq_A = speaker_sequences[speaker_A]
            seq_B = speaker_sequences[speaker_B]

            # Calculate changes (derivatives) for each feature dimension
            changes_A = np.diff(seq_A, axis=0)  # Shape: (n_turns_A - 1, n_features)
            changes_B = np.diff(seq_B, axis=0)  # Shape: (n_turns_B - 1, n_features)

            # Align sequences to same length (use minimum length)
            min_len = min(len(changes_A), len(changes_B))
            if min_len < 2:
                continue

            changes_A_aligned = changes_A[:min_len]
            changes_B_aligned = changes_B[:min_len]

            # Calculate correlation for each feature dimension
            dimension_correlations = []
            for feat_idx in range(len(feature_cols)):
                A_feat_changes = changes_A_aligned[:, feat_idx]
                B_feat_changes = changes_B_aligned[:, feat_idx]

                # Check for valid variance
                if np.std(A_feat_changes) > 0 and np.std(B_feat_changes) > 0:
                    corr = np.corrcoef(A_feat_changes, B_feat_changes)[0, 1]
                    if not np.isnan(corr):
                        dimension_correlations.append(corr)

            # Average across dimensions for this speaker pair
            if dimension_correlations:
                pair_synchrony = np.mean(dimension_correlations)
                all_synchrony_scores.append(pair_synchrony)

    if not all_synchrony_scores:
        return {
            'synchrony_mean': 0.0,
            'synchrony_median': 0.0,
            'n_dimensions': 0
        }

    return {
        'synchrony_mean': np.mean(all_synchrony_scores),
        'synchrony_median': np.median(all_synchrony_scores),
        'synchrony_std': np.std(all_synchrony_scores),
        'n_pairs': len(all_synchrony_scores)
    }


def load_pickle_file(pkl_path: str) -> pd.DataFrame:
    """
    Load TRILLsson features from pickle file and convert to DataFrame.

    Args:
        pkl_path: Path to the pickle file

    Returns:
        DataFrame with turn metadata and feature columns
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Extract data
    turn_features = data['turn_features']  # Dict[turn_id, np.ndarray]
    turn_metadata = data['turn_metadata']  # List of dicts

    # Create DataFrame from metadata
    df = pd.DataFrame(turn_metadata)

    # Filter to keep only Teacher/Student speakers
    df = df[df['speaker'].str.contains('Teacher|teacher|Student|student', case=False, na=False)]

    # Filter out turns with duration < 2 or > 25 seconds
    df = df[(df['length_seconds'] >= 2) & (df['length_seconds'] <= 25)]
    df = df.reset_index(drop=True)

    # Create feature columns efficiently
    feature_dim = data['feature_dim']
    feature_data = []

    for _, row in df.iterrows():
        turn_id = row['turn_id']
        if turn_id in turn_features:
            feature_data.append(turn_features[turn_id])
        else:
            feature_data.append(np.zeros(feature_dim))

    # Create feature DataFrame in one go
    feature_cols = [f'trillsson_feature_{i:03d}' for i in range(feature_dim)]
    features_df = pd.DataFrame(feature_data, columns=feature_cols, index=df.index)

    # Concatenate metadata and features
    df = pd.concat([df, features_df], axis=1)

    return df


def process_file(pkl_path: str,
                early_ratio: float = 0.3,
                late_ratio: float = 0.3) -> Dict:
    """
    Process a single pickle file to calculate all alignment metrics.

    Args:
        pkl_path: Path to the pickle file
        early_ratio: Ratio for early portion in convergence
        late_ratio: Ratio for late portion in convergence

    Returns:
        Dictionary containing all alignment metrics
    """
    df = load_pickle_file(pkl_path)

    # Ensure sorted by turn_id
    df = df.sort_values('turn_id').reset_index(drop=True)

    # Get feature column names
    feature_cols = [col for col in df.columns if col.startswith('trillsson_feature_')]

    if not feature_cols:
        warnings.warn(f"No feature columns found in {pkl_path}")
        return None

    # Calculate all metrics
    results = {
        'file': os.path.basename(pkl_path).replace('_trillsson_features.pkl', ''),
        'total_turns': len(df),
        'n_speakers': df['speaker'].nunique(),
        'speakers': ', '.join(df['speaker'].unique())
    }

    # Approach 1: Local Similarity
    local_results = calculate_local_similarity(df, feature_cols)
    results.update(local_results)

    # NEW: Binned Local Similarity (temporal patterns)
    binned_results = calculate_local_similarity_by_bins(df, feature_cols, n_bins=5)
    results.update(binned_results)

    # NEW: Asymmetric Adaptation (teacher-student dynamics)
    asymmetric_results = calculate_asymmetric_adaptation(df, feature_cols)
    results.update(asymmetric_results)

    # NEW: Weighted Local Similarity (filter backchannels)
    weighted_results = calculate_weighted_local_similarity(df, feature_cols, weight_by='length_seconds')
    results.update(weighted_results)

    # Approach 2: Convergence
    convergence_results = calculate_convergence(df, feature_cols, early_ratio, late_ratio)
    results.update(convergence_results)

    # Approach 3a: Proximity (static similarity)
    proximity_results = calculate_proximity(df, feature_cols)
    results.update(proximity_results)

    # Approach 3b: Synchrony (dynamic co-variation)
    synchrony_results = calculate_synchrony(df, feature_cols)
    results.update(synchrony_results)

    return results


def main():
    """Main function to process all pickle files and output advanced alignment metrics."""
    # Input and output paths
    input_dir = Path('trillsson_features_output')
    output_file = 'alignment_metrics_advanced.csv'

    # Configuration
    EARLY_RATIO = 0.3
    LATE_RATIO = 0.3

    # Get all pickle files
    pkl_files = list(input_dir.glob('*.pkl'))

    if not pkl_files:
        print(f"No pickle files found in {input_dir}")
        return

    print(f"Found {len(pkl_files)} pickle files\n")

    # Show sample from first file
    print("Sample from first file:")
    df_sample = load_pickle_file(str(pkl_files[0]))
    print(df_sample.head(6))
    print()

    # Process all files
    all_results = []
    for pkl_file in pkl_files:
        print(f"Processing {pkl_file.name}...")
        try:
            file_results = process_file(
                str(pkl_file),
                early_ratio=EARLY_RATIO,
                late_ratio=LATE_RATIO
            )

            if file_results is not None:
                all_results.append(file_results)
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)

    print(f"\nTotal sessions processed: {len(all_results)}")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
