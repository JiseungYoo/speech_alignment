import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


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
        print(f"Cosine similarity calculation failed: {e}")
        return 0.0


def process_file(csv_path: str) -> List[Dict]:
    """
    Process a single CSV file to calculate cosine similarities
    between consecutive turns with different speakers.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of dictionaries containing similarity results
    """
    df = pd.read_csv(csv_path)
    results = []
    count_missing = df['speaker_label'].isna().sum() + (df['speaker_label'].astype(str).str.strip() == '').sum()
    
    # Get feature column names (_feature_000 to _feature_767)
    # feature_cols = [col for col in df.columns if col.startswith('vector_feature_')]
    feature_cols = ["pcm_loudness_sma_amean", 
                        "pcm_loudness_sma_stddev", "F0finEnv_sma_amean", 
                        "F0finEnv_sma_stddev"]

    # Iterate through consecutive turns
    # Count NaN or empty string entries in 'speaker_label'
    print(f"file: {csv_path}, total turns: {len(df)}, empty rows: {count_missing}")
    for i in range(len(df) - 1):
        current_turn = df.iloc[i]
        next_turn = df.iloc[i + 1]


        # Only calculate similarity if speakers are different
        if current_turn["speaker_label"] != next_turn["speaker_label"]:
            # Debug/logging
            print("Speaker change detected")
            print(
                f"Calculating similarity between turn {i} "
                f"speaker {current_turn['speaker_label']} and turn {i+1} "
                f"speaker {next_turn['speaker_label']}"
            )
            # Extract feature vectors
            vec1 = current_turn[feature_cols].values
            vec2 = next_turn[feature_cols].values

            # Calculate cosine similarity
            similarity = calculate_cosine_similarity(vec1, vec2)

            # Store result
            results.append({
                'file': Path(csv_path).name,
                'turn_no_1': i,
                'turn_no_2': i+1,
                'speaker_1': current_turn['speaker_label'],
                'speaker_2': next_turn['speaker_label'],
                'cosine_similarity': similarity
            })
        # if i > 5:
        #     break

    return results




    # Get all CSV files
    csv_files = list(input_dir.glob('*.csv'))

    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    # print(f"Found {len(csv_files)} CSV files")
    # exit()

    # Process all files
    all_results = []
    file_details = []
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        file_results = process_file(str(csv_file))
        mean_cosine_similarity = np.mean([res['cosine_similarity'] for res in file_results]) if file_results else 0.0
        all_results.extend(file_results)
        file_details.append((csv_file.name, len(file_results), mean_cosine_similarity))
        print(f"Found {len(file_results)} speaker-change pairs in file {csv_file.name}, mean cosine similarity: {mean_cosine_similarity:.4f}")

    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)

    for i in file_details:
        print(f"File: {i[0]}, Speaker-change pairs: {i[1]}, Mean Cosine Similarity: {i[2]:.4f}")

    print(f"\nTotal pairs processed: {len(all_results)}")
    print(f"Results saved to: {output_file}")

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Mean cosine similarity: {results_df['cosine_similarity'].mean():.4f}")
    print(f"  Std cosine similarity: {results_df['cosine_similarity'].std():.4f}")
    print(f"  Min cosine similarity: {results_df['cosine_similarity'].min():.4f}")
    print(f"  Max cosine similarity: {results_df['cosine_similarity'].max():.4f}")


if __name__ == '__main__':
    main()
