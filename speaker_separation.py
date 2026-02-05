from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

def get_file_prefixes(filename: str) -> str:
    """Get common file prefixes across the three directories."""
    prefix = filename.stem.split('_combined')[0]
    # print("working on file prefix:", prefix)
    return prefix

def get_validated_inputs(args):
    """Combines 1 & 2: Validates folders and matches filename prefixes."""

    for d in [args.audio_dir, args.feature_dir, args.transcript_dir]:
        if not Path(d).is_dir():
            print(f"Error: Directory not found: {d}")
            return None, None, None
    
    # Map prefixes to filenames
    audio_dir = Path(args.audio_dir)
    audio_files = {get_file_prefixes(p): p.name for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.wav', '.mp3')}
    print("audio_files:", [f for f in audio_files.values()])
    
    transcript_dir = Path(args.transcript_dir)
    transcript_files = {get_file_prefixes(p): p.name for p in transcript_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.csv', '.txt')}
    
    feature_dir = Path(args.feature_dir)
    feature_files = {get_file_prefixes(p): p.name for p in feature_dir.iterdir() if p.is_file() and p.suffix.lower() == '.csv'}

    if not audio_files or not transcript_files or not feature_files:
        print("Error: no files found for the file prefixes.")
        return None, None, None
    else:
        # Find intersection of all three
        common_prefixes = sorted(list(set(audio_files.keys()) & set(transcript_files.keys()) & set(feature_files.keys())))

        if not common_prefixes:
            print("Error: No matching filename prefixes found across the three folders.")
            return None, None, None

        print(f"Successfully matched {len(common_prefixes)} file pairs.")
        return common_prefixes, audio_files, transcript_files, feature_files

def run_bimodal_clustering(prefix, trans_path, feat_path, args):
    """Combines 3, 4, 5: Clusters features globally across the file and assigns labels."""
    
    df_trans = pd.read_csv(trans_path)
    df_feat = pd.read_csv(feat_path)

    # 1. Parse multiple columns (e.g., "f0_median, f0_std")
    # Robust parsing: handles lists from argparse OR a single comma-separated string
    
    if isinstance(args.feature_col, list):
        print(f"{len(args.feature_col)} features provided: {args.feature_col}")
        # Join into one string and split to handle ['f0_median,', 'f0_std']
        raw_string = ",".join(args.feature_col)
        feature_list = [c.strip() for c in raw_string.split(',') if c.strip()]
    else:
        print("ERROR: --feature_col should be a list of column names.")

    # Validation Check
    missing = [c for c in feature_list if c not in df_feat.columns]
    if missing:
        print(f"Error: Could not find columns: {missing}")
        print(f"Available columns in file: {list(df_feat.columns)[:5]}...") # Show sample
        return [], "Error"
    else:
        print(f"All specified features found.")
    
    # Validation: Ensure the number of rows match
    if len(df_trans) != len(df_feat):
        print(f"Warning: Row mismatch for {prefix}. Transcript: {len(df_trans)}, Features: {len(df_feat)}")
    
    # 2. Extract feature matrix (Rows x N_features)
    # GMM handles multiple dimensions automatically
    # print(f"Type of columns names: {df_feat[args.feature_col].dtypes}")
    features = df_feat[args.feature_col].values
    
    # 3. Fit a 1-speaker model vs a 2-speaker model
    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(features)
    gmm2 = GaussianMixture(n_components=2, random_state=42).fit(features)
    
    # 4. Extract and Print Means
    # .means_ returns an array of arrays, so we flatten it
    cluster_centers = gmm2.means_ # Shape: (2, N_features)

    print(f"\n--- Bimodal Analysis: {prefix} ---")
    for idx, col in enumerate(args.feature_col):
        c1_val = cluster_centers[0][idx]
        c2_val = cluster_centers[1][idx]
        print(f"  {col:15} | Center 1: {c1_val:>8.4f} | Center 2: {c2_val:>8.4f}")


    # 5. Check Bimodality
    # AIC (Akaike Information Criterion)
    if gmm1.aic(features) < gmm2.aic(features):
        print(f"[!] Warning for {prefix}: Data appears UNIMODAL.")
        print(f"The features in '{args.feature_col}' does not clearly separate into two speakers.")
        bimodality_status = "unimodal"
    else:
        print(f"{prefix}: Confirmed bimodal distribution.")
        bimodality_status = "bimodal"

    # 5. Assign Labels
    # Use the FIRST feature in your list (e.g., median) to decide who is "Low"
    predicted_labels = gmm2.predict(features)
    low_pitch_cluster = np.argmin(cluster_centers[:, 0])
    
    # Predict labels for every row in the feature file at once
    predicted_labels = gmm2.predict(features)

    results = []
    for i in range(len(df_trans)):
        # Extract metadata from transcript row
        row_trans = df_trans.iloc[i]
        row_feat = df_feat.iloc[i]
        # Identify as Speaker_Low or Speaker_High based on the cluster mean
        raw_label = predicted_labels[i]
        gen_label = "Speaker_Low" if raw_label == low_pitch_cluster else "Speaker_High"
        entry = {
            "file_prefix": prefix,
            "generated_speaker_label": gen_label,
            "original_speaker_label": row_trans[args.speaker_col] if args.speaker_col in df_trans.columns else "N/A",
            "start_time": row_trans[args.start_col],
            "end_time": row_trans[args.end_col],
            "transcript": row_trans[args.text_col]
        }
        for feat in args.feature_col:
            entry[feat] = row_feat[feat]
        
        results.append(entry)
    
    return results, bimodality_status

def save_individual_report(prefix, data, output_dir):
    """6: Saves a single CSV file for the specific file prefix."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
        
    df = pd.DataFrame(data)
    save_path = output_path / f"{prefix}.csv"
    df.to_csv(save_path, index=False)
    print(f"Generated: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Per-file Bimodal Speaker Labelling")
    parser.add_argument("--audio_dir", default="../audio", help="Directory containing audio files")
    parser.add_argument("--feature_dir", default="../audio/feats_transcript_combined", help="Directory containing feature files")
    parser.add_argument("--transcript_dir", default="../audio/feats_transcript_combined", help="Directory containing transcript files")
    parser.add_argument("--output_dir", default="./speaker_separation", help="Directory containing output files")
    parser.add_argument("--start_col", default="start_time", help="Column name for start time")
    parser.add_argument("--end_col", default="end_time", help="Column name for end time")
    parser.add_argument("--text_col", default="text", help="Column name for transcript text")
    parser.add_argument("--feature_col", default=["F0final_sma_median", "F0final_sma_std"], nargs="+", help="Column name(s) for feature values")
    parser.add_argument("--speaker_col", default="speaker", help="Column name for speaker labels in transcript files")
    
    args = parser.parse_args()


    # Step 1 & 2
    prefixes, audio_map, trans_map, feat_map = get_validated_inputs(args)
    if not prefixes:
        return

    # Step 3, 4, 5, 6

    for prefix in prefixes:
        t_path = Path(args.transcript_dir) / trans_map[prefix]
        f_path = Path(args.feature_dir) / feat_map[prefix]
        
        # Process and Cluster
        file_results, bimod_status = run_bimodal_clustering(prefix, t_path, f_path, args)
        
        # Save per-pair file
        # feature_names = [c.strip() for c in args.feature_col.split(',')]
        feature_suffix = "_".join(args.feature_col)
        output_dir_name = f"{args.output_dir}_{feature_suffix}/{bimod_status}"
        save_individual_report(prefix, file_results, output_dir_name)

if __name__ == "__main__":
    main()