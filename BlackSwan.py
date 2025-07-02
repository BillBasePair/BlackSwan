# ===== PART 1 - Imports and Utilities ===== #

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from collections import defaultdict, Counter
import os
import time
import random
import io
import RNA
from Levenshtein import distance
import sklearn
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import warnings
import tkinter as tk
from tkinter import filedialog
warnings.filterwarnings('ignore')

# Initialize session state
if "files" not in st.session_state:
    st.session_state["files"] = []
    st.session_state["round_map"] = {}
    st.session_state["params"] = {
        "black_swan_seed": 100,
        "artifact_threshold": 0.80,
        "temperature": 25.0,
        "dangling_ends": 2
    }
    st.session_state["run_log"] = []
    st.session_state["output_dir"] = None
    st.session_state["full_seq_map"] = {}
    st.session_state["fold_cache"] = {}
    st.session_state["repetitive_pattern_logs"] = 0
    st.session_state["skip_pattern_logs"] = {}

log_buffer = []

def preprocess_sequence(seq, seq_type):
    if seq_type == "DNA":
        seq = seq.replace("T", "U")
    return seq

def has_repetitive_pattern(seq, max_repeat_length=4):
    for length in range(2, min(max_repeat_length + 1, len(seq) // 4 + 1)):
        for i in range(len(seq) - length + 1):
            pattern = seq[i:i+length]
            if len(pattern) * 4 > len(seq):
                continue
            if pattern * 4 in seq:
                return True
    return False

def log_to_file(message, output_dir, timestamp):
    log_buffer.append(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {message}\n")
    if len(log_buffer) >= 100 or "Analysis completed successfully" in message:
        with open(os.path.join(output_dir, f"run_log_{timestamp}.txt"), "a", encoding="utf-8") as f:
            f.writelines(log_buffer)
        log_buffer.clear()

def compute_base_pair_distance(seq, structure):
    stack = []
    pairs = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    distances = [abs(j - i) for j, i in pairs]
    return np.mean(distances) if distances else 0.0

def detect_primers(sequences):
    sample_size = min(1000, len(sequences))
    sampled_seqs = random.sample(sequences, sample_size) if sample_size < len(sequences) else sequences
    forward_candidates = []
    reverse_candidates = []
    
    for seq in sampled_seqs:
        if len(seq) >= 43:
            forward_candidates.append(seq[:19])
            reverse_candidates.append(seq[-19:])
    
    def find_consensus(candidates, expected_len=19):
        if not candidates:
            return None, 0
        counter = Counter(candidates)
        consensus = counter.most_common(1)[0][0]
        matches = sum(1 for c in candidates if distance(c, consensus) <= len(consensus) * 0.6)
        return consensus, matches / len(candidates)
    
    forward_primer, forward_match_rate = find_consensus(forward_candidates)
    reverse_primer, reverse_match_rate = find_consensus(reverse_candidates)
    
    if forward_primer is None or reverse_primer is None or forward_match_rate < 0.2 or reverse_match_rate < 0.2:
        st.session_state["run_log"].append("Primer detection failed, using defaults: Forward=GCGCCGGAGTTCTCAATGC, Reverse=GCATGCCGGTCGGTCTACT")
        return "GCGCCGGAGTTCTCAATGC", "GCATGCCGGTCGGTCTACT", 19, 19, 32
    
    random_region_len = 32
    return forward_primer, reverse_primer, len(forward_primer), len(reverse_primer), random_region_len

def trim_sequence(full_seq, primer_fwd_seq, primer_rev_seq, primer_fwd_len=19, primer_rev_len=19, artifact_threshold=0.7, trimming_stats=None):
    if len(full_seq) < 43:
        return None, f"Sequence too short: {len(full_seq)} (expected >= 43)"
    min_dist = float('inf')
    best_start = 0
    for i in range(0, 5):
        if i + len(primer_fwd_seq) <= len(full_seq):
            dist = distance(full_seq[i:i+len(primer_fwd_seq)], primer_fwd_seq)
            if dist < min_dist:
                min_dist = dist
                best_start = i
    if min_dist > len(primer_fwd_seq) * 0.6:
        if trimming_stats is not None and trimming_stats["Forward primer mismatch"] < 50:
            st.session_state["run_log"].append(f"Forward primer mismatch: dist={min_dist}, seq={full_seq[:len(primer_fwd_seq)]}, primer={primer_fwd_seq}")
        return None, "Forward primer mismatch"
    min_dist = float('inf')
    best_end = len(full_seq)
    for i in range(0, 5):
        start = len(full_seq) - len(primer_rev_seq) - i
        if start >= 0:
            dist = distance(full_seq[start:start+len(primer_rev_seq)], primer_rev_seq)
            if dist < min_dist:
                min_dist = dist
                best_end = start + len(primer_rev_seq)
    if min_dist > len(primer_rev_seq) * 0.6:
        return None, "Reverse primer mismatch"
    trimmed = full_seq[best_start:best_end]
    if len(trimmed) < 28 or len(trimmed) > 36:
        if trimming_stats is not None and trimming_stats.get(f"Invalid trimmed length: {len(trimmed)} (expected 28-36)", 0) < 10:
            pass
        fixed_start = 19
        fixed_end = -19
        trimmed = full_seq[fixed_start:fixed_end if fixed_end < 0 else len(full_seq) + fixed_end]
        if len(trimmed) < 28 or len(trimmed) > 36:
            return None, f"Invalid trimmed length: {len(trimmed)} (expected 28-36)"
    if not set(trimmed).issubset("ACGT"):
        return None, "Invalid nucleotides"
    if any(trimmed.count(n) / len(trimmed) > artifact_threshold for n in "ACGT"):
        st.session_state["run_log"].append(f"Warning: High nucleotide composition in {trimmed}")
    if has_repetitive_pattern(trimmed):
        return None, "Repetitive pattern detected"
    return trimmed, "Valid"

# ---- END OF SECTION 1 ----

# ===== PART 2 - Main Processing and UI (Analysis Tab) ===== #

st.set_page_config(page_title="Black Swan Detection", layout="wide")

st.markdown("# 🦢 Black Swan Detection for SELEX")
st.markdown("**Identify rare, high-potential aptamer sequences.**")
st.markdown("Upload FASTQ files, assign rounds, and detect Black Swan sequences.")

tabs = st.tabs(["Analysis", "Results"])

with tabs[0]:
    st.header("📁 Upload and Configure Analysis")
    uploaded_files = st.file_uploader(
        "Upload FASTQ Files (select multiple)", type=["fastq", "fq"], accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")
        col1, col2 = st.columns(2)
        with col1:
            seed_percent = st.number_input(
                "Black Swan Seed (% of sequences)", min_value=1.0, max_value=50.0, value=10.0, step=1.0
            )
            artifact_threshold = st.slider(
                "Artifact Filter: Max % of Single Nucleotide", min_value=50, max_value=100, value=80, step=1
            )
            seq_type = st.selectbox("Sequence Type", options=["DNA", "RNA"], index=0)
        with col2:
            temperature = st.number_input(
                "Folding Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0, step=0.1
            )
            dangling_ends = st.selectbox(
                "Dangling Ends", options=["None (0)", "Some (1)", "All (2)"], index=2
            )
            use_default_primers = st.checkbox("Use Default Primers", value=True)

        round_options = ["Unassigned"] + [f"Round {i}" for i in range(1, 21)]
        round_assignments = {}
        for file in uploaded_files:
            sel = st.selectbox(
                f"Round for {file.name}", options=round_options, key=f"round_{file.name}"
            )
            round_assignments[file.name] = sel

        # Initialize directory history in session state
        if "directory_history" not in st.session_state:
            st.session_state["directory_history"] = [
                r"C:\Users\gwjac\Documents\black_swan_results",
                os.path.expanduser("~"),  # User's home directory
                os.path.expanduser("~/Desktop"),  # Desktop directory
            ]

        st.write("Select or enter the output directory:")
        # Dropdown for directory selection
        selected_dir = st.selectbox(
            "Choose a directory or select 'Custom' to enter a new path",
            options=st.session_state["directory_history"] + ["Custom"],
            index=0,
            key="directory_select"
        )
        # Show text input if 'Custom' is selected
        if selected_dir == "Custom":
            output_dir = st.text_input(
                "Enter custom directory path (e.g., C:\\Users\\gwjac\\Documents\\black_swan_results)",
                value=r"C:\Users\gwjac\Documents\black_swan_results"
            )
        else:
            output_dir = selected_dir

        # Validate the directory
        try:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                st.success(f"Created directory: {output_dir}")
            if not os.access(output_dir, os.W_OK):
                st.error(f"Directory {output_dir} is not writable. Please choose a directory where you have write permissions.")
                st.session_state["output_dir"] = None
                st.stop()
            # Add valid directory to history if not already present
            if output_dir not in st.session_state["directory_history"]:
                st.session_state["directory_history"].insert(0, output_dir)
                # Limit history to 5 entries
                st.session_state["directory_history"] = st.session_state["directory_history"][:5]
        except Exception as e:
            st.error(f"Invalid directory path: {output_dir}. Error: {str(e)}. Please select or enter a valid writable directory.")
            st.session_state["output_dir"] = None
            st.stop()
        
        st.session_state["output_dir"] = output_dir
        st.write(f"Selected Output Directory: {output_dir}")

        if st.button("🚀 Run Analysis", key="run_analysis_button"):
            if "Unassigned" in round_assignments.values():
                st.error("Please assign SELEX rounds to all files.")
            else:
                # Initialize timestamped log
                timestamp = time.strftime("%y%m%d_%H%M")
                st.session_state["run_log"].append(f"Output directory: {st.session_state['output_dir']}")
                log_to_file(f"Output directory: {st.session_state['output_dir']}", st.session_state["output_dir"], timestamp)
                st.session_state["repetitive_pattern_logs"] = 0
                st.session_state["skip_pattern_logs"] = {f.name: 0 for f in uploaded_files}

                # Update parameters
                st.session_state["params"].update({
                    "seq_type": seq_type,
                    "temperature": temperature,
                    "dangling_ends": int(dangling_ends.split("(")[1].split(")")[0]),
                    "seed_percent": seed_percent,
                    "artifact_threshold": artifact_threshold / 100.0
                })

                # Preprocess FASTQ files
                all_sequences = []
                with st.spinner("🧬 Preprocessing FASTQ files..."):
                    from itertools import islice
                    chunk_size = 10000
                    for f in uploaded_files:
                        try:
                            text = f.getvalue().decode("utf-8", errors="ignore")
                            parser = SeqIO.parse(io.StringIO(text), "fastq")
                            preprocess_stats = Counter()  # Track stats for this file
                            while True:
                                chunk = list(islice(parser, chunk_size))
                                if not chunk:
                                    break
                                for r in chunk:
                                    try:
                                        full_seq = str(r.seq).upper()
                                        qual_scores = r.letter_annotations.get("phred_quality", [])
                                        if qual_scores and any(q < 20 for q in qual_scores[:19]):
                                            preprocess_stats[f"Skipped low-quality forward primer in {f.name}"] += 1
                                            continue
                                        adapters = ["CTGTCTCTA", "AGATCGGAAGAG"]
                                        for adapter in adapters:
                                            if len(full_seq) > 70 and full_seq.endswith(adapter):
                                                full_seq = full_seq[:-len(adapter)]
                                                preprocess_stats[f"Removed adapter {adapter} in {f.name}"] += 1
                                                break
                                        if len(full_seq) > 70:
                                            preprocess_stats[f"Truncated sequence to 70 nt in {f.name}"] += 1
                                            full_seq = full_seq[:70]
                                        if 60 <= len(full_seq) <= 80:
                                            all_sequences.append(full_seq)
                                    except UnicodeDecodeError as e:
                                        st.session_state["run_log"].append(f"Skipping sequence in {f.name} due to decode error: {str(e)}")
                                        log_to_file(f"Skipping sequence in {f.name} due to decode error: {str(e)}", st.session_state["output_dir"], timestamp)
                                        continue
                            # Log summary for this file
                            if preprocess_stats:
                                summary = f"Preprocessing summary for {f.name}: {dict(preprocess_stats)}"
                                st.session_state["run_log"].append(summary)
                                log_to_file(summary, st.session_state["output_dir"], timestamp)
                        except UnicodeDecodeError as e:
                            st.session_state["run_log"].append(f"Error decoding {f.name}: {str(e)}")
                            log_to_file(f"Error decoding {f.name}: {str(e)}", st.session_state["output_dir"], timestamp)
                            st.error(f"Invalid FASTQ file {f.name}: {str(e)}")
                            continue
                        except Exception as e:
                            st.session_state["run_log"].append(f"Error reading {f.name}: {str(e)}")
                            log_to_file(f"Error reading {f.name}: {str(e)}", st.session_state["output_dir"], timestamp)
                            st.error(f"Invalid FASTQ file {f.name}: {str(e)}")
                            continue

                if not all_sequences:
                    st.error("No valid sequences found. Check FASTQ file lengths (expected ~70 nt).")
                    st.session_state["run_log"].append("No valid sequences found")
                    log_to_file("No valid sequences found", st.session_state["output_dir"], timestamp)
                    st.stop()

                # Detect primers
                with st.spinner("🔍 Detecting primers..."):
                    if use_default_primers:
                        st.session_state["primer_fwd_seq"] = "GCGCCGGAGTTCTCAATGC"
                        st.session_state["primer_rev_seq"] = "GCATGCCGGTCGGTCTACT"
                        primer_fwd_len, primer_rev_len, random_region_len = 19, 19, 32
                        st.session_state["run_log"].append("Using default primers: Forward=GCGCCGGAGTTCTCAATGC, Reverse=GCATGCCGGTCGGTCTACT")
                        log_to_file("Using default primers: Forward=GCGCCGGAGTTCTCAATGC, Reverse=GCATGCCGGTCGGTCTACT", st.session_state["output_dir"], timestamp)
                    else:
                        forward_primer, reverse_primer, primer_fwd_len, primer_rev_len, random_region_len = detect_primers(all_sequences)
                        st.session_state["primer_fwd_seq"] = forward_primer
                        st.session_state["primer_rev_seq"] = reverse_primer
                        st.session_state["run_log"].append(f"Detected primers: Forward={forward_primer}, Reverse={reverse_primer}, Lengths={primer_fwd_len}+{random_region_len}+{primer_rev_len}")
                        log_to_file(f"Detected primers: Forward={forward_primer}, Reverse={reverse_primer}, Lengths={primer_fwd_len}+{random_region_len}+{primer_rev_len}", st.session_state["output_dir"], timestamp)

                # Process FASTQ files
                valid_files = []
                round_data = defaultdict(Counter)
                seen_sequences = set()
                trimming_stats = Counter()
                with st.spinner("🧬 Processing FASTQ files..."):
                    for f in uploaded_files:
                        try:
                            text = f.getvalue().decode("utf-8", errors="ignore")
                            parser = SeqIO.parse(io.StringIO(text), "fastq")
                            if next(parser, None) is None:
                                st.warning(f"{f.name} appears empty or invalid.")
                                continue
                            valid_files.append(f)
                            round_label = round_assignments[f.name]
                            parser = SeqIO.parse(io.StringIO(text), "fastq")
                            for r in parser:
                                try:
                                    full_seq = str(r.seq).upper()
                                    qual_scores = r.letter_annotations.get("phred_quality", [])
                                    if qual_scores and any(q < 20 for q in qual_scores[:19]):
                                        trimming_stats[f"Skipped low-quality forward primer in {f.name}"] += 1
                                        continue
                                    adapters = ["CTGTCTCTA", "AGATCGGAAGAG"]
                                    for adapter in adapters:
                                        if len(full_seq) > 70 and full_seq.endswith(adapter):
                                            full_seq = full_seq[:-len(adapter)]
                                            trimming_stats[f"Removed adapter {adapter} in {f.name}"] += 1
                                            break
                                    if len(full_seq) > 70:
                                        trimming_stats[f"Truncated sequence to 70 nt in {f.name}"] += 1
                                        full_seq = full_seq[:70]
                                    trimmed, reason = trim_sequence(
                                        full_seq,
                                        st.session_state["primer_fwd_seq"],
                                        st.session_state["primer_rev_seq"],
                                        primer_fwd_len,
                                        primer_rev_len,
                                        st.session_state["params"]["artifact_threshold"],
                                        trimming_stats
                                    )
                                    trimming_stats[reason] += 1
                                    if trimmed is None:
                                        continue
                                    seq_key = (round_label, trimmed)
                                    if seq_key not in seen_sequences:
                                        seen_sequences.add(seq_key)
                                        round_data[round_label][trimmed] += 1
                                        st.session_state["full_seq_map"][seq_key] = full_seq
                                except UnicodeDecodeError as e:
                                    st.session_state["run_log"].append(f"Skipping sequence in {f.name} due to decode error: {str(e)}")
                                    log_to_file(f"Skipping sequence in {f.name} due to decode error: {str(e)}", st.session_state["output_dir"], timestamp)
                                    continue
                        except UnicodeDecodeError as e:
                            st.session_state["run_log"].append(f"Error decoding {f.name}: {str(e)}")
                            log_to_file(f"Error decoding {f.name}: {str(e)}", st.session_state["output_dir"], timestamp)
                            st.error(f"Invalid FASTQ file {f.name}: {str(e)}")
                            continue
                        except Exception as e:
                            st.session_state["run_log"].append(f"Error processing {f.name}: {str(e)}")
                            log_to_file(f"Error processing {f.name}: {str(e)}", st.session_state["output_dir"], timestamp)
                            st.error(f"Invalid FASTQ file {f.name}: {str(e)}")
                            continue

                st.session_state["run_log"].append(f"Trimming stats: {dict(trimming_stats)}")
                log_to_file(f"Trimming stats: {dict(trimming_stats)}", st.session_state["output_dir"], timestamp)

                if not valid_files:
                    st.error("No valid FASTQ files processed. Check file format or content.")
                    st.session_state["run_log"].append("No valid FASTQ files processed")
                    log_to_file("No valid FASTQ files processed", st.session_state["output_dir"], timestamp)
                    st.stop()

                # Select final round
                try:
                    final_round = max(
                        round_data.keys(),
                        key=lambda x: int(x.split()[-1]) if x != "Unassigned" else 0
                    )
                    st.session_state["run_log"].append(f"Final round: {final_round}")
                    log_to_file(f"Final round: {final_round}", st.session_state["output_dir"], timestamp)
                except ValueError as e:
                    st.error(f"Error selecting final round: {str(e)}. Assign valid round numbers.")
                    st.session_state["run_log"].append(f"Round selection error: {str(e)}")
                    log_to_file(f"Round selection error: {str(e)}", st.session_state["output_dir"], timestamp)
                    st.stop()

                # Compute rarity scores
                with st.spinner("🧮 Computing rarity scores..."):
                    rarity_scores = []
                    prior_rounds = [r for r in round_data.keys() if r != final_round and r != "Unassigned"]
                    for seq, count in round_data[final_round].items():
                        freq_final = count / sum(round_data[final_round].values())
                        if not prior_rounds:
                            temporal_rarity = np.log1p(1 / (freq_final + 1e-10))
                            st.session_state["run_log"].append("Single round detected, using log-scaled frequency")
                            log_to_file("Single round detected, using log-scaled frequency", st.session_state["output_dir"], timestamp)
                        else:
                            freq_prior = sum(round_data[r].get(seq, 0) for r in prior_rounds) / sum(sum(round_data[r].values()) for r in prior_rounds)
                            temporal_rarity = np.log1p(1 / (freq_final + 1e-10)) if freq_prior == 0 else np.log1p(1 / (freq_final / (freq_prior + 1e-10)))
                        rarity_scores.append((seq, count, temporal_rarity))
                    rarity_scores.sort(key=lambda x: x[2], reverse=True)
                    st.session_state["run_log"].append(f"Computed {len(rarity_scores)} rarity scores")
                    log_to_file(f"Computed {len(rarity_scores)} rarity scores", st.session_state["output_dir"], timestamp)

                # Select top sequences
                black_swan_seed = min(200, max(10, int(len(rarity_scores) * (seed_percent / 100.0))))
                top_seqs = rarity_scores[:black_swan_seed]
                st.session_state["run_log"].append(f"Selected {len(top_seqs)} sequences for Black Swan detection")
                log_to_file(f"Selected {len(top_seqs)} sequences for Black Swan detection", st.session_state["output_dir"], timestamp)

                # Extract features
                features = []
                progress_bar = st.progress(0)
                with st.spinner("🔬 Extracting features..."):
                    def extract_features(args):
                        seq, count, rarity, full_seq, params, final_round = args
                        try:
                            RNA.cvar.temperature = params["temperature"]
                            RNA.cvar.dangles = params["dangling_ends"]
                            RNA.cvar.noGU = 0
                            cache_key = (full_seq, params["temperature"], params["dangling_ends"])
                            if cache_key not in st.session_state["fold_cache"]:
                                structure, mfe = RNA.fold(preprocess_sequence(full_seq, params["seq_type"]))
                                fc = RNA.fold_compound(preprocess_sequence(full_seq, params["seq_type"]))
                                _, ensemble_energy = fc.pf()
                                st.session_state["fold_cache"][cache_key] = (structure, mfe, ensemble_energy)
                            else:
                                structure, mfe, ensemble_energy = st.session_state["fold_cache"][cache_key]
                            energy_path_score = abs(mfe - ensemble_energy) / (len(full_seq) + 1e-6)
                            if np.isnan(energy_path_score) or np.isinf(energy_path_score):
                                energy_path_score = 0.0
                            base_pair_dist = compute_base_pair_distance(full_seq, structure)
                            loops = structure.split(")")
                            loop_lengths = [len(l) for l in loops if l and all(c == "." for c in l)]
                            loop_len_mean = np.mean(loop_lengths) if loop_lengths else 0.0
                            loop_len_var = np.var(loop_lengths) if loop_lengths else 0.0
                            loop_len_max = max(loop_lengths) if loop_lengths else 0.0
                            stems = structure.split(".")
                            stem_lengths = [len(s) for s in stems if s and all(c in "()" for c in s)]
                            stem_loop_count = len([s for s in stems if s and all(c in "()" for c in s)])
                            paired_bases = sum(1 for c in structure if c in "()")
                            paired_fraction = paired_bases / len(structure) if len(structure) > 0 else 0.0
                            accessibility = sum(1 for c in structure if c == ".") / len(structure) if len(structure) > 0 else 0.5
                            structural_divergence = distance(structure, "." * len(structure)) / len(structure)
                            binding_pocket_potential = sum(1 for i in range(len(structure)-4) if structure[i:i+5].count(".") >= 3) / len(structure)
                            return {
                                "Sequence": seq,
                                "Frequency": count,
                                "Temporal Rarity": rarity,
                                "MFE": mfe,
                                "Energy Path Score": energy_path_score,
                                "Base Pair Distance": base_pair_dist,
                                "Loop Length Mean": loop_len_mean,
                                "Loop Length Variance": loop_len_var,
                                "Loop Length Max": loop_len_max,
                                "Stem-loop Count": stem_loop_count,
                                "Paired Bases Fraction": paired_fraction,
                                "Accessibility": accessibility,
                                "Structural Divergence": structural_divergence,
                                "Binding Pocket Potential": binding_pocket_potential
                            }
                        except Exception as e:
                            st.session_state["run_log"].append(f"Error extracting features for {seq[:10]}...: {str(e)}")
                            log_to_file(f"Error extracting features for {seq[:10]}...: {str(e)}", st.session_state["output_dir"], timestamp)
                            return None

                    for i, (seq, count, rarity) in enumerate(top_seqs):
                        full_seq = st.session_state["full_seq_map"].get((final_round, seq), seq)
                        feature = extract_features((seq, count, rarity, full_seq, st.session_state["params"], final_round))
                        if feature is not None:
                            features.append(feature)
                        progress_bar.progress(min(1.0, (i + 1) / len(top_seqs)))

                if not features:
                    st.error("No features extracted. Check sequence data or folding parameters.")
                    st.session_state["run_log"].append("No features extracted")
                    log_to_file("No features extracted", st.session_state["output_dir"], timestamp)
                    st.stop()

                # Create feature DataFrame
                df_features = pd.DataFrame(features)
                st.session_state["run_log"].append(f"Feature DataFrame created with {len(df_features)} entries")
                log_to_file(f"Feature DataFrame created with {len(df_features)} entries", st.session_state["output_dir"], timestamp)

                # Filter low-variance features
                feature_columns = [
                    "Temporal Rarity", "MFE", "Energy Path Score", "Base Pair Distance",
                    "Loop Length Mean", "Loop Length Variance", "Loop Length Max",
                    "Stem-loop Count", "Paired Bases Fraction", "Accessibility",
                    "Structural Divergence", "Binding Pocket Potential"
                ]
                variances = df_features[feature_columns].var()
                selected_features = [col for col in feature_columns if variances[col] > 1e-10]
                if not selected_features:
                    st.error("No features with sufficient variance. Adjust data or parameters.")
                    st.session_state["run_log"].append("No features with sufficient variance")
                    log_to_file("No features with sufficient variance", st.session_state["output_dir"], timestamp)
                    st.stop()

                # Save Black Swan features
                features_csv_path = os.path.join(st.session_state["output_dir"], "black_swan_features.csv")
                features_excel_path = os.path.join(st.session_state["output_dir"], f"black_swan_features_TAMU_lamotrigine_BlackSwanDetection_v18_{timestamp}.xlsx")
                df_features.to_csv(features_csv_path, index=False)
                df_features.to_excel(features_excel_path, index=False)
                st.session_state["run_log"].append(f"Black Swan features saved to {features_csv_path} and {features_excel_path}")
                log_to_file(f"Black Swan features saved to {features_csv_path} and {features_excel_path}", st.session_state["output_dir"], timestamp)

                # Detect Black Swans
                with st.spinner("🦢 Detecting Black Swans..."):
                    normalized_features = (df_features[selected_features] - df_features[selected_features].mean()) / df_features[selected_features].std()
                    contamination = min(0.5, max(0.2, 20 / len(normalized_features)))
                    clf = IsolationForest(contamination=contamination, random_state=42)
                    predictions = clf.fit_predict(normalized_features)
                    df_features["Black Swan"] = predictions == -1
                    black_swans = df_features[df_features["Black Swan"]]
                    st.session_state["run_log"].append(f"Detected {len(black_swans)} Black Swans")
                    log_to_file(f"Detected {len(black_swans)} Black Swans", st.session_state["output_dir"], timestamp)

                    # Save Black Swan sequences
                    fasta_content = ""
                    for idx, row in black_swans.iterrows():
                        full_seq = st.session_state["full_seq_map"].get((final_round, row["Sequence"]), row["Sequence"])
                        fasta_content += f">BlackSwan_{idx+1}_Rarity_{row['Temporal Rarity']:.2f}\n{full_seq}\n"
                    fasta_path = os.path.join(st.session_state["output_dir"], "black_swan_candidates.fasta")
                    with open(fasta_path, "w") as f:
                        f.write(fasta_content)
                    st.session_state["run_log"].append(f"Black Swan FASTA saved to {fasta_path}")
                    log_to_file(f"Black Swan FASTA saved to {fasta_path}", st.session_state["output_dir"], timestamp)

                # t-SNE visualization (3D)
                with st.spinner("📊 Generating 3D t-SNE visualization..."):
                    perplexity = min(30, max(5, len(normalized_features) // 2))
                    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity, n_iter=1000)
                    tsne_results = tsne.fit_transform(normalized_features)
                    st.session_state["run_log"].append(f"3D t-SNE completed with {len(normalized_features)} samples, perplexity={perplexity}, KL divergence={tsne.kl_divergence_}")
                    log_to_file(f"3D t-SNE completed with {len(normalized_features)} samples, perplexity={perplexity}, KL divergence={tsne.kl_divergence_}", st.session_state["output_dir"], timestamp)
                    # Static 3D t-SNE plot
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(tsne_results[~df_features["Black Swan"], 0], tsne_results[~df_features["Black Swan"], 1], tsne_results[~df_features["Black Swan"], 2], 
                               c="blue", label="False", alpha=0.6)
                    ax.scatter(tsne_results[df_features["Black Swan"], 0], tsne_results[df_features["Black Swan"], 1], tsne_results[df_features["Black Swan"], 2], 
                               c="red", label="True", alpha=0.6)
                    ax.set_xlabel("t-SNE Component 1")
                    ax.set_ylabel("t-SNE Component 2")
                    ax.set_zlabel("t-SNE Component 3")
                    ax.set_title("3D t-SNE Visualization of Black Swan Candidates")
                    ax.legend(title="Black Swan")
                    tsne_path = os.path.join(st.session_state["output_dir"], f"tsne_plot_3d_{timestamp}.png")
                    plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    st.session_state["run_log"].append(f"3D t-SNE plot saved to {tsne_path}")
                    log_to_file(f"3D t-SNE plot saved to {tsne_path}", st.session_state["output_dir"], timestamp)

                st.success("Analysis complete! Check the Results tab for Black Swan candidates.")
                st.session_state["run_log"].append("Analysis completed successfully")
                log_to_file("Analysis completed successfully", st.session_state["output_dir"], timestamp)

# ===== END PART 2 ===== #

# ===== PART 3 - Results Tab (Visualizations and Downloads) ===== #

with tabs[1]:
    st.header("📊 Results")
    if "run_log" in st.session_state and st.session_state["run_log"]:
        st.subheader("Run Log")
        for log in st.session_state["run_log"]:
            st.text(log)
        log_path = os.path.join(st.session_state["output_dir"], f"run_log_{timestamp}.txt")
        with open(log_path, "r", encoding="utf-8") as f:
            st.download_button("⬇ Download Run Log", f, f"run_log_{timestamp}.txt")
        
        if os.path.exists(features_csv_path):
            st.subheader("Black Swan Features")
            st.dataframe(df_features)
            with open(features_csv_path, "rb") as f:
                st.download_button("⬇ Download Features CSV", f, "black_swan_features.csv")
            with open(features_excel_path, "rb") as f:
                st.download_button("⬇ Download Features Excel", f, f"black_swan_features_TAMU_lamotrigine_BlackSwanDetection_v18_{timestamp}.xlsx")
        
        if os.path.exists(fasta_path):
            st.subheader("Black Swan Sequences")
            st.dataframe(black_swans[["Sequence", "Frequency", "Temporal Rarity", "MFE", "Black Swan"]])
            with open(fasta_path, "rb") as f:
                st.download_button("⬇ Download FASTA", f, "black_swan_candidates.fasta")
        
        if os.path.exists(tsne_path):
            st.subheader("3D t-SNE Visualization")
            # Label the most extreme Black Swans
            # Compute distances from origin for Black Swans
            black_swan_indices = df_features.index[df_features["Black Swan"]]
            tsne_distances = np.linalg.norm(tsne_results[black_swan_indices], axis=1)
            # Get indices of the top 3 most extreme Black Swans (in df_features space)
            extreme_indices = black_swan_indices[np.argsort(tsne_distances)[-3:]]
            # Map to Black Swan IDs (1-based, matching FASTA)
            black_swan_ids = {idx: i+1 for i, idx in enumerate(black_swan_indices)}
            # Static 3D t-SNE plot with labels
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne_results[~df_features["Black Swan"], 0], tsne_results[~df_features["Black Swan"], 1], tsne_results[~df_features["Black Swan"], 2], 
                       c="blue", label="False", alpha=0.6)
            ax.scatter(tsne_results[df_features["Black Swan"], 0], tsne_results[df_features["Black Swan"], 1], tsne_results[df_features["Black Swan"], 2], 
                       c="red", label="True", alpha=0.6)
            # Add labels for extreme Black Swans
            for idx in extreme_indices:
                black_swan_id = black_swan_ids[idx]
                ax.text(tsne_results[idx, 0], tsne_results[idx, 1], tsne_results[idx, 2], f"BS{black_swan_id}", 
                        color='black', fontsize=10, weight='bold')
            ax.set_xlabel("t-SNE Component 1")
            ax.set_ylabel("t-SNE Component 2")
            ax.set_zlabel("t-SNE Component 3")
            ax.set_title("3D t-SNE Visualization of Black Swan Candidates")
            ax.legend(title="Black Swan")
            tsne_path = os.path.join(st.session_state["output_dir"], f"tsne_plot_3d_{timestamp}.png")
            plt.savefig(tsne_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            st.session_state["run_log"].append(f"3D t-SNE plot saved to {tsne_path}")
            log_to_file(f"3D t-SNE plot saved to {tsne_path}", st.session_state["output_dir"], timestamp)
            st.image(tsne_path, caption="3D t-SNE Plot of Black Swan Candidates")
            # Interactive 3D t-SNE plot with labels
            tsne_df = pd.DataFrame({
                "t-SNE Component 1": tsne_results[:, 0],
                "t-SNE Component 2": tsne_results[:, 1],
                "t-SNE Component 3": tsne_results[:, 2],
                "Black Swan": df_features["Black Swan"].astype(str),
                "Sequence": df_features["Sequence"],
                "Label": ["BS" + str(black_swan_ids[idx]) if idx in extreme_indices else "" for idx in df_features.index]
            })
            fig_tsne = px.scatter_3d(
                tsne_df, 
                x="t-SNE Component 1", 
                y="t-SNE Component 2", 
                z="t-SNE Component 3",
                color="Black Swan",
                color_discrete_map={"True": "red", "False": "blue"},
                text="Label",
                hover_data=["Sequence"],
                title="Interactive 3D t-SNE Visualization of Black Swan Candidates",
                opacity=0.6
            )
            fig_tsne.update_traces(textposition="top center")
            fig_tsne.update_layout(
                scene=dict(
                    xaxis_title="t-SNE Component 1",
                    yaxis_title="t-SNE Component 2",
                    zaxis_title="t-SNE Component 3"
                )
            )
            st.plotly_chart(fig_tsne, use_container_width=True, height=800)

        # 3D PCA Visualization
        st.subheader("3D PCA Visualization")
        # Select the three features for PCA
        features_for_pca = df_features[['Base Pair Distance', 'Loop Length Max', 'Loop Length Variance']]
        # Standardize the features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_for_pca)
        # Run PCA with 3 components
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(features_scaled)
        # Calculate explained variance ratios
        explained_variance_ratio = pca.explained_variance_ratio_
        st.write(f"Explained variance ratio for PCA Component 1: {explained_variance_ratio[0]:.2f}")
        st.write(f"Explained variance ratio for PCA Component 2: {explained_variance_ratio[1]:.2f}")
        st.write(f"Explained variance ratio for PCA Component 3: {explained_variance_ratio[2]:.2f}")
        st.write(f"Total explained variance: {sum(explained_variance_ratio):.2f}")
        # Label the most extreme Black Swans
        pca_distances = np.linalg.norm(X_pca[black_swan_indices], axis=1)
        extreme_pca_indices = black_swan_indices[np.argsort(pca_distances)[-3:]]
        # Static 3D PCA plot with labels
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[~df_features["Black Swan"], 0], X_pca[~df_features["Black Swan"], 1], X_pca[~df_features["Black Swan"], 2], 
                   c="blue", label="False", alpha=0.6)
        ax.scatter(X_pca[df_features["Black Swan"], 0], X_pca[df_features["Black Swan"], 1], X_pca[df_features["Black Swan"], 2], 
                   c="red", label="True", alpha=0.6)
        # Add labels for extreme Black Swans
        for idx in extreme_pca_indices:
            black_swan_id = black_swan_ids[idx]
            ax.text(X_pca[idx, 0], X_pca[idx, 1], X_pca[idx, 2], f"BS{black_swan_id}", 
                    color='black', fontsize=10, weight='bold')
        ax.set_xlabel(f"PCA Component 1 ({explained_variance_ratio[0]*100:.1f}% variance)")
        ax.set_ylabel(f"PCA Component 2 ({explained_variance_ratio[1]*100:.1f}% variance)")
        ax.set_zlabel(f"PCA Component 3 ({explained_variance_ratio[2]*100:.1f}% variance)")
        ax.set_title("3D PCA Visualization of Black Swan Candidates")
        ax.legend(title="Black Swan")
        pca_plot_path = os.path.join(st.session_state["output_dir"], f"pca_plot_3d_{timestamp}.png")
        plt.savefig(pca_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        st.session_state["run_log"].append(f"3D PCA plot saved to {pca_plot_path}")
        log_to_file(f"3D PCA plot saved to {pca_plot_path}", st.session_state["output_dir"], timestamp)
        st.image(pca_plot_path, caption="3D PCA Plot of Black Swan Candidates")
        # Interactive 3D PCA plot with labels
        pca_df = pd.DataFrame({
            "PCA Component 1": X_pca[:, 0],
            "PCA Component 2": X_pca[:, 1],
            "PCA Component 3": X_pca[:, 2],
            "Black Swan": df_features["Black Swan"].astype(str),
            "Sequence": df_features["Sequence"],
            "Label": ["BS" + str(black_swan_ids[idx]) if idx in extreme_pca_indices else "" for idx in df_features.index]
        })
        fig_plotly = px.scatter_3d(
            pca_df, 
            x="PCA Component 1", 
            y="PCA Component 2", 
            z="PCA Component 3",
            color="Black Swan",
            color_discrete_map={"True": "red", "False": "blue"},
            text="Label",
            hover_data=["Sequence"],
            title="Interactive 3D PCA Visualization of Black Swan Candidates",
            opacity=0.6
        )
        fig_plotly.update_traces(textposition="top center")
        fig_plotly.update_layout(
            scene=dict(
                xaxis_title=f"PCA Component 1 ({explained_variance_ratio[0]*100:.1f}% variance)",
                yaxis_title=f"PCA Component 2 ({explained_variance_ratio[1]*100:.1f}% variance)",
                zaxis_title=f"PCA Component 3 ({explained_variance_ratio[2]*100:.1f}% variance)"
            )
        )
        st.plotly_chart(fig_plotly, use_container_width=True, height=800)

# ---- END OF SECTION 3 and SCRIPT ----