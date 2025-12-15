import os
import glob
import json
import logging

logger = logging.getLogger(__name__)

def create_jsonl_dataset(audio_directory: str, output_filename: str = "train.jsonl") -> str:
    """
    Scans a directory for .wav files and matching .txt transcripts,
    creating a JSONL manifest file for VoxCPM training.
    
    Args:
        audio_directory: Path to the folder containing audio and text files.
        output_filename: Name of the output file.
        
    Returns:
        Path to the created dataset file.
    """
    if not os.path.isdir(audio_directory):
        raise FileNotFoundError(f"Audio directory not found: {audio_directory}")

    audio_files = glob.glob(os.path.join(audio_directory, "*.wav"))
    if not audio_files:
        raise ValueError(f"No .wav files found in {audio_directory}")

    dataset_path = os.path.join(audio_directory, output_filename)
    valid_samples = 0

    with open(dataset_path, 'w', encoding='utf-8') as f:
        for wav_path in audio_files:
            base_path = os.path.splitext(wav_path)[0]
            txt_path = base_path + ".txt"
            
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as tf:
                        text = tf.read().strip()
                    
                    if text:
                        entry = {
                            "audio": wav_path,
                            "text": text,
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        valid_samples += 1
                except Exception as e:
                    logger.warning(f"Error processing {txt_path}: {e}")
    
    if valid_samples == 0:
        raise RuntimeError(f"No valid samples found in {audio_directory}! Ensure .wav files have matching .txt transcripts.")

    logger.info(f"Created dataset at {dataset_path} with {valid_samples} samples.")
    return dataset_path