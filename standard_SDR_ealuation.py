import torch
import torchaudio
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio


bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
sample_rate = bundle.sample_rate

cpu_device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # windows
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # macOS
model.to(device)

DATASET_FOLDER =  "./musdb18hq/test" # dataset should be inside the project folder
DATASET_FOLDER_TRIMMED = "./musdb18hq_trimmed" # trimmed dataset will be saved here

def load_dataset(path=DATASET_FOLDER_TRIMMED):
    """
        Load the dataset from the specified folder.
        Each subfolder in the dataset corresponds to a song.
        Each song contains multiple stems (e.g., mixture, drums, bass, etc.).
    Args:
        path (str): Path to the dataset folder.
    Returns:
        dataset_dict (dict): A dictionary where keys are track folders and values are dictionaries of stems.
    """
    dataset_dict = {}

    for track_folder in tqdm(os.listdir(path)):
        track_path = os.path.join(path, track_folder)
        if not os.path.isdir(track_path):
            continue

        # Prepare a sub-dictionary for this song
        stems_dict = {}
        
        for stem_name in ["mixture", "drums", "bass", "vocals", "other", "new_mixture"]:
            file_path = os.path.abspath(os.path.join(track_path, f"{stem_name}.wav"))
            
            if not os.path.isfile(file_path):
                print(f"Warning: file not found {file_path}")
                continue

            # Load full audio
            waveform, sr = torchaudio.load(file_path)

            stems_dict[stem_name] = waveform
            
        dataset_dict[track_folder] = stems_dict
        
    return dataset_dict

def separate_sources(
    model,
    mix,
    sample_rate=sample_rate,
    overlap=0.0,  # set to 0.0 to avoid chunk repetition
    device=None,
    normalize=False,
):
    """
    Separate sources from a mixture using the provided model.
    Args:
        model: The separation model.
        mix: The input mixture tensor (batch, channels, length).
        sample_rate: Sample rate of the audio.
        overlap: Overlap between segments in seconds.
        device: Device to run the model on (CPU or GPU).
        normalize: Whether to normalize the input mixture.
    Returns:
        final: The separated sources tensor (batch, sources[drums, bass, other, vocel], channels, length). #CORRECT ORDER UPDATED
    """

    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    model.to(device)
    
    batch, channels, length = mix.shape

    # normalize the input by its RMS
    if normalize:
        # mix = mix / torch.sqrt(torch.mean(mix ** 2, dim=-1, keepdim=True))
        mix = mix / torch.max(torch.abs(mix))

    # chunk_len for entire 30s, no overlap
    chunk_len = int(mix.shape[2] * (1 + overlap))  # effectively 30s if overlap=0
    start = 0
    end = chunk_len

    overlap_frames = int(overlap * sample_rate)
    fade = torchaudio.transforms.Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

    # Prepare final buffer
    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model(chunk).to(device)
            
        out = fade(out)
        final[:, :, :, start:end] += out

        if start == 0:
            fade.fade_in_len = overlap_frames
            start += chunk_len - overlap_frames
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0

    return final

def evaluate_sdr(original_stem: torch.Tensor, predicted_stem: torch.Tensor,
                 device: torch.device=None, sdr_type: str = "scale_invariant") -> float:
    if device is None:
        device = original_stem.device

    original_stem = original_stem.to(device)
    predicted_stem = predicted_stem.to(device)
    
    if sdr_type == "scale_invariant":
        sdr_metric = ScaleInvariantSignalDistortionRatio().to(device)
        return sdr_metric(predicted_stem, original_stem).item()
    
    elif sdr_type == "standard":
        # Sposta su CPU e converti in float64 (lo fa TorchMetrics comunque)
        original_stem_cpu = original_stem.to('cpu')
        predicted_stem_cpu = predicted_stem.to('cpu')

        sdr_metric = SignalDistortionRatio(zero_mean=False)
        return sdr_metric(predicted_stem_cpu, original_stem_cpu).item()

    else:
        raise ValueError("sdr_type must be either 'scale_invariant' or 'standard'.")
    
def compute_sdrs(model, device, stems_dict, separated_sources, sdr_type):
    """Compute SDRs for each stem."""
    sdr_results = {stem: [] for stem in model.sources}
    for i, stem_name in enumerate(model.sources):
        # test normalization
        original_stem = stems_dict[stem_name].to(device)
        predicted_stem = separated_sources[0, i].to(device)
        sdr_value = evaluate_sdr(original_stem, predicted_stem, device=device, sdr_type=sdr_type)
        sdr_results[stem_name].append(sdr_value)

    return sdr_results

def evaluate_sdr_across_dataset(dataset_dict, model, sample_rate, device, normalize, sdr_type, verbose=False):
    """
    Evaluate the SDR for each track in the dataset.
    Args:
        dataset_dict (dict): Dictionary containing the dataset.
        model: The separation model.
        sample_rate: Sample rate of the audio.
        device: Device to run the model on (CPU or GPU).
        normalize: Whether to normalize the input mixture.
        verbose (bool): Whether to print detailed logs.
        sdr_type (str): Type of SDR to compute "standard" or "scale_invariant" (default: "scale_invariant").
    Returns:
        average_sdr (dict): Dictionary containing the average SDR for each stem.
    """
    sdr_results = {stem: [] for stem in ["bass", "drums", "vocals", "other"]}

    sdr_results_list = []

    for track_name, stems_dict in tqdm(dataset_dict.items()):
        if verbose:
            print(f"Processing track: {track_name}")

        # Ensure the mixture exists in the stems
        if "new_mixture" not in stems_dict:
            mixture = stems_dict["mixture"].to(device).unsqueeze(0)  
            if verbose:
                print(f"Skipping track {track_name} as it does not contain a new mixture.")
        else:
            # Load the mixture and move it to the correct device
            mixture = stems_dict["new_mixture"].to(device).unsqueeze(0)  # Add batch dimension
        
        # Perform source separation
        separated_sources = separate_sources(model, mixture, sample_rate=sample_rate, device=device, normalize=normalize)

        # Evaluate SDR for each stem
        sdr_results = compute_sdrs(model, device, stems_dict, separated_sources, sdr_type=sdr_type)

        if verbose:
            print(f"SDR results for track {track_name}:")
            for stem, sdr_value in sdr_results.items():
                print(f"{stem}: {sdr_value[0]:.2f} dB")
        
        # Convert sdr_results to a tensor for this track
        sdr_tensor = torch.tensor([sdr_results[stem][0] for stem in model.sources])
        
        # Append the SDR tensor to the collection
        sdr_results_list.append(sdr_tensor)

    # Calculate the average and standard deviation SDR for each stem
    if sdr_results_list:
        sdr_collection = torch.stack(sdr_results_list)
        average_sdr = {stem: torch.mean(sdr_collection[:, i]).item() for i, stem in enumerate(model.sources)}
        std_sdr = {stem: torch.std(sdr_collection[:, i]).item() for i, stem in enumerate(model.sources)}
    else:
        average_sdr = {stem: None for stem in model.sources}
        std_sdr = {stem: None for stem in model.sources}

    if verbose:
        print("\nAverage SDR for each stem:")
        for stem, avg_sdr in average_sdr.items():
            print(f"{stem}: {avg_sdr:.2f} dB" if avg_sdr is not None else f"{stem}: No data")
        print("\nStandard Deviation of SDR for each stem:")
        for stem, std in std_sdr.items():
            print(f"{stem}: {std:.2f} dB" if std is not None else f"{stem}: No data")

    return sdr_collection

def plot_sdr_results(sdr_collection, model, name):
    """
    Create a box plot for the SDR results for each stem using raw data.

    Args:
        sdr_collection (torch.Tensor): Tensor containing all SDR results [n_samples, n_stems].
        model: Model object containing the sources list.
        name (str): Name of the plot to be displayed.
    """
    stems = model.sources
    
    plt.figure(figsize=(10, 6))

    plt.rcParams.update({'font.size': 16})  # Sets default font size for all elements
    
    # Extract data for each stem
    box_data = [sdr_collection[:, i].numpy() for i in range(len(stems))]
    
    box_plot = plt.boxplot(box_data, labels=stems, patch_artist=True, showfliers=False, orientation="horizontal", notch=True)
    
    # Color the boxes
    colors = ['blue', 'orange', 'green', 'red']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('SDR (dB)')
    plt.ylabel('Stem')
    plt.grid(axis='x', alpha=0.3)
    if name is None:
        plt.title('Box Plot of SDR Results for Each Stem when stem trakc gain is changed')
    else:
        plt.savefig(f"{name}_sdr_results.eps", format='eps', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load the dataset
    dataset_dict = load_dataset(DATASET_FOLDER_TRIMMED)

    # Evaluate SDR across the dataset
    sdr_collection = evaluate_sdr_across_dataset(dataset_dict, model, sample_rate, cpu_device, normalize=False, sdr_type="standard", verbose=True)

    # Plot the SDR results
    plot_sdr_results(sdr_collection, model, name="sdr_results")

if __name__ == "__main__":
    main()
    # Uncomment the line below to run the evaluation with scale-invariant SDR
    # sdr_collection = evaluate_sdr_across_dataset(dataset_dict, model, sample_rate, cpu_device, normalize=False, sdr_type="scale_invariant", verbose=True)
    # plot_sdr_results(sdr_collection, model, name="sdr_results_scale_invariant")