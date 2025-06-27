import IPython
import torch
import torchaudio
import os
import matplotlib.pyplot as plt
from IPython.display import Audio
from mir_eval import separation
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
from tqdm import tqdm

# -----------------------------
# 1) Load the Model
# -----------------------------
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

sample_rate = bundle.sample_rate
print(f"Sample rate: {sample_rate}")

# Ensure "plots" folder exists
os.makedirs("./plots", exist_ok=True)

# -----------------------------
# 2) Set Up Dataset Loading
# -----------------------------

# Load dataset and choose the length of the tracks for each song (in seconds)
DATASET_FOLDER = "/Users/alessandromanattini/Desktop/MAE/CAPSTONE/musdb18hq/test"
SEGMENT = 30  # We'll keep exactly 30 seconds from each track

# !!! --> PAY ATTENTION TO THE SILENT PORTION OF THE TRACKS, THEY CAN CAUSE ERRORS IN THE SEPARATION PROCESS <-- !!!
# You should choose a portion of the track that contains all the instrument without the channel sums being zero
# NB: if you encounter a silent portion of the track, you will skip the evaluation for that stem!
# It's not raising an error, but i

track_folders = sorted(
    folder for folder in os.listdir(DATASET_FOLDER)
    if os.path.isdir(os.path.join(DATASET_FOLDER, folder))
)

# Dictionary to store {track_folder -> {stem_name -> waveform}}
dataset_dict = {}

# Each subfolder in musdb18hq/test corresponds to a track
for track_folder in track_folders:
    track_path = os.path.join(DATASET_FOLDER, track_folder)
    if not os.path.isdir(track_path):
        continue

    # Prepare a sub-dictionary for this track
    stems_dict = {}
    stem_names = ["mixture", "drums", "bass", "vocals", "other"]

    for stem_name in stem_names:
        file_path = os.path.join(track_path, f"{stem_name}.wav")
        if not os.path.isfile(file_path):
            print(f"Warning: file not found {file_path}")
            continue

        # Load full audio
        waveform, sr = torchaudio.load(file_path)

        # Keep only the first 30s
        segment_samples = SEGMENT * sr
        waveform_segment = waveform[:, :segment_samples]

        stems_dict[stem_name] = waveform_segment

    dataset_dict[track_folder] = stems_dict

print("Loaded tracks:", list(dataset_dict.keys()))

# -----------------------------
# 3) Choose a Track to Process
# -----------------------------

# Now you have a dictionary with track_folder as the key,
# and a sub-dict with "mixture", "drums", "bass", "vocals", "other" waveforms
track_names = list(dataset_dict.keys())

# For example, pick track #6 or any valid index
track_chosen = track_names[5]
print("Chosen track name:", track_chosen)

stems_available = list(dataset_dict[track_chosen].keys())
print("Stems:", stems_available)  # e.g. ['mixture', 'drums', 'bass', 'vocals', 'other']

# Check duration
mixture_waveform = dataset_dict[track_chosen]["mixture"]
duration_seconds = mixture_waveform.shape[1] / sample_rate
print(f"Duration (seconds): {duration_seconds}")

# Ensure we have all 5 stems
if len(stems_available) < 5:
    print("Warning: Not all stems found. This track might be incomplete.")

# -----------------------------
# 4) Prepare the Data for Separation
# -----------------------------
mixture_waveform = mixture_waveform.to(device)

# We'll do a simple normalization across channels
ref = mixture_waveform.mean(0)  # shape (samples,)
print("Mixture shape:", mixture_waveform.shape)
print("Reference shape:", ref.shape)
print("Reference mean:", ref.mean().item())
print("Reference std:", ref.std().item())

mixture_norm = (mixture_waveform - ref.mean()) / ref.std()

# -----------------------------
# 5) Separation Function
# -----------------------------
def separate_sources(
    model,
    mix,
    segment=30,
    overlap=0.0,  # set to 0.0 to avoid chunk repetition
    device=None
):
    """
    Apply model to a given mixture. Use fade, and add segments together in order to add model segment by segment.

    Args:
        segment (int): segment length in seconds
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.
    """

    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape

    # chunk_len for entire 30s, no overlap
    chunk_len = int(sample_rate * segment * (1 + overlap))  # effectively 30s if overlap=0
    start = 0
    end = chunk_len

    overlap_frames = int(overlap * sample_rate)
    fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

    # Prepare final buffer
    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model(chunk)
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

# -----------------------------
# 6) Run Separation
# -----------------------------
print("Separating 30-second track with no overlap...")
sources_tensor = separate_sources(
    model,
    mixture_norm[None],  # shape (1, channels, samples)
    segment=30,
    overlap=0.0,
    device=device
)[0]  # shape (4, channels, samples)

# Undo normalization
sources_tensor = sources_tensor * ref.std() + ref.mean()

# Build a dict {stem_name -> predicted_stem}
stem_names = model.sources  # ['drums', 'bass', 'other', 'vocals'] typically
predicted_stems = dict(zip(stem_names, list(sources_tensor)))

# -----------------------------
# 7) Evaluate with mir_eval
# -----------------------------
def output_results(original_source: torch.Tensor, predicted_source: torch.Tensor, source: str):
    # Move to CPU
    original_np = original_source.detach().cpu().numpy()
    predicted_np = predicted_source.detach().cpu().numpy()

    # If shape is (C, T), that's fine for mir_eval if C=2
    # but let's ensure it's (2, T) not (T, 2)
    # Usually PyTorch waveforms are (channels, samples),
    # which is correct for bss_eval_sources.

    # Assumiamo che i tensori abbiano la forma corretta (C, T). 
    # Verifichiamo l'energia del riferimento (somma degli assoluti per ogni canale).
    energy = original_source.abs().sum(dim=1)
    print(f"{source} - Energy per channel: {energy}")
    
    # Se uno dei canali ha energia inferiore alla soglia, saltiamo l'evaluation per questo stem
    if (energy < 1e-3).any():
        print(f"Warning: {source} reference appears silent or nearly silent. Skipping evaluation for this stem.")
        return None  # oppure ritorna un valore di default o una stringa informativa
    sdr, sir, sar, _ = separation.bss_eval_sources(
        reference_sources=original_np,
        estimated_sources=predicted_np
    )

    # print(f"--- {source} ---")
    # print("SDR:", sdr.mean())
    # print("SIR:", sir.mean())
    # print("SAR:", sar.mean())
    # print("----------------")

    #return Audio(predicted_source, rate=sample_rate)
    return sdr, sir, sar

# Retrieve references from dataset_dict
drums_ref = dataset_dict[track_chosen]["drums"].to(device)
bass_ref = dataset_dict[track_chosen]["bass"].to(device)
vocals_ref = dataset_dict[track_chosen]["vocals"].to(device)
other_ref = dataset_dict[track_chosen]["other"].to(device)

# Predicted
drums_pred = predicted_stems["drums"]
bass_pred  = predicted_stems["bass"]
vocals_pred = predicted_stems["vocals"]
other_pred = predicted_stems["other"]

# Evaluate each stem
output_results(drums_ref, drums_pred, "Drums")
output_results(bass_ref, bass_pred, "Bass")
output_results(vocals_ref, vocals_pred, "Vocals")
output_results(other_ref, other_pred, "Other")


# Plot the spectrogram of the given STFT, in this case plot the separate sources 
def plot_multiple_spectrograms(*tensors, titles=None, filename="spectrogram.png"):
    n = len(tensors)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2*n))
    for i, tensor in enumerate(tensors):
        magnitude = tensor.abs()
        spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
        axes[i].imshow(spectrogram, cmap="viridis", vmin=-60, vmax=0, origin="lower", aspect="auto")
        if titles:
            axes[i].set_title(titles[i])

    plt.tight_layout()
    plt.savefig(filename)  
    plt.close(fig)



# -----------------------------------------------------------------------------------
# We are going to try to find a mean value of SDR, SIR, SAR for all the tracks
# -----------------------------------------------------------------------------------

sdrlistDrums, sirlistDrums, sarlistDrums = [], [], []
sdrlistBass,  sirlistBass,  sarlistBass  = [], [], []
sdrlistVocals,sirlistVocals,sarlistVocals= [], [], []
sdrlistOther, sirlistOther, sarlistOther = [], [], []

# drums_metrics = [sirlistDrums, sarlistDrums, sdrlistDrums]
# bass_metrics = [sirlistBass, sarlistBass, sdrlistBass]
# vocals_metrics = [sirlistVocals, sarlistVocals, sdrlistVocals]
# other_metrics = [sirlistOther, sarlistOther, sdrlistOther]

# Find the SDR, SAR, SIR for all the tracks and store them in the array
for track in tqdm(track_names, desc="Processing tracks"):
    mixture_waveform = dataset_dict[track]["mixture"]
    duration_seconds = mixture_waveform.shape[1] / sample_rate

    mixture_waveform = mixture_waveform.to(device)

    # We'll do a simple normalization across channels
    ref = mixture_waveform.mean(0)  
    mixture_norm = (mixture_waveform - ref.mean()) / ref.std()

    print("Separatig tracks for : ", track + "...")

    sources_tensor = separate_sources(
        model,
        mixture_norm[None],  # shape (1, channels, samples)
        segment=30,
        overlap=0.0,
        device=device
    )[0] 

    sources_tensor = sources_tensor * ref.std() + ref.mean()

    stem_names = model.sources
    predicted_stems = dict(zip(stem_names, list(sources_tensor)))

    drums_ref = dataset_dict[track]["drums"].to(device)
    bass_ref = dataset_dict[track]["bass"].to(device)
    vocals_ref = dataset_dict[track]["vocals"].to(device)
    other_ref = dataset_dict[track]["other"].to(device)

    drums_pred = predicted_stems["drums"]
    bass_pred  = predicted_stems["bass"]
    vocals_pred = predicted_stems["vocals"]
    other_pred = predicted_stems["other"]

    resultD = output_results(drums_ref, drums_pred, "Drums")
    if resultD is not None:
        sdrD, sirD, sarD = resultD
        sdrlistDrums.append(sdrD)
        sirlistDrums.append(sirD)
        sarlistDrums.append(sarD)

    resultB = output_results(bass_ref, bass_pred, "Bass")
    if resultB is not None:
        sdrB, sirB, sarB = resultB
        sdrlistBass.append(sdrB)
        sirlistBass.append(sirB)
        sarlistBass.append(sarB)

    resultV = output_results(vocals_ref, vocals_pred, "Vocals")
    if resultV is not None:
        sdrV, sirV, sarV = resultV
        sdrlistVocals.append(sdrV)
        sirlistVocals.append(sirV)
        sarlistVocals.append(sarV)

    resultO = output_results(other_ref, other_pred, "Other")
    if resultO is not None:
        sdrO, sirO, sarO = resultO
        sdrlistOther.append(sdrO)
        sirlistOther.append(sirO)
        sarlistOther.append(sarO)

# 2) Compute average
meanSDRDrums = sum(sdrlistDrums) / len(sdrlistDrums) if sdrlistDrums else None
meanSIRDrums = sum(sirlistDrums) / len(sirlistDrums) if sirlistDrums else None
meanSARDrums = sum(sarlistDrums) / len(sarlistDrums) if sarlistDrums else None


print("--- DRUMS MEAN ---")
print("SDR:", meanSDRDrums)
print("SIR:", meanSIRDrums)
print("SAR:", meanSARDrums)
print("----------------")

meanSDRBass = sum(sdrlistBass) / len(sdrlistBass)
meanSIRBass = sum(sirlistBass) / len(sirlistBass)
meanSARBass = sum(sarlistBass) / len(sarlistBass)

print("--- BASS MEAN ---")
print("SDR:", meanSDRBass)
print("SIR:", meanSIRBass)
print("SAR:", meanSARBass)
print("----------------")

meanSDRVocals = sum(sdrlistVocals) / len(sdrlistVocals)
meanSIRVocals = sum(sirlistVocals) / len(sirlistVocals)
meanSARVocals = sum(sarlistVocals) / len(sarlistVocals)

print("--- VOCALS MEAN ---")
print("SDR:", meanSDRVocals)
print("SIR:", meanSIRVocals)
print("SAR:", meanSARVocals)
print("----------------")

meanSDROther = sum(sdrlistOther) / len(sdrlistOther)
meanSIROther = sum(sirlistOther) / len(sirlistOther)
meanSAROther = sum(sarlistOther) / len(sarlistOther)

print("--- OTHER MEAN ---")
print("SDR:", meanSDROther)
print("SIR:", meanSIROther)
print("SAR:", meanSAROther)
print("----------------")



