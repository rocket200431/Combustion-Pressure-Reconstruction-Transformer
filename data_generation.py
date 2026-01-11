import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ===================== CONFIG =====================

DATA_PATH = "test12.txt"
PRESSURE_COLUMN = 0

WINDOW_SIZE = 100
MASK_START = 50
MASK_END   = 60

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.1
TEST_RATIO  = 0.2

BATCH_SIZE = 64
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ===================== STEP 1: READ RAW PRESSURE SIGNAL =====================

sampling_freq = None
pressure_vals = []
reading = False

with open(DATA_PATH, "r") as f:
    for line in f:
        if sampling_freq is None and "Sampling_frequency" in line:
            sampling_freq = float(line.strip().split()[-1])
            print(f"Sampling frequency = {sampling_freq} Hz")
            continue

        if "BEGIN_DATA" in line:
            reading = True
            continue
        if "END_DATA" in line:
            break

        if reading:
            parts = line.split()
            if len(parts) >= 3:
                pressure_vals.append(float(parts[PRESSURE_COLUMN]))

pressure = torch.tensor(pressure_vals, dtype=torch.float32)
N_TOTAL = pressure.shape[0]
print(f"Total pressure samples loaded: {N_TOTAL}")

# ===================== STEP 2: TRAIN-ONLY NORMALIZATION =====================

train_cut = int(TRAIN_RATIO * N_TOTAL)
mean = pressure[:train_cut].mean()
std  = pressure[:train_cut].std()

pressure = (pressure - mean) / (std + 1e-8)

# ===================== STEP 3: WINDOW CREATION + SHUFFLING =====================

def create_windows(signal, window_size):
    windows = []
    for i in range(len(signal) - window_size + 1):
        windows.append(signal[i : i + window_size])

    windows = torch.stack(windows)  # (N_windows, WINDOW_SIZE)

    # 🔹 SHUFFLE WINDOWS (order only, not time inside window)
    perm = torch.randperm(windows.shape[0])
    windows = windows[perm]

    return windows

windows = create_windows(pressure, WINDOW_SIZE)
print(f"Total windows created: {windows.shape[0]}")

# ===================== STEP 4: TRAIN / VAL / TEST SPLIT =====================

N = windows.shape[0]

train_end = int(TRAIN_RATIO * N)
val_end   = train_end + int(VAL_RATIO * N)

train_windows = windows[:train_end]
val_windows   = windows[train_end:val_end]
test_windows  = windows[val_end:]

# ===================== STEP 5: DATASET WITH MASKING =====================

class MaskedPressureDataset(Dataset):
    def __init__(self, windows, mask_start, mask_end):
        self.windows = windows
        self.mask_start = mask_start
        self.mask_end = mask_end

    def __len__(self):
        return self.windows.shape[0]

    def __getitem__(self, idx):
        window = self.windows[idx]          # (WINDOW_SIZE,)

        value = window.clone()
        mask  = torch.zeros_like(value)

        # Apply mask
        mask[self.mask_start:self.mask_end] = 1.0
        value[self.mask_start:self.mask_end] = 0.0

        # Input: (WINDOW_SIZE, 2) -> [value, mask]
        X = torch.stack([value, mask], dim=-1)

        # Target: masked pressure values
        y = window[self.mask_start:self.mask_end]

        return X, y

train_ds = MaskedPressureDataset(train_windows, MASK_START, MASK_END)
val_ds   = MaskedPressureDataset(val_windows, MASK_START, MASK_END)
test_ds  = MaskedPressureDataset(test_windows, MASK_START, MASK_END)

# ===================== STEP 6: DATALOADERS (NO SHUFFLING) =====================

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

print("DataLoaders created (window-level shuffling applied)")

# ===================== HELPER =====================

def get_dataloaders():
    return train_loader, val_loader, test_loader
