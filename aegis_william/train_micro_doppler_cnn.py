import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using compute device: {device}")

# ---------------------------------------------------------
# 1. Dataset & Preprocessing
# ---------------------------------------------------------
DRONE_CLASSES = {'D1', 'D2', 'D3', 'D4', 'D5', 'D6'}
BIRD_CLASSES = {'seagull', 'black-headed gull', 'heron', 'pigeon', 'raven', 'seagull and black-headed gull'}
CLUTTER_CLASSES = {'CR'}
HUMAN_CLASSES = {'human_walk', 'human_run'}

CLASS_MAP = {
    'drone': 0,
    'bird': 1,
    'clutter': 2,
    'human': 3
}
CLASS_NAMES = ['drone', 'bird', 'clutter', 'human']

class RadarAugmentations:
    @staticmethod
    def add_complex_awgn(signal_complex: np.ndarray, snr_db_range=(15, 30)) -> np.ndarray:
        sig_power = np.mean(np.abs(signal_complex) ** 2)
        target_snr_db = np.random.uniform(*snr_db_range)
        noise_power = sig_power / (10 ** (target_snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)
        noise = np.random.normal(0, noise_std, signal_complex.shape) + 1j * np.random.normal(0, noise_std, signal_complex.shape)
        return signal_complex + noise

    @staticmethod
    def random_doppler_shift(rd_map: np.ndarray, max_shift_bins: int = 12) -> np.ndarray:
        shift = np.random.randint(-max_shift_bins, max_shift_bins + 1)
        return np.roll(rd_map, shift=shift, axis=1)

    @staticmethod
    def random_doppler_flip(rd_map: np.ndarray, p: float = 0.5) -> np.ndarray:
        if np.random.rand() < p:
            return np.flip(rd_map, axis=1)
        return rd_map


class RadarDataset(Dataset):
    SPLIT_MAP = {'train': 1, 'val': 2, 'test': 3}

    def __init__(self, raw_data, split='train', filter_edges=True, apply_aug=True):
        self.split_code = self.SPLIT_MAP[split.lower()]
        self.apply_aug = (split == 'train') and apply_aug
        self.window = np.hanning(256).reshape(1, 256)
        self.samples = []
        self._index(raw_data, filter_edges)

    def _index(self, data, filter_edges):
        for m_idx in range(len(data)):
            raw_label = np.asarray(data[m_idx, 0]).squeeze()
            label_str = str(raw_label.item() if raw_label.ndim == 0 else raw_label[0]).strip()
            
            if label_str in DRONE_CLASSES:
                c_idx = CLASS_MAP['drone']
            elif label_str in BIRD_CLASSES:
                c_idx = CLASS_MAP['bird']
            elif label_str in CLUTTER_CLASSES:
                c_idx = CLASS_MAP['clutter']
            else:
                c_idx = CLASS_MAP['human']

            complex_matrix = np.asarray(data[m_idx, 1])
            splits = np.asarray(data[m_idx, 4]).flatten()
            edge_flags = np.asarray(data[m_idx, 5]).flatten()
            num_segments = complex_matrix.shape[1]

            for s_idx in range(num_segments):
                if splits[s_idx] != self.split_code:
                    continue
                if filter_edges and edge_flags[s_idx] == 1:
                    continue
                vec = complex_matrix[:, s_idx]
                self.samples.append((vec, c_idx))

    def __len__(self):
        return len(self.samples)

    def _process_signal(self, raw_vec):
        sig = raw_vec.reshape((5, 256))
        if self.apply_aug:
            sig = RadarAugmentations.add_complex_awgn(sig, snr_db_range=(15, 30))
        
        windowed = sig * self.window
        rd_fft = np.fft.fftshift(np.fft.fft(windowed, axis=1), axes=1)
        rd_map = 20 * np.log10(np.abs(rd_fft) + 1e-12)
        
        if self.apply_aug:
            rd_map = RadarAugmentations.random_doppler_shift(rd_map, max_shift_bins=12)
            rd_map = RadarAugmentations.random_doppler_flip(rd_map, p=0.5)
            
        mean = np.mean(rd_map)
        std = np.std(rd_map) + 1e-8
        rd_map = (rd_map - mean) / std
        return rd_map.astype(np.float32)

    def __getitem__(self, idx):
        raw_vec, label = self.samples[idx]
        rd_tensor = self._process_signal(raw_vec)
        x = torch.from_numpy(rd_tensor).unsqueeze(0)  # (1, 5, 256)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ---------------------------------------------------------
# 2. Model Architecture
# ---------------------------------------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 5), padding=(1, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MicroDopplerCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, dropout_rate=0.3):
        super().__init__()
        # Stage 1: (5, 256) -> (5, 128)
        self.stage1 = nn.Sequential(
            ConvBNAct(in_channels, 24, kernel_size=(3, 5), padding=(1, 2)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        # Stage 2: (5, 128) -> (5, 64)
        self.stage2 = nn.Sequential(
            ConvBNAct(24, 48, kernel_size=(3, 5), padding=(1, 2)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        # Stage 3: (5, 64) -> (2, 32)
        self.stage3 = nn.Sequential(
            ConvBNAct(48, 96, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        # Stage 4: (2, 32) -> (1, 16)
        self.stage4 = nn.Sequential(
            ConvBNAct(96, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
def train_model():
    print("Loading radar_data.npy...")
    raw_data = np.load("radar_data.npy", allow_pickle=True)
    
    print("Building datasets (Train / Val / Test)...")
    train_dataset = RadarDataset(raw_data, split='train', apply_aug=True)
    val_dataset   = RadarDataset(raw_data, split='val', apply_aug=False)
    test_dataset  = RadarDataset(raw_data, split='test', apply_aug=False)
    
    print(f"Dataset split sizes: Train={len(train_dataset):,}, Val={len(val_dataset):,}, Test={len(test_dataset):,}")
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    # Class weights for balanced loss
    class_counts = [46760, 5749, 1280, 4028]
    total_samples = sum(class_counts)
    weights = [total_samples / (len(class_counts) * c) for c in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    model = MicroDopplerCNN(in_channels=1, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 5
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    print("\n--- Starting Training ---")
    start_time = time.time()
    best_val_acc = 0.0
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)
            
        scheduler.step()
        train_acc = correct / total
        train_loss = train_loss / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)
                
        val_acc = val_correct / val_total
        val_loss = val_loss / val_total
        print(f"Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:6.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:6.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "micro_doppler_cnn.pth")
            
    train_duration = time.time() - start_time
    print(f"\nTraining completed in {train_duration:.2f}s! Best Val Accuracy: {best_val_acc*100:.2f}%")
    
    # ---------------------------------------------------------
    # 4. Evaluation on Test Set
    # ---------------------------------------------------------
    print("\n--- Evaluating on Test Split (Split 3) ---")
    model.load_state_dict(torch.load("micro_doppler_cnn.pth", map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b = x_b.to(device)
            logits = model(x_b)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = np.mean(all_preds == all_targets)
    
    print(f"Overall Test Accuracy: {test_acc*100:.2f}%\n")
    print("Class-wise Metrics:")
    for c_idx, c_name in enumerate(CLASS_NAMES):
        mask = (all_targets == c_idx)
        c_acc = np.mean(all_preds[mask] == c_idx) if np.sum(mask) > 0 else 0
        tp = np.sum((all_preds == c_idx) & (all_targets == c_idx))
        fp = np.sum((all_preds == c_idx) & (all_targets != c_idx))
        fn = np.sum((all_preds != c_idx) & (all_targets == c_idx))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {c_name:10s} (N={np.sum(mask):5d}): Accuracy={c_acc*100:6.2f}%, Precision={precision*100:6.2f}%, Recall={recall*100:6.2f}%, F1={f1*100:6.2f}%")

    return model, raw_data

if __name__ == '__main__':
    train_model()
