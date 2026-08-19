import json
import math
import numpy as np

def analyze_segment(vec1280):
    """Computes micro-Doppler spectral properties from a 1280-element complex vector."""
    mat = vec1280.reshape((5, 256))
    win = np.hanning(256)
    fft_res = np.fft.fftshift(np.fft.fft(mat * win, axis=1), axes=1)
    mag = np.abs(fft_res)
    profile = np.mean(mag, axis=0)
    total_pwr = np.sum(profile) + 1e-12
    bins = np.arange(-128, 128)
    centroid = float(np.sum(bins * profile) / total_pwr)
    spread = float(np.sqrt(np.sum(((bins - centroid)**2) * profile) / total_pwr))
    pwr_norm = profile / total_pwr
    entropy = float(-np.sum(pwr_norm * np.log2(pwr_norm + 1e-12)))
    peak_pwr = float(np.max(profile))
    return {
        "centroid": round(centroid, 2),
        "spread": round(spread, 2),
        "entropy": round(entropy, 2),
        "peak_pwr": round(peak_pwr, 2)
    }

def extract_track_points(raw_data, track_idx, sample_stride=2):
    """Extracts downsampled points with micro-Doppler metrics from a track row."""
    row = raw_data[track_idx]
    label_raw = str(row[0][0])
    seg_matrix = row[1]
    ranges = row[2].flatten()
    timestamps = row[3].flatten()
    
    num_segs = seg_matrix.shape[1]
    points = []
    t0 = timestamps[0]
    
    for s_idx in range(0, num_segs, sample_stride):
        t = float(timestamps[s_idx] - t0)
        r = float(ranges[s_idx])
        vec = seg_matrix[:, s_idx]
        metrics = analyze_segment(vec)
        
        if s_idx > 0:
            dt = max(1e-4, float(timestamps[s_idx] - timestamps[s_idx - sample_stride]))
            dr = float(ranges[s_idx] - ranges[s_idx - sample_stride])
            spd = abs(dr / dt)
        else:
            spd = 0.0
            
        points.append({
            "t": round(t, 2),
            "r": round(r, 2),
            "spd": round(spd, 2),
            "metrics": metrics
        })
    
    return {
        "label": label_raw,
        "track_idx": track_idx,
        "points": points,
        "total_duration": round(float(timestamps[-1] - t0), 2)
    }

def build_scenarios():
    print("Loading radar_data.npy...")
    data = np.load("radar_data.npy", allow_pickle=True)
    
    print("Extracting representative radar tracks...")
    track_d1_0 = extract_track_points(data, 0, sample_stride=2)     # D1 Rogue drone (122s)
    track_d3   = extract_track_points(data, 19, sample_stride=2)    # D3 Quadcopter (156s)
    track_d6   = extract_track_points(data, 32, sample_stride=2)    # D6 Commercial UAV (55s)
    track_gull = extract_track_points(data, 78, sample_stride=2)    # Seagull biological (50s)
    track_cr   = extract_track_points(data, 111, sample_stride=2)   # Corner Reflector / Clutter (30s)

    scenarios = {
        "scenario_radar_d1": {
            "id": "scenario_radar_d1",
            "name": "Live FMCW Radar: D1 Incursion (radar_data.npy)",
            "description": "True FMCW 77 GHz radar track of DJI Matrice class drone (D1) penetrating EKCH restricted airspace. Features real micro-Doppler harmonics.",
            "duration": 120,
            "tracks": [
                {
                    "id": "UAS-0421",
                    "raw_source": "radar_data.npy [Track 0: D1]",
                    "kind": "radar-d1-rogue",
                    "label": "drone",
                    "data": track_d1_0,
                    "geo": {
                        "start_x": 4200, "start_y": -3900,
                        "target_x": 800, "target_y": -600,
                        "alt": 128, "climb": -0.4
                    },
                    "identity": {
                        "model": "DJI Matrice 300 RTK",
                        "mass": 3.6,
                        "remote_id": None,
                        "operator": None
                    }
                },
                {
                    "id": "BRD-1102",
                    "raw_source": "radar_data.npy [Track 78: seagull]",
                    "kind": "radar-bird",
                    "label": "bird",
                    "data": track_gull,
                    "geo": {
                        "start_x": 2600, "start_y": -3100,
                        "target_x": 3800, "target_y": -4200,
                        "alt": 140, "climb": 0.2
                    }
                },
                {
                    "id": "SAS1745",
                    "raw_source": "ADS-B / Synthetic",
                    "kind": "aircraft",
                    "label": "aircraft",
                    "x": -1400, "y": -1100, "alt": 340, "spd": 92, "hdg": 222, "climb": 9.5,
                    "model": "A320-251N", "rid": "SAS1745 · SQUAWK 3421"
                },
                {
                    "id": "DLH2431",
                    "raw_source": "ADS-B / Synthetic",
                    "kind": "aircraft",
                    "label": "aircraft",
                    "x": 6800, "y": 5600, "alt": 1750, "spd": 104, "hdg": 214, "climb": -7.2,
                    "model": "A321-131", "rid": "DLH2431 · SQUAWK 1055"
                },
                {
                    "id": "HEMS-04",
                    "raw_source": "ADS-B / Synthetic",
                    "kind": "helicopter",
                    "label": "helicopter",
                    "x": -5200, "y": 3600, "alt": 310, "spd": 52, "hdg": 104, "climb": 0,
                    "model": "EC135 T3", "rid": "OY-HUS · HEMS PRIORITY"
                },
                {
                    "id": "WTG-MDG",
                    "raw_source": "radar_data.npy [Track 111: CR]",
                    "kind": "radar-clutter",
                    "label": "clutter",
                    "data": track_cr,
                    "geo": {
                        "start_x": 1100, "start_y": 6250,
                        "target_x": 1100, "target_y": 6250,
                        "alt": 104, "climb": 0
                    }
                }
            ]
        },
        "scenario_radar_multiclass": {
            "id": "scenario_radar_multiclass",
            "name": "Micro-Doppler Discrimination: Drone (D3) vs Seagull vs Clutter",
            "description": "Simultaneous real FMCW radar recordings demonstrating spectral discrimination between multirotor drones, biological wingbeats, and static corner reflectors.",
            "duration": 150,
            "tracks": [
                {
                    "id": "UAS-0914",
                    "raw_source": "radar_data.npy [Track 19: D3]",
                    "kind": "radar-d3-incursion",
                    "label": "drone",
                    "data": track_d3,
                    "geo": {
                        "start_x": -3800, "start_y": 3200,
                        "target_x": -400, "target_y": 200,
                        "alt": 95, "climb": -0.2
                    },
                    "identity": {
                        "model": "DJI Phantom 4 Pro",
                        "mass": 1.38,
                        "remote_id": None,
                        "operator": None
                    }
                },
                {
                    "id": "UAS-0388",
                    "raw_source": "radar_data.npy [Track 32: D6]",
                    "kind": "radar-d6-friendly",
                    "label": "drone",
                    "data": track_d6,
                    "geo": {
                        "start_x": -6300, "start_y": -4200,
                        "target_x": -3200, "target_y": -2800,
                        "alt": 96, "climb": 0
                    },
                    "identity": {
                        "model": "DJI Mavic 3 Enterprise",
                        "mass": 0.92,
                        "remote_id": "DK-CAA-88213",
                        "operator": "Kastrup Survey ApS"
                    }
                },
                {
                    "id": "BRD-4410",
                    "raw_source": "radar_data.npy [Track 78: seagull]",
                    "kind": "radar-bird",
                    "label": "bird",
                    "data": track_gull,
                    "geo": {
                        "start_x": 1800, "start_y": -2200,
                        "target_x": 4200, "target_y": -1100,
                        "alt": 115, "climb": 0.5
                    }
                },
                {
                    "id": "CLT-CR01",
                    "raw_source": "radar_data.npy [Track 111: CR]",
                    "kind": "radar-clutter",
                    "label": "clutter",
                    "data": track_cr,
                    "geo": {
                        "start_x": 2200, "start_y": 4800,
                        "target_x": 2200, "target_y": 4800,
                        "alt": 85, "climb": 0
                    }
                },
                {
                    "id": "SAS0912",
                    "raw_source": "ADS-B / Synthetic",
                    "kind": "aircraft",
                    "label": "aircraft",
                    "x": 7500, "y": 6200, "alt": 1800, "spd": 110, "hdg": 218, "climb": -8.0,
                    "model": "CRJ-900", "rid": "SAS0912 · SQUAWK 4102"
                }
            ]
        }
    }
    
    js_content = "/* Auto-generated from radar_data.npy */\nwindow.RADAR_SCENARIOS = " + json.dumps(scenarios) + ";\n"
    with open("radar_scenarios.js", "w") as f:
        f.write(js_content)
    
    print(f"Successfully generated radar_scenarios.js ({len(js_content) // 1024} KB)")

if __name__ == "__main__":
    build_scenarios()
