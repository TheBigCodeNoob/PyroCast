import numpy as np
import tensorflow as tf

class ModelRunner:
    # Indices in the v2 stack produced by gee_layer.py (last band = Temp_Raw debug).
    # 0-7  : Blue, Green, Red, NIR, SWIR1, SWIR2, NDVI, NDMI
    # 8-13 : Temp_Max, Humidity_Min, Wind_Speed, Precip, ERC, FM100
    # 14   : Elevation
    # 15-17: LC_Forest, LC_Wetland, LC_Open
    # 18   : Pop_Density
    # 19   : Temp_Raw (debug; stripped before inference)
    V2_CHANNELS = 19

    # v1 layout (legacy 15ch model).  Maps v1 index -> source band in the v2 stack
    # produced by gee_layer.py.  v1 had Slope at index 13 which v2 doesn't carry, so
    # we synthesize a zero plane there.
    V1_FROM_V2_INDEX = [0, 1, 2, 3, 4, 5, 6, 7,  # optical + indices (same)
                         8, 9, 10, 11,            # tmmx, rmin, vs, pr (same)
                         14,                      # Elevation (was index 12 in v1)
                         None,                    # Slope (no longer present -> zeros)
                         18]                      # Pop_Density (was index 14 in v1)

    def __init__(self, expected_channels=None):
        self.IMG_SIZE = 256
        # Default to the v2 layout but adapt if the loaded model declares 15.
        self.CHANNELS = expected_channels if expected_channels in (15, 19) else self.V2_CHANNELS
        self.is_v1 = (self.CHANNELS == 15)
        if self.is_v1:
            print("[ModelRunner] Adapting v2 GEE stack -> v1 15ch model.")
        else:
            print("[ModelRunner] Using native v2 19ch layout.")

    def preprocess(self, raw_patch):
        """
        Convert the v2 GEE stack (CHANNELS+1 bands incl. Temp_Raw debug) into the
        layout the loaded model expects.
        """
        img = np.array(raw_patch, dtype=np.float32)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Strip the Temp_Raw debug band from the v2 stack regardless of model version.
        if img.shape[2] == self.V2_CHANNELS + 1:
            img = img[:, :, :self.V2_CHANNELS]

        # If the model is v1 (15ch), reproject the v2 bands into v1 order.
        if self.is_v1 and img.shape[2] == self.V2_CHANNELS:
            v1 = np.zeros((img.shape[0], img.shape[1], 15), dtype=np.float32)
            for v1_idx, v2_idx in enumerate(self.V1_FROM_V2_INDEX):
                if v2_idx is not None:
                    v1[:, :, v1_idx] = img[:, :, v2_idx]
                # else: stays zero (e.g. Slope no longer collected)
            img = v1
        
        # Check shape
        if img.shape != (self.IMG_SIZE, self.IMG_SIZE, self.CHANNELS):
            # Pad or Crop if GEE returned slightly different size
            # Simple resize via padding
            temp = np.zeros((self.IMG_SIZE, self.IMG_SIZE, self.CHANNELS), dtype=np.float32)
            h, w, c = img.shape
            min_h = min(h, self.IMG_SIZE)
            min_w = min(w, self.IMG_SIZE)
            min_c = min(c, self.CHANNELS)
            temp[:min_h, :min_w, :min_c] = img[:min_h, :min_w, :min_c]
            img = temp

        return img

    def predict_batch(self, model, raw_patches_list):
        """
        Runs batch inference.
        """
        # DEBUG: Extract Raw Temp stats from the v2 stack (the GEE layer always
        # produces V2_CHANNELS+1 bands regardless of which model is loaded).
        raw_temps = []
        debug_idx = self.V2_CHANNELS  # Temp_Raw is the last band of the GEE output
        for p in raw_patches_list:
            if p.shape[2] == self.V2_CHANNELS + 1:
                temp_val = np.mean(p[:, :, debug_idx])
                if temp_val > 0:  # Only log non-zero temps
                    raw_temps.append(temp_val)
        
        if raw_temps:
            avg_temp = np.mean(raw_temps)
            # Temperature should be in Kelvin (273-320K range)
            if avg_temp < 200:
                print(f"WARNING: Raw temp {avg_temp:.2f} is suspiciously low. May indicate data issue.")
            else:
                print(f"Raw GFS Temp: {avg_temp:.2f}K ({avg_temp-273.15:.1f}°C)")

        # 1. Preprocess Batch (reshapes & re-orders to whichever layout the model expects).
        processed_batch = np.array([self.preprocess(p) for p in raw_patches_list])

        # DEBUG: Check feature stats. Indices below match the *processed* batch layout.
        # v2 (19ch):  Temp=8, Hum=9, Wind=10, Precip=11, ERC=12, FM100=13, Elev=14
        # v1 (15ch):  Temp=8, Hum=9, Wind=10, Precip=11, Elev=12
        avg_temp = np.mean(processed_batch[:, :, :, 8])
        avg_hum = np.mean(processed_batch[:, :, :, 9])
        avg_wind = np.mean(processed_batch[:, :, :, 10])
        print(f"--- BATCH DEBUG STATS ---")
        print(f"Avg Normalized Temp:  {avg_temp:.4f} (Expected ~0.5-0.9)")
        print(f"Avg Normalized Hum:   {avg_hum:.4f} (Expected ~0.1-0.6)")
        print(f"Avg Normalized Wind:  {avg_wind:.4f} (Expected ~0.1-0.3)")
        if not self.is_v1:
            avg_erc = np.mean(processed_batch[:, :, :, 12])
            avg_fm100 = np.mean(processed_batch[:, :, :, 13])
            print(f"Avg Normalized ERC:   {avg_erc:.4f} (Expected ~0.2-0.8 in fire season)")
            print(f"Avg Normalized FM100: {avg_fm100:.4f} (Lower = drier fuel)")
        
        # 2. Run Prediction
        # Returns shape (Batch, 1) or (Batch, 2) depending on output layer
        preds = model.predict(processed_batch, verbose=0)
        
        # 3. Extract Probabilities
        if preds.shape[-1] == 1:
            probs = preds.flatten()
        else:
            probs = preds[:, 1]
            
        # 4. Post-Process: Ocean/Water Masking.
        # NDVI is at index 6 in both v1 and v2. Elevation moved from 12 (v1) -> 14 (v2).
        elev_idx = 12 if self.is_v1 else 14
        final_probs = []
        for i, prob in enumerate(probs):
            elev = np.mean(processed_batch[i, :, :, elev_idx])
            ndvi = np.mean(processed_batch[i, :, :, 6])
            
            # Final water check: More strict thresholds
            # Normalized elevation: 0.01 = ~40m, so 0.005 = ~20m
            if elev <= 0.005 and ndvi < 0.12:
                final_probs.append(0.0)  # Definitely water/ocean
            else:
                # Ensure JSON compliance: Replace NaN/Inf with 0.0
                val = float(prob)
                if np.isnan(val) or np.isinf(val):
                    val = 0.0
                final_probs.append(val)
                
        return final_probs