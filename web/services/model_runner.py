import numpy as np
import tensorflow as tf

class ModelRunner:
    def __init__(self):
        self.IMG_SIZE = 256
        self.CHANNELS = 15

    def preprocess(self, raw_patch):
        """
        Mirrors logic from Training.py/tester.py:
        1. Handle NaNs/Zeros (Safety Net)
        2. Normalize if needed
        """
        # Convert to float32
        img = np.array(raw_patch, dtype=np.float32)
        
        # Safety Net: Replace NaNs and Infinities with 0
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check shape (Allow 16 channels for debug)
        if img.shape[2] == 16:
            # Drop the debug band for the model
            debug_band = img[:, :, 15]
            img = img[:, :, :15]
        
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
        # DEBUG: Extract Raw Temp stats before preprocessing strips it
        raw_temps = []
        for p in raw_patches_list:
            if p.shape[2] == 16:
                temp_val = np.mean(p[:, :, 15])
                if temp_val > 0:  # Only log non-zero temps
                    raw_temps.append(temp_val)
        
        if raw_temps:
            avg_temp = np.mean(raw_temps)
            # Temperature should be in Kelvin (273-320K range)
            if avg_temp < 200:
                print(f"WARNING: Raw temp {avg_temp:.2f} is suspiciously low. May indicate data issue.")
            else:
                print(f"Raw GFS Temp: {avg_temp:.2f}K ({avg_temp-273.15:.1f}°C)")

        # 1. Preprocess Batch
        processed_batch = np.array([self.preprocess(p) for p in raw_patches_list])
        
        # DEBUG: Check feature stats
        # Channels: 0-7 (Optical), 8 (Temp), 9 (Hum), 10 (Wind), 11 (Precip), 12-14 (Topo/Pop)
        avg_temp = np.mean(processed_batch[:, :, :, 8])
        avg_hum = np.mean(processed_batch[:, :, :, 9])
        avg_wind = np.mean(processed_batch[:, :, :, 10])
        
        print(f"--- BATCH DEBUG STATS ---")
        print(f"Avg Normalized Temp: {avg_temp:.4f} (Expected ~0.5-0.9)")
        print(f"Avg Normalized Hum:  {avg_hum:.4f} (Expected ~0.1-0.6)")
        print(f"Avg Normalized Wind: {avg_wind:.4f} (Expected ~0.1-0.3)")
        
        # 2. Run Prediction
        # Returns shape (Batch, 1) or (Batch, 2) depending on output layer
        preds = model.predict(processed_batch, verbose=0)
        
        # 3. Extract Probabilities
        if preds.shape[-1] == 1:
            probs = preds.flatten()
        else:
            probs = preds[:, 1]
            
        # 4. Post-Process: Ocean/Water Masking
        # Heuristic: If Elevation (Band 12) is 0 AND NDVI (Band 6) is low (< 0.1), it's likely water.
        # Note: Band indices in processed_batch:
        # 0: Blue, 1: Green, 2: Red, 3: NIR, 4: SWIR1, 5: SWIR2
        # 6: NDVI, 7: NDMI
        # 8: Temp, 9: Hum, 10: Wind, 11: Precip
        # 12: Elevation, 13: Slope, 14: Pop
        
        final_probs = []
        for i, prob in enumerate(probs):
            # Ocean filtering should already be done in gee_layer.py
            # This is a final safety net for any missed ocean points
            elev = np.mean(processed_batch[i, :, :, 12])
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