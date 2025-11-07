"""
=============================================================================
APP 1: SLOPE FAILURE HAZARD ZONATION
=============================================================================

Implements the Slope Failure methodology from the manual.
Zonation is determined by the HEIGHT (H) of the potential source area.

METHODOLOGY (Per Manual):
1. Identify source areas using LHZM levels (typically 3-5)
2. Calculate height (H) by tracing uphill using steepest ascent
3. Red Zone = Source area + 1H downslope
4. Yellow Zone = 10m uphill + 2H downslope (max 100m)
5. Post-process with area filtering and smoothing

INPUTS (GeoTIFF):
- Filled DEM
- Flow Direction (D8)
- LHZM raster (levels 1-5)

OUTPUT (GeoTIFF):
- 0 = Background
- 1 = Yellow zone
- 2 = Red zone

Author: Based on SF.py with bug fixes and manual compliance
Date: October 2025
Version: 1.0
=============================================================================
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import threading
import numpy as np
import rasterio
from scipy.ndimage import generic_filter, label
import math
from numba import njit
import os
from datetime import datetime


@njit(cache=True)
def calculate_slope_height_numba(dem, cell_size, lhzm_mask):
    """
    Calculates slope height using the pure "steepest ascent" methodology,
    BUT ONLY WITHIN LHZM AREAS (like Java code does).
    This is critical: valley LHZM patches will have small H, slope LHZM will have large H.
    """
    rows, cols = dem.shape
    height_grid = np.zeros_like(dem, dtype=np.float32)

    # Numba is extremely fast at running simple loops like this
    for r in range(rows):
        for c in range(cols):
            # Numba works with NaN for NoData, which we provide
            if np.isnan(dem[r, c]):
                continue
            
            # Only calculate height for LHZM cells
            if not lhzm_mask[r, c]:
                continue

            current_h = 0.0
            cr, cc = r, c
            
            # Steepest ascent loop - ONLY WITHIN LHZM AREAS
            for _ in range(rows + cols): # Safety break
                best_slope = -1.0
                next_pos_r, next_pos_c = -1, -1

                # Check all 8 neighbors
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0: continue
                        
                        nr, nc = cr + dr, cc + dc
                        
                        if not (0 <= nr < rows and 0 <= nc < cols) or np.isnan(dem[nr, nc]):
                            continue
                        
                        # CRITICAL: Stop if neighbor is NOT LHZM (like Java line 1348)
                        if not lhzm_mask[nr, nc]:
                            continue
                        
                        elev_diff = dem[nr, nc] - dem[cr, cc]
                        if elev_diff <= 0: continue
                        
                        dist = cell_size * (math.sqrt(2.0) if abs(dr) + abs(dc) == 2 else 1.0)
                        slope = elev_diff / dist
                        
                        if slope > best_slope:
                            best_slope = slope
                            next_pos_r, next_pos_c = nr, nc
                
                if next_pos_r != -1:
                    current_h += dem[next_pos_r, next_pos_c] - dem[cr, cc]
                    cr, cc = next_pos_r, next_pos_c
                else:
                    break # Reached a peak or exited LHZM
            
            height_grid[r, c] = current_h
            
    return height_grid

@njit(cache=True)
def simulate_runout_numba(dem, flow_dir, height_grid, initiation_points, max_runout_len_m, cell_size, fd_map_keys, fd_map_vals):
    """
    Numba-accelerated version of the runout simulation.
    FIXED: Only simulates runout from BOTTOM EDGE (toe) of source areas.
    """
    rows, cols = dem.shape
    zone_grid = np.zeros_like(dem, dtype=np.int16)
    init_coords = np.argwhere(initiation_points)
    
    # STEP 1: Mark ALL initiation points (source areas) as RED
    for i in range(len(init_coords)):
        r, c = init_coords[i]
        zone_grid[r, c] = 2
    
    # STEP 2: Identify TRUE BOTTOM EDGE cells (toe of slopes, not valley edges)
    # A cell is a true bottom edge if:
    # 1. Its downslope neighbor is NOT LHZM, AND
    # 2. It has at least one UPSLOPE LHZM neighbor (i.e., it's at bottom of a slope)
    bottom_edge_cells = []
    
    for i in range(len(init_coords)):
        r, c = init_coords[i]
        
        # Get flow direction
        fd_val = flow_dir[r, c]
        
        fd_idx = -1
        for j in range(len(fd_map_keys)):
            if fd_map_keys[j] == fd_val:
                fd_idx = j
                break
        
        if fd_idx == -1:
            continue
        
        # Get downslope neighbor
        dr, dc = fd_map_vals[fd_idx]
        nr, nc = r + dr, c + dc
        
        # Check if downslope neighbor is within bounds
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        
        # CONDITION 1: Downslope neighbor is NOT LHZM
        if not initiation_points[nr, nc]:
            # CONDITION 2: Check if this cell has UPSLOPE LHZM neighbors
            # (i.e., it's at the bottom of a slope, not an isolated cell)
            has_upslope_lhzm = False
            
            for kr in range(-1, 2):
                for kc in range(-1, 2):
                    if kr == 0 and kc == 0:
                        continue
                    
                    nbr_r, nbr_c = r + kr, c + kc
                    
                    if not (0 <= nbr_r < rows and 0 <= nbr_c < cols):
                        continue
                    
                    # Check if this neighbor is LHZM and flows INTO current cell
                    if initiation_points[nbr_r, nbr_c]:
                        nbr_fd = flow_dir[nbr_r, nbr_c]
                        nbr_fd_idx = -1
                        for j in range(len(fd_map_keys)):
                            if fd_map_keys[j] == nbr_fd:
                                nbr_fd_idx = j
                                break
                        
                        if nbr_fd_idx != -1:
                            nbr_dr, nbr_dc = fd_map_vals[nbr_fd_idx]
                            # If neighbor flows to current cell
                            if nbr_r + nbr_dr == r and nbr_c + nbr_dc == c:
                                has_upslope_lhzm = True
                                break
                
                if has_upslope_lhzm:
                    break
            
            # Only add if it has upslope LHZM (i.e., it's at bottom of a slope)
            if has_upslope_lhzm:
                bottom_edge_cells.append((r, c))
    
    # STEP 3: Simulate runout ONLY from bottom edge cells
    for idx in range(len(bottom_edge_cells)):
        r, c = bottom_edge_cells[idx]
        
        H = height_grid[r, c]
        if H <= 0: continue

        red_dist_m = H
        yellow_dist_m = min(2 * H, max_runout_len_m)
        dist_traveled = 0.0
        cr, cc = r, c

        for _ in range(rows + cols):
            if np.isnan(dem[cr, cc]): break
            
            fd_val = flow_dir[cr, cc]
            
            fd_idx = -1
            for j in range(len(fd_map_keys)):
                if fd_map_keys[j] == fd_val:
                    fd_idx = j
                    break
            
            if fd_idx == -1: break

            dr, dc = fd_map_vals[fd_idx]
            nr, nc = cr + dr, cc + dc

            if not (0 <= nr < rows and 0 <= nc < cols) or np.isnan(dem[nr, nc]): break
            
            step_dist = cell_size * (math.sqrt(2.0) if abs(dr) + abs(dc) == 2 else 1.0)
            dist_traveled += step_dist

            # Only mark cells that are NOT already source area (red)
            if dist_traveled <= red_dist_m:
                if zone_grid[nr, nc] < 2: zone_grid[nr, nc] = 2
            elif dist_traveled <= yellow_dist_m:
                if zone_grid[nr, nc] < 1: zone_grid[nr, nc] = 1
            else:
                break
            cr, cc = nr, nc
            
    return zone_grid

def add_uphill_buffer(zone_grid, dem, flow_dir, initiation_points, profile, buffer_distance_m):
    """Adds a 10m yellow zone buffer UPSLOPE from initiation points."""
    print("Adding 10m uphill yellow zone buffer...")
    cell_size = abs(profile['transform'][0])
    rows, cols = dem.shape
    ud_map = {
    1:  (0, -1),   # West (opposite of East)
    2:  (-1, -1),  # Northwest (opposite of Southeast)
    4:  (-1, 0),   # North (opposite of South)
    8:  (-1, 1),   # Northeast (opposite of Southwest)
    16: (0, 1),    # East (opposite of West)
    32: (1, 1),    # Southeast (opposite of Northwest)
    64: (1, 0),    # South (opposite of North)
    128: (1, -1)   # Southwest (opposite of Northeast)
}
    init_coords = np.argwhere(initiation_points)
    for r, c in init_coords:
        dist_traveled = 0.0
        cr, cc = r, c
        for _ in range(int(buffer_distance_m / cell_size) + 2):
            if dem.mask[cr, cc]: break
            fd_val = flow_dir[cr, cc]
            if fd_val not in ud_map: break
            dr, dc = ud_map[fd_val]
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or dem.mask[nr, nc]: break
            step_dist = cell_size * (math.sqrt(2) if abs(dr) + abs(dc) == 2 else 1.0)
            dist_traveled += step_dist
            if dist_traveled > buffer_distance_m:
                break
            if zone_grid[nr, nc] == 0:
                zone_grid[nr, nc] = 1
            cr, cc = nr, nc
            
    return zone_grid

@njit(cache=True)
def majority_filter_numba(window):
    """
    Numba-accelerated version of the majority filter.
    It finds the most frequent non-zero value in a 3x3 window.
    """
    # The window from generic_filter is a 1D array of 9 elements
    # We will count the occurrences of 1 (Yellow) and 2 (Red)
    count_1 = 0
    count_2 = 0
    for i in range(window.size):
        val = window[i]
        if val == 1:
            count_1 += 1
        elif val == 2:
            count_2 += 1
    
    # --- Determine the winner ---
    # If Red is more common, the result is Red
    if count_2 > count_1:
        return 2
    # If Yellow is more common, the result is Yellow
    elif count_1 > count_2:
        return 1
    # If they are tied, or if there are no non-zero values,
    # we need to check the original center pixel's value.
    # The center pixel is always at index 4 in a 3x3 window.
    center_val = window[4]
    
    # If the center was already a valid zone, keep it in a tie.
    # Otherwise, the area is probably background (0).
    if center_val == 1 or center_val == 2:
        return center_val
    else:
        return 0

# --- GUI APPLICATION ---

class LandslideAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("App 1: Slope Failure Hazard Zonation (Height-based)")
        self.geometry("650x700")
        
           # --- Variables (Simplified) ---
           
        self.dem_path = tk.StringVar()
        self.flowdir_path = tk.StringVar()
        self.lhzm_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.min_height = tk.DoubleVar(value=5.0)
        self.max_runout_len = tk.DoubleVar(value=100.0)
        self.min_area = tk.DoubleVar(value=0.0)
        self.smooth_output = tk.BooleanVar(value=True)
        self.lhzm_levels_vars = {i: tk.BooleanVar(value=(i >= 3)) for i in range(1, 6)}
        self.cancel_flag = False
        
        self._build_gui()

    def _build_gui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        file_frame = ttk.LabelFrame(main_frame, text="Input/Output Files", padding="10")
        file_frame.pack(fill=tk.X, expand=True, pady=5)
        self._create_file_entry(file_frame, 0, "1. Filled DEM (.tif):", self.dem_path)
        self._create_file_entry(file_frame, 1, "2. Flow Direction (.tif):", self.flowdir_path)
        self._create_file_entry(file_frame, 2, "3. LHZM Raster (.tif):", self.lhzm_path)
        self._create_file_entry(file_frame, 3, "4. Output File (.tif):", self.output_path, save=True)
        
        param_frame = ttk.LabelFrame(main_frame, text="Calculation Parameters", padding="10")
        param_frame.pack(fill=tk.X, expand=True, pady=5)
        
        lhzm_level_frame = ttk.Frame(param_frame)
        lhzm_level_frame.grid(row=0, column=0, columnspan=4, sticky='w', pady=5)
        ttk.Label(lhzm_level_frame, text="LHZM Levels to use as Source Areas:").pack(side=tk.LEFT)
        for i in range(1, 6):
                ttk.Checkbutton(lhzm_level_frame, text=f"Lv.{i}", variable=self.lhzm_levels_vars[i]).pack(side=tk.LEFT)
        
        ttk.Separator(param_frame, orient='horizontal').grid(row=1, column=0, columnspan=4, sticky='ew', pady=10)
        ttk.Label(param_frame, text="Min Slope Height (m):").grid(row=2, column=0, sticky='w')
        ttk.Entry(param_frame, textvariable=self.min_height, width=10).grid(row=2, column=1, sticky='w')
        ttk.Label(param_frame, text="Max Runout (m):").grid(row=2, column=2, sticky='w', padx=10)
        ttk.Entry(param_frame, textvariable=self.max_runout_len, width=10).grid(row=2, column=3, sticky='w')
        
        post_frame = ttk.LabelFrame(main_frame, text="Post-Processing", padding="10")
        post_frame.pack(fill=tk.X, expand=True, pady=5)
        ttk.Checkbutton(post_frame, text="Smooth Output (Majority Filter)", variable=self.smooth_output).grid(row=0, column=0, sticky='w')
        ttk.Label(post_frame, text="Min Area (m²):").grid(row=0, column=1, sticky='w', padx=10)
        ttk.Entry(post_frame, textvariable=self.min_area, width=10).grid(row=0, column=2, sticky='w')
        
        action_frame = ttk.Frame(main_frame, padding="10")
        action_frame.pack(fill=tk.BOTH, expand=True)
        self.run_button = ttk.Button(action_frame, text="Start Calculation", command=self._start_calculation_thread)
        self.run_button.pack(pady=5)
        self.cancel_button = ttk.Button(action_frame, text="Cancel", command=self.cancel_operation, state='disabled')
        self.cancel_button.pack(pady=5)
        self.log_window = scrolledtext.ScrolledText(action_frame, height=10, state='disabled', wrap='word')
        self.log_window.pack(fill=tk.BOTH, expand=True)

    def _create_file_entry(self, parent, r, label_text, var, save=False):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=r, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=var, width=50)
        entry.grid(row=r, column=1, sticky=(tk.W, tk.E))
        button_text = "Save As..." if save else "Browse..."
        action = lambda: self._browse_save(var) if save else self._browse_file(var)
        button = ttk.Button(parent, text=button_text, command=action)
        button.grid(row=r, column=2, sticky=tk.W, padx=5)
        parent.columnconfigure(1, weight=1)

    def _browse_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("GeoTIFF", "*.tif;*.tiff")])
        if path: var.set(path)

    def _browse_save(self, var):
        path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("GeoTIFF", "*.tif;*.tiff")])
        if path: var.set(path)

    def log(self, message: str):
      def _append(): 
        self.log_window.config(state='normal')
        self.log_window.insert(tk.END, message + "\n")
        self.log_window.config(state='disabled')
        self.log_window.see(tk.END)
      self.after(0, _append)
    
    def _start_calculation_thread(self):
        if not all([self.dem_path.get(), self.flowdir_path.get(), self.lhzm_path.get(), self.output_path.get()]):
            messagebox.showerror("Input Error", "All input and output file paths are required.")
            return

        # Disable run button and enable cancel button
        self.run_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.cancel_flag = False
        
        # Clear log window
        self.log_window.config(state='normal')
        self.log_window.delete(1.0, tk.END)
        self.log_window.config(state='disabled')
        self.log("--- Starting Calculation ---")
        
        # Force GUI update to show enabled cancel button
        self.update_idletasks()
        
        # Start calculation in background thread
        thread = threading.Thread(target=self._run_calculation, daemon=True)
        thread.start()
        
    def cancel_operation(self):
        """Set flag to cancel the current operation."""
        self.cancel_flag = True
        self.log("Cancelling operation...")
        self.cancel_button.config(state='disabled')

    def _run_calculation(self):
        try:
            # Step 1: Load Data
            self.log("Loading raster files...")
            dem, profile = self._read_raster(self.dem_path.get())
            flow_dir, _ = self._read_raster(self.flowdir_path.get())
            lhzm, _ = self._read_raster(self.lhzm_path.get())
            
            if self.cancel_flag: 
                self.log("Operation cancelled by user.")
                return
            
            # Step 2: Find LHZM initiation points FIRST
            self.log("Finding initiation points from LHZM...")
            selected_levels = [lvl for lvl, var in self.lhzm_levels_vars.items() if var.get()]
            self.log(f"Using LHZM levels: {selected_levels}")
            initiation_points = np.isin(lhzm.filled(0), selected_levels) & (~dem.mask)
            
            if self.cancel_flag: 
                self.log("Operation cancelled by user.")
                return
            
            # Step 3: Calculate Slope Height (H) ONLY WITHIN LHZM AREAS
            self.log("Calculating slope heights (H) within LHZM areas using optimized Numba JIT compiler...")
            self.log("(The first run may be slower as the code is compiled.)")
            self.log("CRITICAL: Height is only calculated within LHZM, like Java methodology.")
            
            dem_for_numba = dem.filled(np.nan)
            cell_size = abs(profile['transform'][0])
            height_grid_raw = calculate_slope_height_numba(dem_for_numba, cell_size, initiation_points)
            height_grid = np.ma.masked_array(height_grid_raw, mask=dem.mask)
            
            self.log("Finished calculating all slope heights.")

            if self.cancel_flag: 
                self.log("Operation cancelled by user.")
                return
            
            # Step 4: Filter those points by height
            self.log(f"Filtering points by minimum height of {self.min_height.get()}m...")
            final_init_points = initiation_points & (height_grid >= self.min_height.get())
            num_points = np.sum(final_init_points)
            if num_points == 0: raise ValueError("No initiation points found meeting all criteria.")
            self.log(f"Found {num_points} valid initiation points after height filtering.")
            
            if self.cancel_flag: 
                self.log("Operation cancelled by user.")
                return
            
             # Step 5: Simulate Downhill Runout using the final points
            self.log("Simulating downslope runout using optimized Numba JIT compiler...")
            
            # Prepare the flow direction map for Numba
            fd_map = {1:(0,1), 2:(1,1), 4:(1,0), 8:(1,-1), 16:(0,-1), 32:(-1,-1), 64:(-1,0), 128:(-1,1)}
            fd_map_keys = np.array(list(fd_map.keys()), dtype=np.int16)
            fd_map_vals = np.array(list(fd_map.values()), dtype=np.int8)

            # This is the call to the new, fast function
            zone_grid = simulate_runout_numba(
                dem.filled(np.nan),
                flow_dir.filled(0).astype(np.int16),
                height_grid.filled(0).astype(np.float32),
                final_init_points.filled(False),
                self.max_runout_len.get(),
                cell_size,
                fd_map_keys,
                fd_map_vals
            )
            
            if self.cancel_flag:
                self.log("Operation cancelled by user.")
                return
            
             # Step 6: Add Uphill Buffer
            self.log("Adding 10m uphill yellow zone buffer...")
            zone_grid = add_uphill_buffer(zone_grid, dem, flow_dir.filled(0), final_init_points, profile, 10.0)

            cell_size = abs(profile['transform'][0])
            if self.min_area.get() > 0:
                self.log(f"Filtering zones smaller than {self.min_area.get()} m²...")
                min_pixels = self.min_area.get() / (cell_size * cell_size)
                for zone_val in [1, 2]:
                    labeled_array, _ = label(zone_grid == zone_val)
                    if labeled_array.max() == 0: continue
                    unique_labels, counts = np.unique(labeled_array, return_counts=True)
                    small_labels = unique_labels[1:][counts[1:] < min_pixels]
                    zone_grid[np.isin(labeled_array, small_labels)] = 0

            if self.smooth_output.get():
                self.log("Smoothing output with majority filter...")
                zone_grid = generic_filter(zone_grid, majority_filter_numba, size=3)
                zone_grid = generic_filter(zone_grid, majority_filter_numba, size=3)

            if self.cancel_flag: return

            self.log(f"Saving output to {self.output_path.get()}...")
            out_dir = os.path.dirname(self.output_path.get())
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            profile.update(dtype=rasterio.int16, count=1, nodata=0, compress='lzw')
            with rasterio.open(self.output_path.get(), 'w', **profile) as dst:
                dst.write(zone_grid.astype(np.int16), 1)

            self.log("--- Calculation Finished Successfully! ---")
            self.after(0, lambda: messagebox.showinfo("Success", "Calculation completed and output saved!"))

        except Exception as e:
            self.log(f"--- ERROR: {e} ---")
            messagebox.showerror("Error", f"An error occurred:\n{e}")
        finally:
            self.after(0, lambda: self.run_button.config(state="normal"))
            self.after(0, lambda: self.cancel_button.config(state='disabled'))
            if self.cancel_flag:
                self.log("Operation was cancelled.")

    def _read_raster(self, path):
        with rasterio.open(path) as src:
            data = src.read(1, masked=True)
            profile = src.profile
        return data, profile

if __name__ == "__main__":
    app = LandslideAnalyzerApp()
    app.mainloop()