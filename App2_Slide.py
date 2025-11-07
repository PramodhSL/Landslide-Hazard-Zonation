"""
=============================================================================
APP 2: SLIDE HAZARD ZONATION (Length-based)
=============================================================================

Implements the Slide methodology from the manual.
Zonation is determined by the LENGTH (L) of the landslide block.

METHODOLOGY (Per Manual):
1. Input: Mapped slide blocks (binary raster: 1=slide, 0=background)
2. Input: Slide length (L) measured from scarp to toe
3. Trace downslope from slide blocks using D8 flow direction
4. Rank A (clearly defined slide):
   - Slide block itself = Red Zone
   - 0 to 0.5L downslope = Red Zone (max 100m total)
   - 0.5L to 1L downslope = Yellow Zone (max 250m total)
5. Rank B/C (poorly defined slide):
   - Slide block itself = Yellow Zone
   - 0 to 1L downslope = Yellow Zone (max 250m total)
   - No Red Zone
6. Post-process with smoothing

INPUTS (GeoTIFF):
- Filled DEM
- Flow Direction (D8)
- Slide Block Raster (Binary: 1=slide block, 0=background)

PARAMETERS:
- Slide Rank: A (clearly defined) or B/C (poorly defined)
- Slide Length (L): meters (measured from scarp to toe)

OUTPUT (GeoTIFF):
- 0 = Background
- 1 = Yellow zone
- 2 = Red zone (Rank A only)

NOTE: This is for EXISTING mapped slides, not hazard zones.
Use App1 (Slope Failure) for LHZM hazard zonation.

Author: Created for manual compliance
Date: November 2025
Version: 2.0 (Fixed to use binary slide blocks)
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


@njit(cache=True)
def simulate_slide_runout_numba(dem, flow_dir, slide_blocks, slide_length_m, rank, cell_size, fd_map_keys, fd_map_vals):
    """
    Simulate runout from slide blocks based on slide length (L).
    
    Rank A: Red = 0.5L (max 100m), Yellow = 1L (max 250m)
    Rank B/C: Yellow only = 1L (max 250m)
    """
    rows, cols = dem.shape
    zone_grid = np.zeros_like(dem, dtype=np.int16)
    
    # Mark slide blocks
    slide_coords = np.argwhere(slide_blocks)
    for idx in range(len(slide_coords)):
        r, c = slide_coords[idx]
        if rank == 1:  # Rank A
            zone_grid[r, c] = 2  # Slide block is red
        else:  # Rank B or C
            zone_grid[r, c] = 1  # Slide block is yellow
    
    # Calculate runout distances based on rank
    if rank == 1:  # Rank A
        red_dist_m = min(0.5 * slide_length_m, 100.0)
        yellow_dist_m = min(1.0 * slide_length_m, 250.0)
    else:  # Rank B or C
        red_dist_m = 0.0  # No red zone
        yellow_dist_m = min(1.0 * slide_length_m, 250.0)
    
    # Simulate runout from each slide block cell
    for idx in range(len(slide_coords)):
        r, c = slide_coords[idx]
        
        dist_traveled = 0.0
        cr, cc = r, c
        
        for _ in range(rows + cols):
            if np.isnan(dem[cr, cc]):
                break
            
            # Get flow direction
            fd_val = flow_dir[cr, cc]
            
            fd_idx = -1
            for i in range(len(fd_map_keys)):
                if fd_map_keys[i] == fd_val:
                    fd_idx = i
                    break
            
            if fd_idx == -1:
                break
            
            dr, dc = fd_map_vals[fd_idx]
            nr, nc = cr + dr, cc + dc
            
            if not (0 <= nr < rows and 0 <= nc < cols) or np.isnan(dem[nr, nc]):
                break
            
            # Calculate distance
            step_dist = cell_size * (math.sqrt(2.0) if abs(dr) + abs(dc) == 2 else 1.0)
            dist_traveled += step_dist
            
            # Assign zone based on distance and rank
            if rank == 1 and dist_traveled <= red_dist_m:  # Rank A red zone
                if zone_grid[nr, nc] < 2:
                    zone_grid[nr, nc] = 2
            elif dist_traveled <= yellow_dist_m:  # Yellow zone
                if zone_grid[nr, nc] < 1:
                    zone_grid[nr, nc] = 1
            else:
                break  # Beyond yellow zone
            
            cr, cc = nr, nc
    
    return zone_grid


@njit(cache=True)
def majority_filter_numba(window):
    """Numba-accelerated majority filter for smoothing."""
    count_1 = 0
    count_2 = 0
    for i in range(window.size):
        val = window[i]
        if val == 1:
            count_1 += 1
        elif val == 2:
            count_2 += 1
    
    if count_2 > count_1:
        return 2
    elif count_1 > count_2:
        return 1
    
    center_val = window[4]
    if center_val == 1 or center_val == 2:
        return center_val
    else:
        return 0


# --- GUI APPLICATION ---

class SlideAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("App 2: Slide Hazard Zonation (Length-based)")
        self.geometry("650x650")
        
        # Variables
        self.dem_path = tk.StringVar()
        self.flowdir_path = tk.StringVar()
        self.slide_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.slide_length = tk.DoubleVar(value=100.0)
        self.slide_rank = tk.StringVar(value="A")
        self.min_area = tk.DoubleVar(value=0.0)
        self.smooth_output = tk.BooleanVar(value=True)
        self.cancel_flag = False
        
        self._build_gui()
    
    def _build_gui(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File inputs
        file_frame = ttk.LabelFrame(main_frame, text="Input/Output Files", padding="10")
        file_frame.pack(fill=tk.X, expand=True, pady=5)
        
        self._create_file_entry(file_frame, 0, "1. Filled DEM (.tif):", self.dem_path)
        self._create_file_entry(file_frame, 1, "2. Flow Direction (.tif):", self.flowdir_path)
        self._create_file_entry(file_frame, 2, "3. Slide Block Raster (.tif):", self.slide_path)
        self._create_file_entry(file_frame, 3, "4. Output File (.tif):", self.output_path, save=True)
        
        # Info label
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=5)
        info_label = ttk.Label(info_frame, text="Note: Slide Block Raster should be binary (1=slide block, 0=background)", 
                              foreground="blue", font=('TkDefaultFont', 9))
        info_label.pack()
        
        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Slide Parameters", padding="10")
        param_frame.pack(fill=tk.X, expand=True, pady=5)
        
        # Slide rank
        ttk.Label(param_frame, text="Slide Rank:").grid(row=0, column=0, sticky='w', pady=5)
        rank_frame = ttk.Frame(param_frame)
        rank_frame.grid(row=0, column=1, columnspan=3, sticky='w', pady=5)
        ttk.Radiobutton(rank_frame, text="A (Clearly defined)", variable=self.slide_rank, value="A").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(rank_frame, text="B/C (Poorly defined)", variable=self.slide_rank, value="B").pack(side=tk.LEFT, padx=5)
        
        # Slide length
        ttk.Label(param_frame, text="Slide Length L (m):").grid(row=1, column=0, sticky='w', pady=5)
        ttk.Entry(param_frame, textvariable=self.slide_length, width=10).grid(row=1, column=1, sticky='w', pady=5)
        ttk.Label(param_frame, text="(Distance from scarp to toe)", font=("Arial", 8)).grid(row=1, column=2, columnspan=2, sticky='w', padx=5)
        
        # Info labels
        info_frame = ttk.LabelFrame(param_frame, text="Zone Distances (Per Manual)", padding="5")
        info_frame.grid(row=2, column=0, columnspan=4, sticky='ew', pady=10)
        
        ttk.Label(info_frame, text="Rank A:", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky='w')
        ttk.Label(info_frame, text="Red = 0.5L (max 100m), Yellow = 1L (max 250m)").grid(row=0, column=1, sticky='w', padx=5)
        
        ttk.Label(info_frame, text="Rank B/C:", font=("Arial", 9, "bold")).grid(row=1, column=0, sticky='w')
        ttk.Label(info_frame, text="Yellow only = 1L (max 250m)").grid(row=1, column=1, sticky='w', padx=5)
        
        # Post-processing
        post_frame = ttk.LabelFrame(main_frame, text="Post-Processing", padding="10")
        post_frame.pack(fill=tk.X, expand=True, pady=5)
        
        ttk.Checkbutton(post_frame, text="Smooth Output (Majority Filter)", variable=self.smooth_output).grid(row=0, column=0, sticky='w')
        ttk.Label(post_frame, text="Min Area (m²):").grid(row=0, column=1, sticky='w', padx=10)
        ttk.Entry(post_frame, textvariable=self.min_area, width=10).grid(row=0, column=2, sticky='w')
        
        # Action buttons
        action_frame = ttk.Frame(main_frame, padding="10")
        action_frame.pack(fill=tk.BOTH, expand=True)
        
        self.run_button = ttk.Button(action_frame, text="Start Calculation", command=self._start_calculation_thread)
        self.run_button.pack(pady=5)
        
        self.cancel_button = ttk.Button(action_frame, text="Cancel", command=self.cancel_operation, state='disabled')
        self.cancel_button.pack(pady=5)
        
        self.log_window = scrolledtext.ScrolledText(action_frame, height=12, state='disabled', wrap='word')
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
        if path:
            var.set(path)
    
    def _browse_save(self, var):
        path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("GeoTIFF", "*.tif;*.tiff")])
        if path:
            var.set(path)
    
    def log(self, message: str):
        def _append():
            self.log_window.config(state='normal')
            self.log_window.insert(tk.END, message + "\n")
            self.log_window.config(state='disabled')
            self.log_window.see(tk.END)
        self.after(0, _append)
    
    def _start_calculation_thread(self):
        if not all([self.dem_path.get(), self.flowdir_path.get(), self.slide_path.get(), self.output_path.get()]):
            messagebox.showerror("Input Error", "All input and output file paths are required.")
            return
        
        self.run_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.cancel_flag = False
        self.log_window.config(state='normal')
        self.log_window.delete(1.0, tk.END)
        self.log_window.config(state='disabled')
        self.log("--- Starting Slide Hazard Zonation ---")
        
        thread = threading.Thread(target=self._run_calculation, daemon=True)
        thread.start()
    
    def cancel_operation(self):
        self.cancel_flag = True
        self.log("Cancelling operation...")
        self.cancel_button.config(state='disabled')
    
    def _run_calculation(self):
        try:
            # Load data
            self.log("Loading raster files...")
            dem, profile = self._read_raster(self.dem_path.get())
            flow_dir, _ = self._read_raster(self.flowdir_path.get())
            slide_raster, _ = self._read_raster(self.slide_path.get())
            
            if self.cancel_flag:
                return
            
            cell_size = abs(profile['transform'][0])
            self.log(f"Grid size: {dem.shape[0]} x {dem.shape[1]}")
            self.log(f"Cell size: {cell_size:.2f} m\n")
            
            # Identify slide blocks (binary: 1=slide, 0=background)
            self.log("Identifying slide blocks from binary raster...")
            slide_blocks = (slide_raster.filled(0) == 1) & (~dem.mask)
            
            num_cells = np.sum(slide_blocks)
            if num_cells == 0:
                raise ValueError("No slide blocks found. Ensure raster has value 1 for slide blocks.")
            self.log(f"Found {num_cells} cells in slide blocks")
            
            # Calculate slide block area
            cell_area = cell_size * cell_size
            total_area = num_cells * cell_area
            self.log(f"Total slide block area: {total_area:.1f} m²\n")
            
            if self.cancel_flag:
                return
            
            # Get parameters
            slide_length = self.slide_length.get()
            rank_str = self.slide_rank.get()
            rank = 1 if rank_str == "A" else (2 if rank_str == "B" else 3)
            
            self.log(f"Slide Length (L): {slide_length} m")
            self.log(f"Slide Rank: {rank_str}")
            
            if rank == 1:
                self.log(f"  Red Zone: 0.5L = {min(0.5 * slide_length, 100.0):.1f} m (max 100m)")
                self.log(f"  Yellow Zone: 1L = {min(slide_length, 250.0):.1f} m (max 250m)")
            else:
                self.log(f"  Yellow Zone only: 1L = {min(slide_length, 250.0):.1f} m (max 250m)")
                self.log(f"  No Red Zone (Rank {rank_str})")
            
            # Simulate runout
            self.log("\nSimulating runout using Numba JIT compiler...")
            
            # Prepare flow direction map
            fd_map = {1:(0,1), 2:(1,1), 4:(1,0), 8:(1,-1), 16:(0,-1), 32:(-1,-1), 64:(-1,0), 128:(-1,1)}
            fd_map_keys = np.array(list(fd_map.keys()), dtype=np.int16)
            fd_map_vals = np.array(list(fd_map.values()), dtype=np.int8)
            
            zone_grid = simulate_slide_runout_numba(
                dem.filled(np.nan),
                flow_dir.filled(0).astype(np.int16),
                slide_blocks,
                slide_length,
                rank,
                cell_size,
                fd_map_keys,
                fd_map_vals
            )
            
            self.log("Runout simulation complete")
            
            if self.cancel_flag:
                return
            
            # Area filtering
            if self.min_area.get() > 0:
                self.log(f"\nFiltering zones smaller than {self.min_area.get()} m²...")
                min_pixels = self.min_area.get() / (cell_size * cell_size)
                for zone_val in [1, 2]:
                    labeled_array, _ = label(zone_grid == zone_val)
                    if labeled_array.max() == 0:
                        continue
                    unique_labels, counts = np.unique(labeled_array, return_counts=True)
                    small_labels = unique_labels[1:][counts[1:] < min_pixels]
                    zone_grid[np.isin(labeled_array, small_labels)] = 0
            
            # Smoothing
            if self.smooth_output.get():
                self.log("Smoothing output with majority filter...")
                zone_grid = generic_filter(zone_grid, majority_filter_numba, size=3)
                zone_grid = generic_filter(zone_grid, majority_filter_numba, size=3)
            
            if self.cancel_flag:
                return
            
            # Save output
            self.log(f"\nSaving output to {self.output_path.get()}...")
            out_dir = os.path.dirname(self.output_path.get())
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            
            profile.update(dtype=rasterio.int16, count=1, nodata=0, compress='lzw')
            with rasterio.open(self.output_path.get(), 'w', **profile) as dst:
                dst.write(zone_grid.astype(np.int16), 1)
            
            self.log("--- Calculation Finished Successfully! ---")
            self.after(0, lambda: messagebox.showinfo("Success", "Calculation completed!"))
        
        except Exception as e:
            self.log(f"--- ERROR: {e} ---")
            messagebox.showerror("Error", f"An error occurred:\n{e}")
        finally:
            self.after(0, lambda: self.run_button.config(state="normal"))
            self.after(0, lambda: self.cancel_button.config(state='disabled'))
    
    def _read_raster(self, path):
        with rasterio.open(path) as src:
            data = src.read(1, masked=True)
            profile = src.profile
        return data, profile


if __name__ == "__main__":
    app = SlideAnalyzerApp()
    app.mainloop()
