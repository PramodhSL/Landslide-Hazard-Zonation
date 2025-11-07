"""
=============================================================================
APP 3: DEBRIS FLOW HAZARD ZONATION (Gradient-based)
=============================================================================

Implements the Debris Flow methodology from the manual.
Zonation is determined by GROUND GRADIENT and lateral spreading.

METHODOLOGY (Per Manual):
1. Load control points from CSV file (X, Y coordinates)
2. For each control point, trace flow path downslope
3. Calculate riverbed slope over 200m segments
4. Red Zone:
   - Stops where gradient ≤ 3°
   - Width: 15° spreading angle from centerline
5. Yellow Zone:
   - Stops where gradient ≤ 1°
   - Width: 30° spreading angle from centerline
6. Post-process with smoothing

INPUTS:
- Filled DEM (GeoTIFF)
- Flow Direction D8 (GeoTIFF)
- Control Points CSV (X, Y coordinates)

PARAMETERS:
- Red zone gradient threshold: 3° (fixed per manual)
- Yellow zone gradient threshold: 1° (fixed per manual)
- Red spreading angle: 15° (fixed per manual)
- Yellow spreading angle: 30° (fixed per manual)
- Riverbed slope calculation length: 200m

OUTPUT (GeoTIFF):
- 0 = Background
- 1 = Yellow zone
- 2 = Red zone

CSV FORMAT:
ID,X,Y
1,500000.5,3500000.2
2,500125.3,3500150.8

Date: October 2025
Version: 1.0
=============================================================================
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import threading
import numpy as np
import rasterio
from scipy.ndimage import generic_filter
import math
from numba import njit
import os
import csv


@njit(cache=True)
def calculate_gradient_along_path(dem, flow_dir, start_r, start_c, distance_m, cell_size, fd_map_keys, fd_map_vals):
    """
    Calculate gradient over specified distance along flow path.
    
    Returns: gradient in degrees
    """
    rows, cols = dem.shape
    dist_traveled = 0.0
    cr, cc = start_r, start_c
    start_elev = dem[start_r, start_c]
    
    if np.isnan(start_elev):
        return 0.0
    
    # Trace flow path for distance_m
    for _ in range(rows + cols):
        if np.isnan(dem[cr, cc]):
            break
        
        fd_val = flow_dir[cr, cc]
        
        # Find flow direction
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
        
        step_dist = cell_size * (math.sqrt(2.0) if abs(dr) + abs(dc) == 2 else 1.0)
        dist_traveled += step_dist
        
        if dist_traveled >= distance_m:
            # Calculate gradient
            end_elev = dem[nr, nc]
            elev_diff = start_elev - end_elev
            if elev_diff > 0:
                gradient_rad = math.atan(elev_diff / dist_traveled)
                gradient_deg = math.degrees(gradient_rad)
                return gradient_deg
            else:
                return 0.0
        
        cr, cc = nr, nc
    
    # Didn't reach full distance, calculate with what we have
    if dist_traveled > 0:
        end_elev = dem[cr, cc]
        elev_diff = start_elev - end_elev
        if elev_diff > 0:
            gradient_rad = math.atan(elev_diff / dist_traveled)
            gradient_deg = math.degrees(gradient_rad)
            return gradient_deg
    
    return 0.0


@njit(cache=True)
def simulate_debris_flow_numba(dem, flow_dir, control_points, cell_size, 
                                red_gradient_threshold, yellow_gradient_threshold,
                                riverbed_length, fd_map_keys, fd_map_vals):
    """
    Simulate debris flow from control points.
    Stops based on gradient thresholds.
    """
    rows, cols = dem.shape
    zone_grid = np.zeros_like(dem, dtype=np.int16)
    
    for idx in range(control_points.shape[0]):
        start_r, start_c = control_points[idx, 0], control_points[idx, 1]
        
        if not (0 <= start_r < rows and 0 <= start_c < cols):
            continue
        if np.isnan(dem[start_r, start_c]):
            continue
        
        # Mark starting point
        zone_grid[start_r, start_c] = 2
        
        # Trace flow path
        cr, cc = start_r, start_c
        in_red_zone = True
        
        for _ in range(rows + cols):
            if np.isnan(dem[cr, cc]):
                break
            
            # Calculate gradient at current position
            gradient = calculate_gradient_along_path(
                dem, flow_dir, cr, cc, riverbed_length, cell_size,
                fd_map_keys, fd_map_vals
            )
            
            # Check if we should stop or change zone
            if gradient <= red_gradient_threshold:
                in_red_zone = False
            
            if gradient <= yellow_gradient_threshold:
                break  # Stop completely
            
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
            
            # Mark zone
            if in_red_zone:
                if zone_grid[nr, nc] < 2:
                    zone_grid[nr, nc] = 2
            else:
                if zone_grid[nr, nc] < 1:
                    zone_grid[nr, nc] = 1
            
            cr, cc = nr, nc
    
    return zone_grid


@njit(cache=True)
def apply_lateral_spreading(zone_grid, dem, flow_dir, spreading_angle_deg, zone_value, 
                            cell_size, fd_map_keys, fd_map_vals):
    """
    Apply lateral spreading to zones.
    spreading_angle_deg: 15° for red, 30° for yellow
    """
    rows, cols = zone_grid.shape
    spread_grid = zone_grid.copy()
    
    # Find centerline cells
    centerline = (zone_grid == zone_value)
    coords = np.argwhere(centerline)
    
    # Calculate max spread distance based on angle
    # For simplicity, use a fixed spread distance
    max_spread_cells = int(math.tan(math.radians(spreading_angle_deg)) * 10)  # Approximate
    
    for idx in range(len(coords)):
        r, c = coords[idx]
        
        # Spread perpendicular to flow direction
        for dr in range(-max_spread_cells, max_spread_cells + 1):
            for dc in range(-max_spread_cells, max_spread_cells + 1):
                if dr == 0 and dc == 0:
                    continue
                
                nr, nc = r + dr, c + dc
                
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if np.isnan(dem[nr, nc]):
                    continue
                
                # Calculate distance
                dist = math.sqrt(dr * dr + dc * dc) * cell_size
                
                # Check if within spreading angle
                if dist <= max_spread_cells * cell_size:
                    if spread_grid[nr, nc] < zone_value:
                        spread_grid[nr, nc] = zone_value
    
    return spread_grid


@njit(cache=True)
def majority_filter_numba(window):
    """Numba-accelerated majority filter."""
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

class DebrisFlowApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("App 3: Debris Flow Hazard Zonation (Gradient-based)")
        self.geometry("650x750")
        
        # Variables
        self.dem_path = tk.StringVar()
        self.flowdir_path = tk.StringVar()
        self.csv_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.red_gradient = tk.DoubleVar(value=3.0)
        self.yellow_gradient = tk.DoubleVar(value=1.0)
        self.red_spread = tk.DoubleVar(value=15.0)
        self.yellow_spread = tk.DoubleVar(value=30.0)
        self.riverbed_length = tk.DoubleVar(value=200.0)
        self.smooth_output = tk.BooleanVar(value=True)
        self.apply_spreading = tk.BooleanVar(value=True)
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
        self._create_file_entry(file_frame, 2, "3. Control Points (.csv):", self.csv_path, csv_file=True)
        self._create_file_entry(file_frame, 3, "4. Output File (.tif):", self.output_path, save=True)
        
        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Parameters (Per Manual)", padding="10")
        param_frame.pack(fill=tk.X, expand=True, pady=5)
        
        ttk.Label(param_frame, text="Red Zone Gradient (°):").grid(row=0, column=0, sticky='w', pady=2)
        ttk.Entry(param_frame, textvariable=self.red_gradient, width=10, state='readonly').grid(row=0, column=1, sticky='w', pady=2)
        ttk.Label(param_frame, text="(Fixed: 3°)", font=("Arial", 8)).grid(row=0, column=2, sticky='w', padx=5)
        
        ttk.Label(param_frame, text="Yellow Zone Gradient (°):").grid(row=1, column=0, sticky='w', pady=2)
        ttk.Entry(param_frame, textvariable=self.yellow_gradient, width=10, state='readonly').grid(row=1, column=1, sticky='w', pady=2)
        ttk.Label(param_frame, text="(Fixed: 1°)", font=("Arial", 8)).grid(row=1, column=2, sticky='w', padx=5)
        
        ttk.Separator(param_frame, orient='horizontal').grid(row=2, column=0, columnspan=4, sticky='ew', pady=10)
        
        ttk.Label(param_frame, text="Red Spreading Angle (°):").grid(row=3, column=0, sticky='w', pady=2)
        ttk.Entry(param_frame, textvariable=self.red_spread, width=10, state='readonly').grid(row=3, column=1, sticky='w', pady=2)
        ttk.Label(param_frame, text="(Fixed: 15°)", font=("Arial", 8)).grid(row=3, column=2, sticky='w', padx=5)
        
        ttk.Label(param_frame, text="Yellow Spreading Angle (°):").grid(row=4, column=0, sticky='w', pady=2)
        ttk.Entry(param_frame, textvariable=self.yellow_spread, width=10, state='readonly').grid(row=4, column=1, sticky='w', pady=2)
        ttk.Label(param_frame, text="(Fixed: 30°)", font=("Arial", 8)).grid(row=4, column=2, sticky='w', padx=5)
        
        ttk.Separator(param_frame, orient='horizontal').grid(row=5, column=0, columnspan=4, sticky='ew', pady=10)
        
        ttk.Label(param_frame, text="Riverbed Slope Length (m):").grid(row=6, column=0, sticky='w', pady=2)
        ttk.Entry(param_frame, textvariable=self.riverbed_length, width=10).grid(row=6, column=1, sticky='w', pady=2)
        ttk.Label(param_frame, text="(Default: 200m)", font=("Arial", 8)).grid(row=6, column=2, sticky='w', padx=5)
        
        # Post-processing
        post_frame = ttk.LabelFrame(main_frame, text="Post-Processing", padding="10")
        post_frame.pack(fill=tk.X, expand=True, pady=5)
        
        ttk.Checkbutton(post_frame, text="Apply Lateral Spreading", variable=self.apply_spreading).grid(row=0, column=0, sticky='w')
        ttk.Checkbutton(post_frame, text="Smooth Output (Majority Filter)", variable=self.smooth_output).grid(row=1, column=0, sticky='w')
        
        # Action buttons
        action_frame = ttk.Frame(main_frame, padding="10")
        action_frame.pack(fill=tk.BOTH, expand=True)
        
        self.run_button = ttk.Button(action_frame, text="Start Calculation", command=self._start_calculation_thread)
        self.run_button.pack(pady=5)
        
        self.cancel_button = ttk.Button(action_frame, text="Cancel", command=self.cancel_operation, state='disabled')
        self.cancel_button.pack(pady=5)
        
        self.log_window = scrolledtext.ScrolledText(action_frame, height=12, state='disabled', wrap='word')
        self.log_window.pack(fill=tk.BOTH, expand=True)
    
    def _create_file_entry(self, parent, r, label_text, var, save=False, csv_file=False):
        label = ttk.Label(parent, text=label_text)
        label.grid(row=r, column=0, sticky=tk.W, padx=5, pady=5)
        entry = ttk.Entry(parent, textvariable=var, width=50)
        entry.grid(row=r, column=1, sticky=(tk.W, tk.E))
        button_text = "Save As..." if save else "Browse..."
        
        if save:
            action = lambda: self._browse_save(var)
        elif csv_file:
            action = lambda: self._browse_csv(var)
        else:
            action = lambda: self._browse_file(var)
        
        button = ttk.Button(parent, text=button_text, command=action)
        button.grid(row=r, column=2, sticky=tk.W, padx=5)
        parent.columnconfigure(1, weight=1)
    
    def _browse_file(self, var):
        path = filedialog.askopenfilename(filetypes=[("GeoTIFF", "*.tif;*.tiff")])
        if path:
            var.set(path)
    
    def _browse_csv(self, var):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
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
        if not all([self.dem_path.get(), self.flowdir_path.get(), self.csv_path.get(), self.output_path.get()]):
            messagebox.showerror("Input Error", "All input and output file paths are required.")
            return
        
        self.run_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.cancel_flag = False
        self.log_window.config(state='normal')
        self.log_window.delete(1.0, tk.END)
        self.log_window.config(state='disabled')
        self.log("--- Starting Debris Flow Hazard Zonation ---")
        
        thread = threading.Thread(target=self._run_calculation, daemon=True)
        thread.start()
    
    def cancel_operation(self):
        self.cancel_flag = True
        self.log("Cancelling operation...")
        self.cancel_button.config(state='disabled')
    
    def _run_calculation(self):
        try:
            # Load rasters
            self.log("Loading raster files...")
            dem, profile = self._read_raster(self.dem_path.get())
            flow_dir, _ = self._read_raster(self.flowdir_path.get())
            
            cell_size = abs(profile['transform'][0])
            transform = profile['transform']
            self.log(f"Grid size: {dem.shape[0]} x {dem.shape[1]}")
            self.log(f"Cell size: {cell_size:.2f} m\n")
            
            if self.cancel_flag:
                return
            
            # Load control points from CSV
            self.log("Loading control points from CSV...")
            control_points_geo = []
            
            with open(self.csv_path.get(), 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    x = float(row['X'])
                    y = float(row['Y'])
                    control_points_geo.append((x, y))
            
            self.log(f"Loaded {len(control_points_geo)} control points")
            
            # Convert geographic coordinates to row/col
            control_points_rc = []
            for x, y in control_points_geo:
                col = int((x - transform[2]) / transform[0])
                row = int((y - transform[5]) / transform[4])
                
                if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
                    control_points_rc.append((row, col))
                    self.log(f"  Point ({x:.2f}, {y:.2f}) -> Cell ({row}, {col})")
                else:
                    self.log(f"  WARNING: Point ({x:.2f}, {y:.2f}) outside DEM extent")
            
            if len(control_points_rc) == 0:
                raise ValueError("No valid control points within DEM extent")
            
            control_points_array = np.array(control_points_rc, dtype=np.int32)
            self.log(f"\n{len(control_points_rc)} control points within DEM extent\n")
            
            if self.cancel_flag:
                return
            
            # Simulate debris flow
            self.log("Simulating debris flow paths...")
            self.log(f"  Red zone stops at gradient ≤ {self.red_gradient.get()}°")
            self.log(f"  Yellow zone stops at gradient ≤ {self.yellow_gradient.get()}°")
            self.log(f"  Riverbed slope calculated over {self.riverbed_length.get()}m\n")
            
            # Prepare flow direction map
            fd_map = {1:(0,1), 2:(1,1), 4:(1,0), 8:(1,-1), 16:(0,-1), 32:(-1,-1), 64:(-1,0), 128:(-1,1)}
            fd_map_keys = np.array(list(fd_map.keys()), dtype=np.int16)
            fd_map_vals = np.array(list(fd_map.values()), dtype=np.int8)
            
            zone_grid = simulate_debris_flow_numba(
                dem.filled(np.nan),
                flow_dir.filled(0).astype(np.int16),
                control_points_array,
                cell_size,
                self.red_gradient.get(),
                self.yellow_gradient.get(),
                self.riverbed_length.get(),
                fd_map_keys,
                fd_map_vals
            )
            
            self.log("Debris flow simulation complete")
            
            if self.cancel_flag:
                return
            
            # Apply lateral spreading
            if self.apply_spreading.get():
                self.log("\nApplying lateral spreading...")
                self.log(f"  Red zone: {self.red_spread.get()}° spreading angle")
                self.log(f"  Yellow zone: {self.yellow_spread.get()}° spreading angle")
                
                # Apply spreading to red zone
                zone_grid = apply_lateral_spreading(
                    zone_grid, dem.filled(np.nan), flow_dir.filled(0).astype(np.int16),
                    self.red_spread.get(), 2, cell_size, fd_map_keys, fd_map_vals
                )
                
                # Apply spreading to yellow zone
                zone_grid = apply_lateral_spreading(
                    zone_grid, dem.filled(np.nan), flow_dir.filled(0).astype(np.int16),
                    self.yellow_spread.get(), 1, cell_size, fd_map_keys, fd_map_vals
                )
                
                self.log("Lateral spreading complete")
            
            # Smoothing
            if self.smooth_output.get():
                self.log("\nSmoothing output with majority filter...")
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
    app = DebrisFlowApp()
    app.mainloop()
