# Red Zone Yellow Zone Applications

## Overview

This project contains three Python applications for landslide hazard zonation, based on the methodologies described in the project manual. Each application is a graphical user interface (GUI) tool built with Tkinter.

## Applications

*   **App 1: Slope Failure (`App1_SlopeFailure.py`)**: Implements the Slope Failure methodology, where zonation is determined by the height (H) of the potential source area.
*   **App 2: Slide (`App2_Slide.py`)**: Implements the Slide methodology, where zonation is determined by the length (L) of the landslide block.
*   **App 3: Debris Flow (`App3_DebrisFlow -Onlywith lateeral spreding.py`)**: Implements the Debris Flow methodology, where zonation is determined by ground gradient and lateral spreading.

## Requirements

These applications require the following Python libraries:

*   `numpy`
*   `rasterio`
*   `scipy`
*   `numba`

## How to Run

1.  **Install Dependencies:**

    You can install the required libraries using pip:

    ```bash
    pip install numpy rasterio scipy numba
    ```

2.  **Run an Application:**

    To run any of the applications, execute the corresponding Python script from your terminal:

    ```bash
    # To run the Slope Failure application
    python "App1_SlopeFailure.py"

    # To run the Slide application
    python "App2_Slide.py"

    # To run the Debris Flow application
    python "App3_DebrisFlow -Onlywith lateeral spreding.py"
    ```

## Input Data

Each application requires specific input data in the form of GeoTIFF (`.tif`) files and, for the Debris Flow app, a CSV file. The GUI for each application will prompt you for the required files.

**Note:** As requested, the data files (e.g., DEM, Flow Direction, LHZM, etc.) are not included in this repository. You will need to provide your own data to run the applications.
