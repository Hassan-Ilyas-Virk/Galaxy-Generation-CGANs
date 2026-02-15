# Data Usage and Processing

This document outlines how data is structured, processed, and used in the Galaxy Generation project, including specific file formats and logic.

## 1. Data Sources & Formats

The project relies on the **Galaxy Zoo 2** dataset.

### A. Main Data File: `gz2_hart16.csv`
- **File**: `data/gz2_hart16.csv`
- **Format**: **CSV** (Comma-Separated Values)
- **Description**: This is the primary dataset containing morphological classifications for ~300,000 galaxies. It aggregates votes from citizen scientists who classified galaxy images.
- **Key Columns**:
    - `dr7objid`: Unique Object ID from the Sloan Digital Sky Survey (SDSS).
    - `t01_smooth_or_features...`: Votes for "Is the galaxy smooth or does it have features?"
    - `t02_edgeon...`: Votes for "Is it edge-on?"
    - `t08_odd_feature...`: Votes for "Is there anything odd?" (e.g., merger).
    - `total_votes`: Total number of people who classified this image.

**Sample Lines:**
```csv
dr7objid,ra,dec,rastring,decstring,sample,gz2_class,total_classifications,total_votes,t01_smooth_or_features_a01_smooth_count,t01_smooth_or_features_a01_smooth_weight,t01_smooth_or_features_a01_smooth_fraction,t01_smooth_or_features_a01_smooth_weighted_fraction,...
587722981736120347,195.45,14.23,13:01:48.00,+14:13:48.0,original,Er,44,44,25,25.0,0.568,0.568,...
587722981736579107,195.78,14.45,13:03:07.20,+14:27:00.0,original,Sb,42,42,4,4.0,0.095,0.095,...
```

### B. Mapping File: `gz2_filename_mapping.csv`
- **File**: `data/gz2_filename_mapping.csv`
- **Format**: **CSV**
- **Description**: Maps the long `dr7objid` to the simple integer filenames used in the image directory.
- **Key Columns**:
    - `objid`: Corresponds to `dr7objid` in the main CSV.
    - `asset_id`: The filename (without extension) of the image (e.g., `1` means `1.jpg`).

**Sample Lines:**
```csv
objid,sample,asset_id
587722981736120347,original,1
587722981736579107,original,2
587722981741363294,original,3
```

### C. Image Data
- **Directory**: `data/images_gz2/images`
- **Format**: **JPEG** (`.jpg`)
- **Description**: The actual galaxy images. Filenames correspond to `asset_id` (e.g., `1.jpg`, `2.jpg`).

## 2. Logic: How Values Identify Things

We use the vote fractions in `gz2_hart16.csv` to classify galaxies and extract physical attributes.

### Morphology Classification Logic
We define 4 mutually exclusive classes. The logic checks conditions in this specific order of priority:

1.  **Merger**:
    - **Logic**: If `t08_odd_feature_a24_merger_weighted_fraction > 0.4`
    - **Meaning**: More than 40% of voters saw a merger/disturbance.

2.  **Edge-on**:
    - **Logic**: (Not Merger) AND `t02_edgeon_a04_yes_weighted_fraction > 0.5`
    - **Meaning**: More than 50% of voters said the disk is viewed edge-on.

3.  **Elliptical**:
    - **Logic**: (Not Merger/Edge-on) AND `smooth_weighted_fraction > 0.6` AND `smooth > features`
    - **Meaning**: Majority said it's smooth (no features) and smooth votes outweigh feature votes.

4.  **Spiral**:
    - **Logic**: (Not Merger/Edge-on) AND `features_weighted_fraction > 0.5`
    - **Meaning**: Majority said it has features/disk.

### Physical Attribute Logic
We derive continuous values from the votes to simulate physical properties:

1.  **Size**:
    - **Source**: `total_votes` (Proxy for brightness/size/prominence)
    - **Logic**: Normalized to [0.3, 1.0] range.
    - **Formula**: `(votes - min) / (max - min) * 0.7 + 0.3`

2.  **Brightness**:
    - **Source**: `smooth_weighted_fraction`
    - **Logic**: Smooth galaxies tend to be brighter/more concentrated in this dataset context.
    - **Formula**: `smooth_frac * 0.8 + 0.2`

3.  **Ellipticity**:
    - **Source**: Shape votes (Round vs. Cigar-shaped)
    - **Logic**: Weighted sum of shape probabilities.
    - **Formula**: `0.0*Round + 0.5*InBetween + 0.9*Cigar`

4.  **Redshift**:
    - **Source**: Simulated (Random)
    - **Logic**: Uniform random value between [0.0, 0.5].

## 3. Download Links

You can download the original datasets here:

-   **Main CSV (classifications)**: [Galaxy Zoo 2 - Hart et al. 2016](https://data.galaxyzoo.org/)
    -   Look for "Table 1" or the main GZ2 catalog.
-   **Images**: [Zenodo - Galaxy Zoo 2 Images](https://zenodo.org/record/3565489)
    -   "Galaxy Zoo 2: Images from Original Sample"

*Note: In this project, we use a subset or processed version of these files.*
