# Release Notes - November 21, 2025

## New Features

### 1. Category-Specific Analysis
- **Granular Heatmaps**: The aggregator now generates separate heatmap CSVs for each category defined in `prompts_categories.csv`.
- **Output Location**: These files are saved in `output/specifics/` (e.g., `heatmap_Arithmetic_20251121.csv`).
- **Automatic Graphing**: Line charts are automatically generated for each category-specific heatmap in `output/graphs/specifics/`.

### 2. Enhanced Derivative & Trend Analysis
- **Multi-Selection Support**: The "Generate Heatmap Derivative" tool now allows selecting multiple input CSVs at once, including a "Select All" option.
- **New Analysis Metrics**:
    - **Derivative**: Rate of change in accuracy vs temperature (`*_derivative.csv`).
    - **Percent Change**: Percentage change in accuracy compared to the previous temperature point (`*_pct_change.csv`).
    - **Trend Analysis**: Linear regression analysis calculating Slope, Intercept, Trend Direction (Positive/Negative/Neutral), and R-Squared (`*_trends.csv`).

### 3. Automated Graph Generation
- **Visualizations**: Integrated `matplotlib` to automatically generate charts for all analysis outputs.
- **Graph Types**:
    - **Line Charts**: For Accuracy, Derivatives, and Percent Change vs Temperature.
    - **Bar Charts**: For Trend Analysis (Slope per Model), color-coded by direction (Green=Positive, Red=Negative).
- **Organization**: Graphs are organized in `output/graphs/` with subdirectories:
    - `main/`
    - `specifics/`
    - `derivatives/`
    - `pct_change/`
    - `trends/`

## Improvements & Bug Fixes

- **UI Stability**: Fixed `IndexError` crashes in interactive dialogs caused by mouse interactions. Added error handling to gracefully fallback to keyboard navigation.
- **Robustness**: Improved CSV parsing to handle empty rows and malformed headers without crashing.
- **ID Parsing**: Enhanced logic to correctly extract Question IDs from filenames during aggregation.

## Dependencies
- Added `matplotlib` to `requirements.txt`. Please run `pip install -r requirements.txt` to update your environment.
