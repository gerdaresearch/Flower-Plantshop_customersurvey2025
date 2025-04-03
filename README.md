# Flower-Plantshop_customersurvey2025
# 🌸 Correspondence Analysis of Flower Preferences (Color Tone + Style/Emotion/Texture)

This repository contains Python scripts and documentation for conducting a correspondence analysis (CA) on consumer preferences for flowers and plants.  
The analysis is based on a survey conducted in **January 2025** as part of a government-supported initiative in Japan:

> **Survey Title:**  
> *"The Flower shop / garden center customer survey 2025: Purchasing behavior and customer evaluations".*  Council for Japanese Flower Production and Distribution Enhancement.  
> **Conducted under:**  
> Council for Japanese Flower Production and Distribution Enhancement. Funded by the FY2024 MAFF (Ministry of Agriculture, Forestry and Fisheries of Japan) project for strengthening sustainable domestic flower production and distribution.  
> **Project Lead:** Aoki Kyoko  
> **Sample:** n = 518  
> **Target:** Consumers who had used one of 9 flower/garden stores across 4 business types (florist, supermarket, home center, garden center) within the past year.

> 🔎 **Note:** The number of respondents for garden centers and individual stores is relatively small.  
> Results for these segments should be interpreted as **reference values only**.

Survey respondents were asked about their preferences regarding flowers and plants, including **color tone, style, texture, and emotional impressions**.  
The collected responses were analyzed using **correspondence analysis (CA)** in Python, segmented by **gender × age, business type, and store**.

### ✳️ Key methodological points:

- Decomposition of SD-style adjective pairs (7-point bipolar scales) into polarity and neutral flags  
- Separate handling of neutral responses (`score = 4`) due to the large number of mid-scale responders  
- Correspondence analysis was performed using only polarized preference data (excluding neutral scores), with neutral rates analyzed in parallel

> 🔒 **Note:** The raw data file is not included in this repository.  
>  
> 📊 **Full survey results and interpretation by the author (Aoki Kyoko) are available at:**  
> - Publications page: [https://gerdaresearch.github.io/publication](https://gerdaresearch.github.io/publication)  
> - Commentary and project archive: [https://gerdaresearch.github.io/year-archive/](https://gerdaresearch.github.io/year-archive/)

---

## 🔍 Objective

To visualize consumer preferences for flowers using correspondence analysis, and explore differences across demographics and stores based on color tone, style, emotion, and texture.

---

## 📂 Data Structure (Overview)

### Input Columns:
- **Q17_1–Q17_15**: Style/texture/emotion preferences (7-point bipolar SD scale)
- **Q15_1–Q15_5**: Color tone preferences (7-point scale)

### Row Groupings:
- **QSAGE**: Gender × Age (8 groups: 1 = Male 20s → 8 = Female 50s)
- **GATE**: Business Type (Florist, Supermarket, Home Center, Garden Store)
- **QBD**: Store (9 stores: e.g., Hibiya-Kadan, Ao Flora, etc.)

---

## ⚙️ Preprocessing Steps

### (1) Bipolar Decomposition and Neutral Flagging
- Split each item into left/right polarity scores and neutral flags
- Neutral (`score = 4`) is excluded from CA but stored for later analysis

### (2) Scoring Rules

| Scale | Left/Negative | Neutral | Right/Positive |
|-------|---------------|---------|----------------|
| 1     | 3 pts         |         |                |
| 2     | 2 pts         |         |                |
| 3     | 1 pt          |         |                |
| 4     |     → flag 1  | ✔️       |     → flag 1   |
| 5     |               |         | 1 pt           |
| 6     |               |         | 2 pts          |
| 7     |               |         | 3 pts          |

---

## 📊 Correspondence Analysis Procedure

### (1) Input Matrix
- Combine Q15 and Q17 transformed variables
- Stack category groups (QSAGE, GATE, QBD) vertically

### (2) Run CA
- Project into CA space using `prince.CA()`
- Extract axis coordinates, contributions, and inertia

### (3) Visualization
- Map CA axes with labeled row/column points
- Axis pairs: 1×2, 1×3, 2×3

---

## 📈 Statistical Indices from CA

| Metric              | Description                                        |
|---------------------|----------------------------------------------------|
| Explained Inertia   | Variance explained by each axis                   |
| Cumulative Inertia  | Total variance explained across selected axes     |
| Total Inertia       | Overall structural strength of the CA table       |

---

## 🔍 Additional Analyses

- **Variable contributions per axis**
- **Store–item structural distance (5D)**
- **Nearest/farthest item rankings per store**
- **Commonly distant & highly variable items**
- **Neutrality rates by age group and item**

---

## 🧷 Note on Usage

- This repository is shared for methodological reference.  
- The original dataset is not provided.  
- Users can adapt the code to their own survey data using similar structure.

---

## 📎 Contact

For further details:  
Visit [https://gerdaresearch.github.io/](https://gerdaresearch.github.io/) .
