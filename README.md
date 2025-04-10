# FlowerPlantshop_Customersurvey2025
# Correspondence Analysis of Flower Preferences (Color Tone + Style/Emotion/Texture)


This repository contains Python scripts and documentation for conducting a correspondence analysis (CA) on consumer preferences for flowers and plants.  
The analysis is based on a survey conducted in **January 2025** as part of a government-supported initiative in Japan:

> **Survey Title:**  
> *"2025 Consumer Survey of Florist and Garden Store Users"*  
> **Conducted under:**  
> The FY2024 MAFF (Ministry of Agriculture, Forestry and Fisheries of Japan) project for strengthening sustainable domestic flower production and distribution  
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

> 📊 **Full survey results and interpretation by the author (Aoki Kyoko) are available at:**  
> - Publications page: [https://gerdaresearch.github.io/publication](https://gerdaresearch.github.io/publication)  
> - Commentary and project archive: [https://gerdaresearch.github.io/year-archive/](https://gerdaresearch.github.io/year-archive/)



---

## 🔍 Objective

To visualize consumer preferences for flowers using correspondence analysis, and explore differences across demographics and stores based on color tone, style, emotion, and texture.

---

## 🔍 Objective

To visualize consumer preferences for flowers (style, texture, emotional impression, and color tone) using correspondence analysis, and to explore structural differences across gender-age segments, business types, and stores.

---

## 📂 Data Structure (Overview)

### Input Columns:
- **Q17_1–Q17_15**: Style/texture/emotion preferences (7-point bipolar SD scale)
- **Q15_1–Q15_5**: Color tone preferences (7-point scale)

### Row Groupings:
- **QSAGE**: Gender × Age (8 groups: e.g., 1 = Male 20s, ..., 8 = Female 50s)
- **GATE**: Business Type (4 types: Florist, Supermarket, Home Center, Garden Store)
- **QBD**: Store (9 stores, e.g., Hibiya-Kadan, Ao Flora, etc.)

---

## ⚙️ Preprocessing Steps

### (1) Bipolar Decomposition and Neutral Flagging

- All SD-scale items (Q17, Q15) are split into three components:
  - Left-side or Negative preference
  - Right-side or Positive preference
  - Neutral flag (`score = 4`)
- Neutral responses are excluded from CA and analyzed separately.

### (2) Scoring Rules

| Scale | Left/Negative | Neutral | Right/Positive |
|-------|---------------|---------|----------------|
| 1     | 3 pts         |         |                |
| 2     | 2 pts         |         |                |
| 3     | 1 pt          |         |                |
| 4     |     → flag 1  | 1 (flag)|     → flag 1   |
| 5     |               |         | 1 pt           |
| 6     |               |         | 2 pts          |
| 7     |               |         | 3 pts          |

- Each item is transformed into three columns, e.g.,  
  `Q17_1_left` (e.g., “Cute”), `Q17_1_center` (neutral flag), `Q17_1_right` (e.g., “Chic”)

### (3) Color Tone Items (Q15_x)
- Processed similarly, with `Q15_1_neg`, `Q15_1_center`, `Q15_1_pos`, etc.

---

## 📊 Correspondence Analysis Procedure

### (1) Create CA Input Matrix

- Combine all transformed columns for Q15 and Q17
- Label columns with dictionary (e.g., `"Q17_1_left"` → `"Cute"`)
- Stack row groups: Gender-Age, Business Type, Store

### (2) Run Correspondence Analysis (CA)

- Project row and column categories into CA space
- Extract axis coordinates, inertia (variance explained), and contributions

### (3) Plotting

- Map the result with labeled rows and columns (color-coded)
- Show multiple axis pairs (e.g., Axis 1 × 2, 1 × 3)

---

## 📈 Statistical Indices from CA

| Metric              | Description                                        |
|---------------------|----------------------------------------------------|
| Explained Inertia   | Variance explained by each axis                   |
| Cumulative Inertia  | Total variance explained across selected axes     |
| Total Inertia       | Overall structural strength of the CA table       |

---

## 🔍 Additional Analyses

- **Variable Contributions per Axis**  
  Identify which items contribute most to each axis (positive/negative).

- **Store–Item Distance Analysis**  
  Compute structural distances (5D) between each store and style/color items.

- **Near–Far Ranking Tables**  
  List closest and farthest items per store based on CA space position.

- **Commonly Distant / High-Variance Items**  
  Find items that are either:
  - Distant from all stores (shared uniqueness), or  
  - Vary significantly between stores (polarizing).

- **Neutrality Rate Analysis**  
  Show which items receive the most "neutral" responses (`score = 4`), by demographic.

---

## 🧪 Evaluation

- Axes are interpreted based on contribution values and content alignment.
- Visual and statistical inspection reveals:
  - Preference clusters by demographic
  - Structural uniqueness of stores
  - Impact of neutrality in preference patterns

---

## 🔧 Note on Usage

This repository includes:
- Full Python code for preprocessing, CA, plotting, and evaluation
- Commented, modular structure to allow customization

> Since the raw data is not provided, this project is shared for methodological reference only.

---

## 📎 Contact

For further information or collaboration on flower consumer research, feel free to reach out via GitHub or [your contact info].

---

<br>

#  花の好みに関するコレスポンデンス分析（色調＋スタイル・感情・質感）

このリポジトリは、花や植物に関する生活者の好みを、**コレスポンデンス分析（CA）**を用いて分析・可視化するためのPythonコードとドキュメントを収録したものです  

この分析は、以下の調査に基づいています：

> **調査名：**  
> 「2025年 花店・園芸店利用者調査」  
> **事業名：**  
> 令和6年度 農林水産省 持続的生産強化対策事業  
> （国産花き生産流通強化推進協議会）  
> **企画～分析～報告：** 青木 恭子  
> **調査時期：** 2025年1月  
> **回答者数：** n = 518（過去1年以内に9店舗を利用した生活者）  
> **対象店舗：** 花店、スーパー、ホームセンター、園芸店（4業態 9店舗）
**注記：** 園芸店カテゴリおよび各店舗単位での回答者数は比較的少数のため、  
> これらの結果は**参考値としてご利用ください**。
回答者には、**色調・スタイル・質感・感情的印象**に関する好みを尋ねました　
この回答結果をもとに、**性年齢別・業態別・店舗別**にコレスポンデンス分析を実施しました

### ✳️ 分析手法の特徴：

- SD法（形容詞対）の左右分解＋中立フラグ化（7点尺度）
- 中庸派（スコア4）が多いため、中立は除外し、別途分析
- 極性スコアのみでCAを実行し、項目別・カテゴリ別の構造を可視化

> 🔒 **ご注意：** 本リポジトリにはローデータは含まれていません  
>  
> 📊 **調査結果・解説は以下で公開しています：**  
> - 調査結果・図表：[https://gerdaresearch.github.io/publication](https://gerdaresearch.github.io/publication)  
> - 解説・解釈：[https://gerdaresearch.github.io/year-archive/](https://gerdaresearch.github.io/year-archive/)

---

## 🔍 目的

花に関する嗜好の構造を可視化し、性年齢や店舗別の違いを明らかにすること

---

## 📂 データ構成（概要）

- **Q17_1〜Q17_15**：スタイル・質感・感情（7点SD法）
- **Q15_1〜Q15_5**：色調（7点尺度）

**行カテゴリ：**
- QSAGE：性別 × 年代（8区分 = 男性20代、男性30代、男性40代、男性50代、女性20代、女性30代、女性40代、女性50代）
- GATE：業態（4区分＝花店、スーパー、ホームセンター、園芸店）
- QBD：店舗（9店舗＝日比谷花壇、青山フラワーマーケット、花専門店（日比谷花壇・青フラ以外）、イオン、スーパー（イオン以外）、カインズ、ホームセンター（カインズ以外）、オザキフラワーパーク、園芸店（オザキ以外）

---

## ⚙️ 前処理ステップ

### (1) SD法項目の左右分極化＋中立フラグ処理
- 左右の嗜好方向にスコア化し、中立（スコア4）はフラグ化

### (2) スコアルール

| スコア | 左／ネガ | 中立 | 右／ポジ |
|--------|----------|------|----------|
| 1      | 3点       |      |          |
| 2      | 2点       |      |          |
| 3      | 1点       |      |          |
| 4      | → フラグ1 | ✔️   | → フラグ1 |
| 5      |           |      | 1点       |
| 6      |           |      | 2点       |
| 7      |           |      | 3点       |

---

## 📊 CA実行の流れ

1. 新列（Q15, Q17）で入力マトリクスを作成  
2. `prince.CA()` で対応分析を実行  
3. 結果（座標・寄与率）を可視化

---

## 📈 主な統計指標

- **寄与率（explained inertia）**  
- **累積寄与率（cumulative inertia）**  
- **総慣性（total inertia）**

---

## 🔍 追加分析内容

- 軸ごとの寄与項目ランキング
- 店舗 × スコア項目間の構造的距離
- 距離の近い／遠い項目のランキング
- 共通に遠い項目／店舗間で差が出る項目の抽出
- 中立率の項目別・属性別ランキング

---

## 🧷 注意事項

- このコードは分析プロセスの共有用です
- ローデータは非公開
- 類似の構造を持つデータに応用可能

---




