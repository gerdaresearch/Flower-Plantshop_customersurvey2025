# Preferences for Flowers × Gender-Age, Business Type — Correspondence Analysis  
# Color Preferences + SD Adjective Pairs — Bipolar Scoring Method, Neutral Responses Flagged Separately  
# Python Code

# 1. Load CSV File

# Path to the CSV file
file_path = "YOUR_FILE.csv"

# Load the CSV file — specify encoding to properly read Japanese fonts
df = pd.read_csv(file_path, encoding="utf-8")


# 2. Preprocessing

# (1) Bipolar scoring for Q17_x and Q15_x (7-point SD scale items)

"""
# Scoring Strategy — Expanded SD Format
# Each 7-point SD item (e.g., “Cute – Chic”) is split into two preference axes:
# → Convert the two poles into separate preference indicators
"""

# ------------------------------------------
# Step 1: Create CA-ready DataFrame from raw data
# ------------------------------------------

# Required column names
q15_columns = [f"Q15_{i}" for i in range(1, 6)]      # Color preferences
q17_columns = [f"Q17_{i}" for i in range(1, 16)]     # Style preferences
id_and_keys = ["KEY", "QSAGE", "GATE", "QBD"]        # Respondent ID and keys

# Extract only the necessary columns
ca_raw_df = df[id_and_keys + q15_columns + q17_columns].copy()


# -----------------------------------------------------------------------------  
# Step 2: Decompose Q15 and Q17 items into left/right components and assign scores  
# -----------------------------------------------------------------------------  

# Decompose the 7-point scale and assign scores (max 3 points for both sides)

"""
# Response Value    Left adjective / Negative side     Right adjective / Positive side     Neutral
# 1                3                                   0                                    0
# 2                2                                   0                                    0
# 3                1                                   0                                    0
# 4                0                                   0                                    1
# 5                0                                   1                                    0
# 6                0                                   2                                    0
# 7                0                                   3                                    0

# A value of 1 in the "neutral" flag column simply indicates neutrality (not a score).
# It is excluded from CA, but used for "neutrality ranking" and identifying middle-ground respondents.
# Neutral clusters are also visualized to identify “safe” or “non-committal” response trends.
"""

# Scoring for Q15 (Color Preferences) using 3-2-1 method
for col in q15_columns:
    base = col + "_"

    # Negative side (Scores 1–3 → 3, 2, 1)
    ca_raw_df[base + "neg"] = df[col].map({1: 3, 2: 2, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0})

    # Positive side (Scores 5–7 → 1, 2, 3)
    ca_raw_df[base + "pos"] = df[col].map({1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3})

    # Neutral (Score = 4 → Flag = 1)
    ca_raw_df[base + "center"] = df[col].apply(lambda x: 1 if x == 4 else 0)


# Q17 (Style preferences using SD method) — No reversed items here, so decomposition is applied uniformly

# Scoring for Q17 (Style preferences) — max 3 points
for col in q17_columns:
    base = col + "_"

    # Left adjective: Scores 1–3 → 3, 2, 1 points
    ca_raw_df[base + "left"] = df[col].map({1: 3, 2: 2, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0})

    # Right adjective: Scores 5–7 → 1, 2, 3 points
    ca_raw_df[base + "right"] = df[col].map({1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 3})

    # Neutral flag: Score = 4 → 1
    ca_raw_df[base + "center"] = df[col].apply(lambda x: 1 if x == 4 else 0)


# -----------------------------------------------------------------------------  
# Step 3: Check the current CA DataFrame (ca_all_input)  
# -----------------------------------------------------------------------------  

# Copy the scored DataFrame for CA input
ca_all_input = ca_raw_df.copy()

# Ensure there are no duplicated column names
assert ca_all_input.columns.duplicated().sum() == 0, "There are duplicated column names"

# Display column count
print("Number of columns:", len(ca_all_input.columns))

# Display all column names
print("List of columns:")
print(ca_all_input.columns.tolist())

print("ca_all_input")  # Optional check



# (2) Define Label Dictionaries for Style (Q17) and Color (Q15)
# -----------------------------------------------------------------------------  
# Step 1: Define label dictionaries  
# -----------------------------------------------------------------------------  

# Labels for decomposed adjectives in Q17 — include polarity and neutral flag

adjective_labels_split = {
    "Q17_1_left": "Cute",             "Q17_1_right": "Chic",              "Q17_1_center": "Cute (Neutral)",
    "Q17_2_left": "Seasonal",         "Q17_2_right": "Standard",          "Q17_2_center": "Seasonal (Neutral)",
    "Q17_3_left": "Casual",           "Q17_3_right": "Formal",            "Q17_3_center": "Casual (Neutral)",
    "Q17_4_left": "Bold",             "Q17_4_right": "Delicate",          "Q17_4_center": "Bold (Neutral)",
    "Q17_5_left": "Gorgeous",         "Q17_5_right": "Chic (Muted)",      "Q17_5_center": "Gorgeous (Neutral)",
    "Q17_6_left": "Rustic",           "Q17_6_right": "Urban",             "Q17_6_center": "Rustic (Neutral)",
    "Q17_7_left": "Western",          "Q17_7_right": "Japanese",          "Q17_7_center": "Western (Neutral)",
    "Q17_8_left": "Gentle",           "Q17_8_right": "Cool",              "Q17_8_center": "Gentle (Neutral)",
    "Q17_9_left": "Silky",            "Q17_9_right": "Textured",          "Q17_9_center": "Silky (Neutral)",
    "Q17_10_left": "Solid",           "Q17_10_right": "Fluffy",           "Q17_10_center": "Solid (Neutral)",
    "Q17_11_left": "Flowy",           "Q17_11_right": "Rounded",          "Q17_11_center": "Flowy (Neutral)",
    "Q17_12_left": "Shadowed",        "Q17_12_right": "Translucent",      "Q17_12_center": "Shadowed (Neutral)",
    "Q17_13_left": "Healing",         "Q17_13_right": "Uplifting",        "Q17_13_center": "Healing (Neutral)",
    "Q17_14_left": "Novel",           "Q17_14_right": "Nostalgic",        "Q17_14_center": "Novel (Neutral)",
    "Q17_15_left": "Lustrous",        "Q17_15_right": "Pure",             "Q17_15_center": "Lustrous (Neutral)"
}


# Labels for Q15 color items
color_labels = {
    "Q15_1": "Vivid Color",
    "Q15_2": "Pastel Color",
    "Q15_3": "Nuanced Color (Muted)",
    "Q15_4": "Dark Color",
    "Q15_5": "Monotone"
}

# Decomposed labels for Q15 (with polarity)
color_labels_split = {}
for q15_col, base_label in color_labels.items():
    color_labels_split[q15_col + "_neg"] = f"{base_label} (Negative)"
    color_labels_split[q15_col + "_pos"] = f"{base_label} (Positive)"
    color_labels_split[q15_col + "_center"] = f"{base_label} (Neutral)"


# Gender-Age Labels (QSAGE column)
qsage_labels = {
    1: "Male, 20s",
    2: "Male, 30s",
    3: "Male, 40s",
    4: "Male, 50s",
    5: "Female, 20s",
    6: "Female, 30s",
    7: "Female, 40s",
    8: "Female, 50s"
}

# Business Type Labels (GATE column)
gate_to_business = {
    1: "Florist Shop",
    2: "Supermarket",
    3: "Home Center",
    4: "Garden Center"
}

# Store Labels (QBD column)
qbd_to_store = {
    1: "Hibiya-Kadan",
    2: "Ao Flora",
    3: "Specialty Flower Shop",
    4: "AEON",
    5: "Other Supermarkets",
    6: "CAINZ",
    7: "Other Home Centers",
    8: "Ozaki",
    9: "Other Garden Stores"
}


# -----------------------------------------------------------------------------
# 【Step 2】DataFrame Cleanup
# -----------------------------------------------------------------------------

# After scoring and decomposition, drop the original Q15 and Q17 columns
ca_all_input.drop(columns=q15_columns + q17_columns, inplace=True)


# -----------------------------------------------------------------------------
# 【Step 3】Merge All Label Dictionaries and Check Definitions
# -----------------------------------------------------------------------------

# Check label correspondence — look for any columns without a defined label

# Merge all label dictionaries
all_label_dict = {}
all_label_dict.update(adjective_labels_split)   # Q17: Style adjectives (left/right/center)
all_label_dict.update(color_labels_split)       # Q15: Color preferences (neg/pos/center)

# Select only score columns from ca_all_input (exclude identifiers and grouping keys)
score_columns = ca_all_input.columns.difference(["QSAGE", "GATE", "QBD", "KEY"], sort=False)

# Identify columns that are not defined in the label dictionary
unlabeled_columns = [col for col in score_columns if col not in all_label_dict]

print("⚠️ Columns without defined labels:")
for col in unlabeled_columns:
    print(f"  - {col}")

print(f"\n✅ Number of defined labels: {len(all_label_dict)} / Score columns: {len(score_columns)}")

# Display mapping list of defined labels
print("🔍 Defined label mappings (only those that are defined):")
for col in score_columns:
    label = all_label_dict.get(col, "(undefined)")
    print(f"{col:20s} → {label}")




# 3. Creation of Average Score Table (CA Input): Cross Tabulation

# (1) Creation of CA Input Matrix (Generalized)

"""
# This function: expanded_ca()

Purpose:
- Converts Q15 and Q17 SD-scale data into polarized scores (negative-positive / left-right)
- Excludes neutral flags (_center columns)
- Aggregates average scores by specified keys (e.g., QSAGE, QBD, GATE)
- Outputs a CA input matrix and performs Correspondence Analysis (CA)

Returns 6 outputs:
  Variable   | Description
  -----------|--------------------------------------------------------------
  mat        | CA input matrix (rows = categories, columns = polarized scores)
  model      | Fitted CA model using prince.CA
  row        | Row category coordinates (e.g., gender-age, store, etc.)
  col        | Column coordinates (style and color items)
  inertia    | Explained variance ratio (inertia) for each axis
  var        | List of columns (polarized score variables) used in the matrix
"""

# Function Definition: expanded_ca
import prince

def expanded_ca(df, q15, q17, keys, labels, label_dict, n=5):
    """
    Extended CA processing (excluding neutral responses) for gender-age + store + business type
    - df: Original DataFrame
    - q15, q17: Lists of Q15 and Q17 column names
    - keys: List of group-by keys (e.g., ['QSAGE', 'QBD', 'GATE'])
    - labels: Dictionary for mapping group labels (e.g., {'QSAGE': ..., 'QBD': ..., 'GATE': ...})
    - label_dict: Label dictionary for score columns
    - n: Number of dimensions (default = 5)
    """

    # Copy only necessary columns
    raw = df[["KEY"] + keys + q15 + q17].copy()

    # Convert Q15 columns to negative/positive/neutral scores
    for col in q15:
        base = col + "_"
        raw[base + "neg"] = df[col].map({1:3, 2:2, 3:1, 4:0, 5:0, 6:0, 7:0})
        raw[base + "pos"] = df[col].map({1:0, 2:0, 3:0, 4:0, 5:1, 6:2, 7:3})
        raw[base + "center"] = df[col].apply(lambda x: 1 if x==4 else 0)

    # Convert Q17 columns similarly
    for col in q17:
        base = col + "_"
        raw[base + "left"] = df[col].map({1:3, 2:2, 3:1, 4:0, 5:0, 6:0, 7:0})
        raw[base + "right"] = df[col].map({1:0, 2:0, 3:0, 4:0, 5:1, 6:2, 7:3})
        raw[base + "center"] = df[col].apply(lambda x: 1 if x==4 else 0)

    # Drop original Q15 and Q17 columns
    raw.drop(columns=q15 + q17, inplace=True)

    # Extract only polarized score columns (exclude _center)
    var = [c for c in raw.columns
           if any(k in c for k in ["_neg", "_pos", "_left", "_right"])
           and not c.endswith("_center")]

    # Create average score matrices for each key
    mats = []
    for k in keys:
        m = raw.groupby(k)[var].mean().round(2)
        m.index = m.index.map(labels[k])
        m.columns = [label_dict.get(c, c) for c in m.columns]
        mats.append(m)

    # Concatenate all matrices by row
    mat = pd.concat(mats, axis=0)

    # Fit CA model using prince.CA
    model = prince.CA(n_components=n, random_state=42).fit(mat)
    row = model.row_coordinates(mat)
    col = model.column_coordinates(mat)
    inertia = np.array(model.explained_inertia_)

    return mat, model, row, col, inertia, var



# 4. Execution of Correspondence Analysis (CA)

# (1) Run Cross Tabulation and CA Function (Neutral Responses Excluded, General Use)

mat, model, row, col, inertia, var = expanded_ca(
    df=df,
    q15=[f"Q15_{i}" for i in range(1, 6)],
    q17=[f"Q17_{i}" for i in range(1, 16)],
    keys=["QSAGE", "QBD", "GATE"],
    labels={"QSAGE": qsage_labels, "QBD": qbd_to_store, "GATE": gate_to_business},
    label_dict=all_label_dict
)


# (2) Table of Explained Variance and Cumulative Variance per Axis

# -----------------------------------------------------------------------------
# 【Step 1】Explained Variance and Cumulative Variance per Axis
# -----------------------------------------------------------------------------

def summarize_axis_inertia(inertia, n_axis=5):
    """
    Returns a table of explained variance and cumulative variance per axis.
    - inertia: A list or array from explained_inertia_
    - n_axis: Number of axes to display (default = 5)
    """
    inertia = np.array(inertia)
    n = min(n_axis, len(inertia))
    df = pd.DataFrame({
        "Axis": [f"Axis {i+1}" for i in range(n)],
        "Explained Variance (%)": (inertia[:n] * 100).round(2),
        "Cumulative Variance (%)": (np.cumsum(inertia[:n]) * 100).round(2)
    })
    return df


# -----------------------------------------------------------------------------
# 【Step 2】Calculate Total Inertia
# -----------------------------------------------------------------------------

def summarize_axis_inertia_full(model):
    inertia = np.array(model.explained_inertia_)
    df = pd.DataFrame({
        "Axis": [f"Axis {i+1}" for i in range(len(inertia))],
        "Explained Variance (%)": (inertia * 100).round(2),
        "Cumulative Variance (%)": (np.cumsum(inertia) * 100).round(2)
    })
    total = model.total_inertia_
    return df, total


# -----------------------------------------------------------------------------
# 【Step 3】Display Variance Summary Table
# -----------------------------------------------------------------------------

df_inertia = summarize_axis_inertia(inertia, n_axis=5)
df_inertia, total_inertia = summarize_axis_inertia_full(model)

print('Explained and Cumulative Variance by Axis — Grouped by Gender-Age, Store, and Business Type')
display(df_inertia)
print(f"Total Inertia: {total_inertia:.6f}")


# (3) Revised Analysis: Rerun After Removing "Business Type" Based on Variance Result

# -----------------------------------------------------------------------------
# 【Step 1】Modify Function Call for Revised CA
# -----------------------------------------------------------------------------

mat, model, row, col, inertia, var = expanded_ca(
    df=df,
    q15=[f"Q15_{i}" for i in range(1, 6)],
    q17=[f"Q17_{i}" for i in range(1, 16)],
    keys=["QSAGE", "QBD"],  # 👈 Removed "GATE"
    labels={
        "QSAGE": qsage_labels,
        "QBD": qbd_to_store
    },
    label_dict=all_label_dict,
    n=5
)


# -----------------------------------------------------------------------------
# 【Step 2】Display Revised Variance Table
# -----------------------------------------------------------------------------

df_inertia = summarize_axis_inertia(inertia, n_axis=5)
df_inertia, total_inertia = summarize_axis_inertia_full(model)

print('Explained and Cumulative Variance by Axis — Grouped by Gender-Age and Store')
display(df_inertia)
print(f"Total Inertia: {total_inertia:.4f}")


# -----------------------------------------------------------------------------
# 【Supplement】Manual Calculation of Explained Variance (Gender-Age Only)
# -----------------------------------------------------------------------------

from sklearn.decomposition import TruncatedSVD
import numpy as np

# Convert CA input matrix to values
X = ca_matrix.values

# Convert to probability matrix
grand_total = X.sum()
P = X / grand_total

# Marginal totals
row_sums = P.sum(axis=1).reshape(-1, 1)
col_sums = P.sum(axis=0).reshape(1, -1)

# Expected matrix and standardized residuals
expected = row_sums @ col_sums
S = (P - expected) / np.sqrt(expected)

# Truncated SVD
svd = TruncatedSVD(n_components=5, random_state=42)
svd.fit(S)

# Explained variance and cumulative contribution
explained = svd.explained_variance_ratio_
cumulative = np.cumsum(explained)

# Summary table
explained_df = pd.DataFrame({
    "Axis": [f"Axis {i+1}" for i in range(len(explained))],
    "Explained Variance (%)": (explained * 100).round(2),
    "Cumulative Variance (%)": (cumulative * 100).round(2)
})

print("▶ Manually Calculated Explained Variance (Axes CA1–): Gender-Age Only")
display(explained_df)


# -----------------------------------------------------------------------------
# 【Step 3】Manual Calculation of Total Inertia
# -----------------------------------------------------------------------------

total_inertia = ca.total_inertia_
print(f"▶ CA Total Inertia (Gender-Age Only): {total_inertia:.6f}")

# Note:
# Total inertia represents the overall dispersion in the cross-tabulation matrix.



# 5. CA Plotting and Mapping

# (1) Preparation for Mapping: General Plotting Function for Axis Pairs (1×2, 1×3, 2×3)

# Define plotting function
import matplotlib.pyplot as plt
from adjustText import adjust_text

def plot_ca_axes_styled(row_coords, col_coords, original_columns, explained, axis_pairs):
    for x_axis, y_axis in axis_pairs:
        fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
        texts = []

        # Set background color
        ax.set_facecolor((0.988, 0.984, 0.965, 0.6))

        # Row categories (Gender-Age + Store)
        for i, label in enumerate(row_coords.index):
            x = row_coords.iloc[i, x_axis]
            y = row_coords.iloc[i, y_axis]
            if "男性" in label or "女性" in label:
                plt.scatter(x, y, color='#405584', label="Gender-Age" if i == 0 else "", zorder=3)
                texts.append(plt.text(x, y, label, color='#405584', fontsize=11, weight='bold'))
            else:
                plt.scatter(x, y, color='#007E7A', marker='o', s=80,
                            edgecolor='white', linewidth=1.5,
                            label="Store" if i == 0 else "", zorder=4)
                texts.append(plt.text(x, y, label, color='#007E7A', fontsize=11, weight='bold'))

        # Column categories (Style & Color Items)
        label_q15_drawn = False
        label_q17_drawn = False
        for i in range(len(col_coords)):
            colname = original_columns[i]
            display_label = col_coords.index[i]
            if colname.startswith("Q15_"):
                c = '#4B5931'; m = '^'; label_color = c
                label_name = "Color" if not label_q15_drawn else None
                label_q15_drawn = True
            elif colname.startswith("Q17_"):
                c = '#990000'; m = 'D'; label_color = c
                label_name = "Style" if not label_q17_drawn else None
                label_q17_drawn = True
            else:
                c = 'gray'; m = 'x'; label_color = c; label_name = None

            ax.scatter(col_coords.iloc[i, x_axis], col_coords.iloc[i, y_axis],
                       color=c, marker=m, edgecolors='white', linewidths=0.5,
                       alpha=0.9, label=label_name, zorder=2)
            texts.append(
                ax.text(col_coords.iloc[i, x_axis], col_coords.iloc[i, y_axis],
                        display_label, color=label_color, fontsize=12)
            )

        # Adjust overlapping text labels
        adjust_text(texts, only_move={'points': 'y', 'text': 'xy'},
                    arrowprops=dict(arrowstyle="-", color='#322C10', lw=0.5))

        # Axis contribution rates
        x_pct = explained[x_axis] * 100
        y_pct = explained[y_axis] * 100
        total_pct = x_pct + y_pct

        plt.axhline(0, color='gray', linestyle='dotted')
        plt.axvline(0, color='gray', linestyle='dotted')
        plt.xlabel(f"Axis {x_axis + 1} ({x_pct:.1f}%)", fontsize=12)
        plt.ylabel(f"Axis {y_axis + 1} ({y_pct:.1f}%)", fontsize=12)

        plt.text(0.98, 0.02,
                 f"Total Contribution: Axis {x_axis + 1} + Axis {y_axis + 1} = {total_pct:.1f}%",
                 transform=plt.gca().transAxes,
                 fontsize=12, color='#322C10', ha='right', va='bottom')

        plt.title(f"Correspondence Analysis Map: Style & Color × Gender-Age & Store — Axis {x_axis+1} × Axis {y_axis+1}", pad=20)
        plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.1))
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()


(2) Run Plotting Function

plot_ca_axes_styled(
    row_coords=row,
    col_coords=col,
    original_columns=var,
    explained=inertia,
    axis_pairs=[(0, 1), (0, 2), (2, 1)]
)


# 6. Interpretation of Axes and Map

# (1) Contribution of Column Categories to Each Axis (A × B)
# Display top contributing variables per axis
# For each axis, identify variables contributing strongly in both positive and negative directions

# -----------------------------------------------------------------------------
# 【Step 1】SVD Decomposition (Extract Structural Components from mat)
# -----------------------------------------------------------------------------

from sklearn.decomposition import TruncatedSVD
import numpy as np

X = mat.T.values  # Transpose: rows = score items, columns = row categories
P = X / X.sum()   # Convert to relative frequencies (probability matrix)
row_sums = P.sum(axis=1).reshape(-1, 1)
col_sums = P.sum(axis=0).reshape(1, -1)
expected = row_sums @ col_sums
S = (P - expected) / np.sqrt(expected)

svd = TruncatedSVD(n_components=5, random_state=42)
svd.fit(S)

"""
# Procedure:
# - Prepare matrix S for SVD
#   (Rows = score items, Columns = categories)
# - Convert to probability matrix
# - Compute expected matrix
# - Compute standardized residuals: S = (P - expected) / sqrt(expected)
# - Apply Truncated SVD to decompose S

# From this, we obtain:
# - components_: structural loadings for each axis
# - explained_variance_: eigenvalue (variance) per axis

# Contribution per variable is calculated as:
#   contribution = (loading²) / variance of that axis
"""

# -----------------------------------------------------------------------------
# 【Step 2】Define Function to Output Axis Contributions and Loadings
# -----------------------------------------------------------------------------

def summarize_axis_contributions(ca_matrix, svd, axis_idx, top_n=10):
    axis_name = f"Axis {axis_idx+1}"
    var_names = ca_matrix.columns
    axis_values = svd.components_[axis_idx][:len(var_names)]  # Ensure length match
    variance = svd.explained_variance_[axis_idx]

    contrib = (axis_values ** 2) / variance
    df = pd.DataFrame({
        f"{axis_name} Loading": axis_values,
        f"{axis_name} Contribution": contrib
    }, index=var_names[:len(axis_values)]).round(3)

    # Top contributing variables in positive and negative directions
    top_pos = df.sort_values(by=f"{axis_name} Loading", ascending=False).head(top_n)
    top_neg = df.sort_values(by=f"{axis_name} Loading").head(top_n)

    summary = pd.concat([
        top_pos[[f"{axis_name} Loading"]],
        top_neg[[f"{axis_name} Loading"]].rename(columns={f"{axis_name} Loading": f"{axis_name} Loading (Negative)"})
    ], axis=1)

    return df.sort_values(by=f"{axis_name} Contribution", ascending=False), summary

# -----------------------------------------------------------------------------
# 【Step 3】Display Contribution and Polarity Summary for Axes 1 to 3
# -----------------------------------------------------------------------------

for i in range(3):  # Axes 1–3
    contrib_df, summary_df = summarize_axis_contributions(mat, svd, axis_idx=i, top_n=10)

    print(f"\n▶ Top Contributions to Axis {i+1} (Gender-Age + Store):")
    display(contrib_df.head(10))

    print(f"\n▶ Strong Positive and Negative Contributors to Axis {i+1}:")
    display(summary_df)


# (2) Distance from Origin in Multidimensional Space
# Determine how far each row category (e.g., gender-age, store) is from the center of the CA space
# Helps identify structurally distinctive attributes

# -----------------------------------------------------------------------------
# 【Step 1】Define Function to Calculate Distance from Origin
# -----------------------------------------------------------------------------

def calculate_distances_from_origin(row_coords, n_axes=5):
    """
    Calculate Euclidean distance from the origin (0,0,…,0) for each row category
    - row_coords: Coordinates of row categories from model.row_coordinates()
    - n_axes: Number of axes to use in distance calculation (default = 5)

    Returns:
    - DataFrame with each axis coordinate and the resulting distance
    """
    coords = row_coords.iloc[:, :n_axes]             # Use only first n axes
    squared = coords ** 2                            # Square each axis value
    distances = np.sqrt(squared.sum(axis=1))         # Euclidean distance
    df = coords.copy()
    df.columns = [f"Axis {i+1}" for i in range(n_axes)]
    df["Distance"] = distances.round(3)
    return df.round(3)


# -----------------------------------------------------------------------------
# 【Step 2】Output Distances by Gender-Age and Store
# -----------------------------------------------------------------------------

print('Distance from Origin in 5D CA Space (Gender-Age and Store)')
df_distance = calculate_distances_from_origin(row, n_axes=5)
display(df_distance)


# -----------------------------------------------------------------------------
#【Step 3】 Visualization of Structural Distances Between Stores and Style/Color Items
# -----------------------------------------------------------------------------

# Define Function to Rank Item Distances per Store

def rank_item_distances_by_store(row_coords, col_coords, top_n=10, axes=5):
    """
    For each store, rank the most distant score items (style/color) in the CA space

    Parameters:
    - row_coords: Row coordinates from CA model (includes gender-age and stores)
    - col_coords: Column coordinates from CA model (score items)
    - top_n: Number of top distant items to return (default = 10)
    - axes: Number of axes to use in distance calculation

    Returns:
    - Dictionary: {store name: DataFrame of top distant items}
    """
    
　stores = [label for label in row_coords.index
              if label not in ["Male, 20s", "Male, 30s", "Male, 40s", "Male, 50s",
                              "Female, 20s", "Female, 30s", "Female, 40s", "Female, 50s"]]
    results = {}

    for store in stores:
        store_vec = row_coords.loc[store].iloc[:axes].values
        distances = col_coords.iloc[:, :axes].apply(
            lambda x: np.sqrt(np.sum((x.values - store_vec) ** 2)), axis=1)
        
        ranked = distances.sort_values(ascending=False).head(top_n)
        results[store] = ranked.to_frame(name="Distance")

    return results

# Execute for each store
store_distance_ranks = rank_item_distances_by_store(row, col, top_n=10, axes=5)

# View ranking for specific stores
print("Top Distant Items for Hibiya-Kadan")
display(store_distance_ranks["日比谷花壇"])

# (Other stores like AoFlora, AEON, Ozaki can be checked similarly)



# (3) Relationship Between Stores and Score Items: Automated Analysis

# -----------------------------------------------------------------------------
# 【Step 1】Generate Full Distance Matrix Between Stores and Score Items
# -----------------------------------------------------------------------------

# This function create_store_item_distance_matrix() generates a distance matrix:
# - Rows = stores
# - Columns = style and color items
# It can be used to compare which items are closest or farthest from each store.
# Useful as a quantitative alternative when visual inspection is difficult.

def create_store_item_distance_matrix(row_coords, col_coords, axes=5):
    """
    Create a structural distance matrix: rows = stores, columns = style/color items
    Can be used to find:
      - The most distant or closest items for each store
      - Items with high variation in distance across stores

    Parameters:
    - row_coords: model.row_coordinates() (includes gender-age and stores)
    - col_coords: model.column_coordinates() (score items)
    - axes: Number of axes used (default = 5)

    Returns:
    - DataFrame: index = store names, columns = score item names, values = distances in 5D space
    """

    # Extract store labels (exclude gender-age)
    stores = [label for label in row_coords.index
              if label not in ["Male, 20s", "Male, 30s", "Male, 40s", "Male, 50s",
                               "Female, 20s", "Female, 30s", "Female, 40s", "Female, 50s"]]

    matrix = {}

    for store in stores:
        store_vec = row_coords.loc[store].iloc[:axes].values
        distances = col_coords.iloc[:, :axes].apply(
            lambda x: np.sqrt(np.sum((x.values - store_vec) ** 2)), axis=1)
        matrix[store] = distances

    return pd.DataFrame(matrix).T.round(4)

# Apply the function to create the matrix
df_store_item_distance = create_store_item_distance_matrix(row, col, axes=5)

# -----------------------------------------------------------------------------
# 【Step 2】Extract Top Near and Far Items for Each Store
# -----------------------------------------------------------------------------

# Define a function to extract closest and farthest items for each store (top 7)
def get_top_near_far_items_by_store(distance_df, top_n=7):
    """
    From the distance matrix, extract:
      - The closest items (high affinity)
      - The farthest items (structural distinction) for each store

    Parameters:
    - distance_df: Output from create_store_item_distance_matrix() (store × item matrix)
    - top_n: Number of items to extract for each direction (default = 7)

    Returns:
    - Dictionary: {store name: DataFrame with near/far item comparison}
    """
    results = {}

    for store in distance_df.index:
        row = distance_df.loc[store]
        closest = row.sort_values().head(top_n).to_frame(name="Distance (Nearest)")
        farthest = row.sort_values(ascending=False).head(top_n).to_frame(name="Distance (Farthest)")
        combined = pd.concat([closest, farthest], axis=1)
        results[store] = combined

    return results

# Run the function to extract near/far items
store_near_far = get_top_near_far_items_by_store(df_store_item_distance, top_n=7)

# View specific results
display(store_near_far["Hibiya-Kadan"])
display(store_near_far["AEON"])
display(store_near_far["Ozaki"])



# -----------------------------------------------------------------------------
# 【Step 3】Compile Near/Far Items for Each Store into One Table
# -----------------------------------------------------------------------------

# Combine the closest and farthest items for each store into a single large DataFrame

# Function Definition: compile_near_far_items_table()

def compile_near_far_items_table(near_far_dict):
    """
    Compile closest/farthest items for each store into one combined DataFrame

    Parameters:
    - near_far_dict: Output from get_top_near_far_items_by_store() (dictionary)

    Returns:
    - DataFrame: Comparison table of near/far items for all stores (rounded to 3 decimals)
    """
    rows = []

    for store, df in near_far_dict.items():
        for i in range(len(df)):
            near_item = df.index[i] if pd.notna(df.index[i]) else ""
            near_val = df.iloc[i, 0]
            far_item = df.index[i] if pd.notna(df.index[i]) else ""
            far_val = df.iloc[i, 1]

            rows.append({
                "Store": store,
                "Rank": i + 1,
                "Closest Item": near_item,
                "Distance (Near)": round(near_val, 3),
                "Farthest Item": far_item,
                "Distance (Far)": round(far_val, 3)
            })

    return pd.DataFrame(rows)

df_store_item_near_far = compile_near_far_items_table(store_near_far)
display(df_store_item_near_far)


# -----------------------------------------------------------------------------
# 【Step 4】Full Data Display (Run Once, Then Reset Settings)
# -----------------------------------------------------------------------------

# Enable full row/column display for manual checking
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

display(df_store_item_near_far)

# IMPORTANT: Reset display options afterward to avoid large and slow outputs
# You may comment out the next line to manually control resetting
# pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')



# (4) Visualization of Store × Score Item Distances: Bubble Chart

# -----------------------------------------------------------------------------
# 【Step 1】Plot with seaborn: Store × Score Item Distances (Bubble Chart)
# -----------------------------------------------------------------------------

# Bubble chart: visualize distances between stores and score items
# Uses seaborn for plotting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data: df_store_item_distance

def plot_bubble_distance_chart(distance_df, top_n=7):
    """
    Visualize store × score item distances using bubble sizes.

    Parameters:
    - distance_df: Distance matrix (store × item) as a DataFrame
    - top_n: Number of closest items to show per store (default = 7)

    Plot specs:
    - X-axis: Score items (e.g., style, color)
    - Y-axis: Stores
    - Bubble size: Distance (smaller = closer)
    """
    records = []

    for store in distance_df.index:
        sorted_items = distance_df.loc[store].sort_values().head(top_n)
        for item, dist in sorted_items.items():
            records.append({
                "Store": store,
                "Item": item,
                "Distance": dist
            })

    df_plot = pd.DataFrame(records)

    plt.figure(figsize=(13, 10), dpi=150)

    ax = sns.scatterplot(
        data=df_plot, x="Item", y="Store", size="Distance",
        sizes=(30, 600), legend="brief", alpha=0.6,
        color="#5B9BD5"  # Blue color
    )

    # Customize legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles[1:], labels=labels[1:], title="Distance",
        loc='upper right', labelspacing=1.5, frameon=False
    )

    plt.title("Closest Style/Color Items by Store (Structural Proximity)", fontsize=14, pad=20)
    plt.xticks(rotation=90, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.show()

# Run seaborn version
plot_bubble_distance_chart(df_store_item_distance, top_n=7)


# -----------------------------------------------------------------------------
# 【Step 2】Plot with matplotlib: Store × Score Item Distances (Bubble Chart)
# -----------------------------------------------------------------------------

# When seaborn legend customization is difficult, use matplotlib instead

def plot_bubble_distance_chart_matplotlib(distance_df, top_n=7):
    import matplotlib.pyplot as plt

    # Prepare data
    records = []
    for store in distance_df.index:
        sorted_items = distance_df.loc[store].sort_values().head(top_n)
        for item, dist in sorted_items.items():
            records.append({
                "Store": store,
                "Item": item,
                "Distance": dist
            })

    df_plot = pd.DataFrame(records)

    # Map for axes
    stores = df_plot["Store"].unique()
    items = df_plot["Item"].unique()
    store_map = {s: i for i, s in enumerate(reversed(stores))}
    item_map = {i: j for j, i in enumerate(items)}

    x = [item_map[i] for i in df_plot["Item"]]
    y = [store_map[s] for s in df_plot["Store"]]
    sizes = [d * 1200 + 50 for d in df_plot["Distance"]]  # Larger bubble for greater distance

    fig, ax = plt.subplots(figsize=(13, 10), dpi=150)
    sc = ax.scatter(x, y, s=sizes, c="#5B9BD5", alpha=0.6)

    # Axis labels
    ax.set_xticks(list(item_map.values()))
    ax.set_xticklabels(list(item_map.keys()), rotation=90, fontsize=16)
    ax.set_yticks(list(store_map.values()))
    ax.set_yticklabels(list(store_map.keys()), fontsize=16)

    # Title and grid
    plt.title("Closest Style/Color Items by Store (Structural Proximity)", fontsize=16, pad=20)
    plt.grid(True, linestyle=':')

    # Legend bubbles
    for size in [0.1, 0.3, 0.5]:
        plt.scatter([], [], s=size * 1200 + 50, c="#5B9BD5", alpha=0.6, label=f"{size:.1f}")

    plt.legend(title="Distance", loc="upper right", frameon=False, fontsize=12, title_fontsize=13)

    plt.tight_layout()
    plt.show()

# Run matplotlib version
plot_bubble_distance_chart_matplotlib(df_store_item_distance, top_n=7)



# (5) Visualization of Common and Variable Distance Items
# Identify items with consistently large distances and those with high variability across stores

# -----------------------------------------------------------------------------
# 【Step 1】Extract Items with Large Average Distances and High Variability
# -----------------------------------------------------------------------------

# Function Definition: analyze_common_and_variable_items()

import pandas as pd

def analyze_common_and_variable_items(distance_df, top_n=10):
    """
    Identify:
    - Items with consistently large distances (i.e., far from all stores = unique or avoided)
    - Items with large differences between stores (i.e., polarizing or distinctive)

    Parameters:
    - distance_df: Output from create_store_item_distance_matrix() (store × score item)
    - top_n: Number of items to return (default = 10)

    Returns:
    - df_mean_top: Items with the highest average distance (commonly distant)
    - df_std_top: Items with the highest standard deviation (variable across stores)
    """

    # Items with the highest average distance (distant for all stores)
    means = distance_df.mean(axis=0).sort_values(ascending=False).head(top_n).to_frame(name="Average Distance")

    # Items with the highest standard deviation (differences between stores)
    stds = distance_df.std(axis=0).sort_values(ascending=False).head(top_n).to_frame(name="Distance Variability")

    return means, stds

# -----------------------------------------------------------------------------
# 【Step 2】Output Most Distant and Most Variable Items
# -----------------------------------------------------------------------------

# Call the function to get top 10
df_common_far, df_variable_items = analyze_common_and_variable_items(df_store_item_distance, top_n=10)

# df_common_far: Items that are distant for all stores — "commonly distinctive items"
print('Items with Large Average Distance Across All Stores')
display(df_common_far)

# df_variable_items: Items with high variability between stores — "discriminative or polarizing items"
print('Items with High Distance Variability Across Stores')
display(df_variable_items)





# 7. Supplementary Analysis: Neutral Responses in SD Items

"""
Neutral responses (e.g., score = 4 in 7-point SD scale) are excluded from CA,
but stored as separate flag variables (columns ending in _center).
These neutral rates are useful for:
  - Identifying "middle-ground" respondents
  - Understanding ambiguity or non-committal preferences
Cluster analysis results indicate that approx. 30–40% of respondents fall into this "neutral" segment,
though it varies by store and gender-age.
Note: Neutral responses do not always equal neutral clusters.
"""

# -----------------------------------------------------------------------------
# (1) Neutral Response Flags by Gender-Age Group
# -----------------------------------------------------------------------------

# Extract columns that flag neutral responses (_center)
center_columns = [col for col in ca_all_input.columns if col.endswith("_center")]

# Group by QSAGE and calculate mean (neutral rate) for each item
neutral_rate_by_age = ca_all_input.groupby("QSAGE")[center_columns].mean().round(3)
neutral_rate_by_age.index = neutral_rate_by_age.index.map(qsage_labels)

# Replace column names with human-readable labels
neutral_rate_by_age.columns = [all_label_dict.get(col, col) for col in neutral_rate_by_age.columns]

# Display neutral rate table by gender-age
print("▶ Neutral Response Rate by Gender-Age Group:")
display(neutral_rate_by_age)

# -----------------------------------------------------------------------------
# (2) Overall Neutrality Ranking by Item
# -----------------------------------------------------------------------------

# Calculate overall average neutral response rate per item
neutral_rate_total = ca_all_input[center_columns].mean().sort_values(ascending=False).round(2)
neutral_rate_total.index = [all_label_dict.get(col, col) for col in neutral_rate_total.index]

print("▶ Neutrality Ranking by Item (Overall Average):")
display(neutral_rate_total)




