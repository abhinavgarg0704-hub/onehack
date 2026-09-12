# FloodSense AI: Scientific & Mathematical Methodology

## 1. Problem Formulation
Flood inundation is an intrinsically spatial and temporal phenomenon governed by hydrological runoff, antecedent soil moisture, local topography, and river hydrodynamics. We formulate spatial flood risk as a calibrated conditional probability estimation problem:

$$P(\text{Flood}_{s, t} = 1 \mid \mathbf{X}_{s, \le t})$$

where:
- $s \in \mathcal{S}$ represents a spatial grid cell in the geographic bounding box of the study area (Kaziranga-Brahmaputra corridor, Assam: $26.45^{\circ}\text{N} - 26.85^{\circ}\text{N}, 93.05^{\circ}\text{E} - 93.65^{\circ}\text{E}$).
- $t$ represents the target evaluation time step.
- $\mathbf{X}_{s, \le t}$ denotes the feature vector available strictly at or prior to time $t$ (zero future information leakage).

---

## 2. Remote Sensing Indices & Mathematical Formulation

### 2.1 Normalized Difference Vegetation Index (NDVI)
$$NDVI = \frac{B8 - B4}{B8 + B4 + \epsilon}$$
Vegetation canopy saturation indicator; decreases as floodwaters submerge understory vegetation.

### 2.2 Normalized Difference Water Index (NDWI, McFeeters 1996)
$$NDWI = \frac{B3 - B8}{B3 + B8 + \epsilon}$$
Leverages high reflectance of water in Green ($B3, 560\text{ nm}$) and high absorption in NIR ($B8, 842\text{ nm}$).

### 2.3 Modified Normalized Difference Water Index (MNDWI, Xu 2006)
$$MNDWI = \frac{B3 - B11}{B3 + B11 + \epsilon}$$
Substitutes SWIR1 ($B11, 1610\text{ nm}$) for NIR, suppressing noise from wet soils, sediment-laden river water, and alluvial sandbars.

---

## 3. Antecedent Precipitation Aggregation
Antecedent rainfall over lag windows captures basin saturation:
$$R_{w}(t) = \sum_{k=0}^{w-1} P(t - k), \quad w \in \{1, 3, 7, 14\} \text{ days}$$
Computed from daily CHIRPS observations.

---

## 4. Topographic Derivations
Using the 30m SRTM DEM, slope $\theta$ is calculated from spatial elevation gradients:
$$\theta = \arctan\left(\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}\right) \times \frac{180}{\pi}$$

Distance to drainage $D(s)$ is computed using Euclidean distance transform:
$$D(s) = \min_{p \in \mathcal{W}} \|s - p\|_2$$
where $\mathcal{W}$ is the permanent water mask from JRC Global Surface Water.

---

## 5. Machine Learning Architectures & Calibrated Risk

### 5.1 Random Forest Classifier
Ensemble of 150 de-correlated decision trees with bootstrap aggregation:
$$P(\text{Flood} = 1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} f_b(\mathbf{x})$$
Weighted balanced loss is applied to account for regional land/water imbalance.

### 5.2 Comparative XGBoost
Gradient boosted decision tree optimizing regularized binary cross-entropy:
$$\mathcal{L}(\phi) = \sum_i l(y_i, \hat{y}_i) + \sum_k \Omega(f_k)$$

### 5.3 Risk Index Mapping
The continuous probability is scaled to a standardized hazard score:
$$\text{Risk Score}(s) = \text{round}(P(\text{Flood} = 1 \mid \mathbf{x}_s) \times 100, 1)$$

Categorized into:
- **LOW**: $0 - 25$
- **MODERATE**: $25 - 50$
- **HIGH**: $50 - 75$
- **VERY HIGH**: $75 - 100$

---

## 6. Historical Evaluation Metrics
Evaluated on holdout spatial peak flood mask:
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Intersection over Union (IoU / Jaccard)**: $\frac{TP}{TP + FP + FN}$
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve.
