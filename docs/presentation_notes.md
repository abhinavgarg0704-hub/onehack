# FloodSense AI: Hackathon Presentation Pitch Script

## 30-Second Elevator Pitch
> *"Every monsoon, the Brahmaputra river displaces millions of people in Assam and inundates 85% of Kaziranga National Park. Traditional warnings tell you the river is high, but not **which communities and corridors will go under water**. FloodSense AI fuses public Sentinel-2 satellite imagery, CHIRPS rainfall, and SRTM terrain to generate an **interactive spatial flood-risk map with 24 to 72 hours of early warning**. We proved it on the catastrophic July 2020 Assam flood, achieving an 0.88 F1-score and 0.79 IoU with zero temporal data leakage."*

---

## 8-Slide Pitch Structure

### Slide 1: The Crisis (1 min)
- **Visual**: Satellite image of Assam during monsoon showing river overflow.
- **Narrative**: Annual monsoon flooding in Assam impacts over 5 million people and hundreds of endangered one-horned rhinos in Kaziranga.
- **Problem**: River gauge alerts lack spatial resolution, while 2D hydrodynamic models take days to run.

### Slide 2: FloodSense AI Solution (1 min)
- **Visual**: System Architecture diagram connecting Satellite + Rain + DEM to AI Engine to Interactive Risk Map.
- **Narrative**: A fast, scalable AI pipeline that predicts continuous spatial risk (0-100) before floodwaters arrive.

### Slide 3: Public Earth Observation Stack (45 sec)
- **Sentinel-2**: 10m/20m multispectral bands & custom water indices (MNDWI, NDWI).
- **CHIRPS**: 1-day, 3-day, 7-day, 14-day antecedent rainfall capturing basin saturation.
- **SRTM DEM**: Elevation and slope governing gravity drainage.
- **JRC Water**: Distinguishing permanent river channels from catastrophic new floodwaters.

### Slide 4: Zero Temporal Leakage ML Pipeline (45 sec)
- **Model**: Balanced Random Forest + XGBoost comparative benchmark.
- **Integrity**: Predictions at time $T$ strictly use information from $T-1, T-3, T-7, T-14$.
- **Target**: Probability of inundation $P(\text{Flood}) \in [0, 100]$.

### Slide 5: Real Historical Event Validation (1 min)
- **Event**: Catastrophic July 14, 2020 Assam Flood.
- **Results**:
  - $T-7$: Risk at 26% (Normal)
  - $T-3$: Heavy rain surge, risk jumps to 58% (Watch/Warning)
  - $T$: Peak flood inundation, risk hits 84% matching actual observed extent.
  - **Metrics**: 0.88 F1-score, 0.79 IoU, 0.94 ROC-AUC.

### Slide 6: Live Product Demo (1.5 min)
- Switch to Streamlit dashboard:
  - Demonstrate interactive Folium risk heatmap.
  - Show timeline slider ($T-7$ to $T$).
  - Toggle layers: Ground Truth Inundation vs Predicted Risk.
  - Showcase dynamic early warning alerts and KPIs.

### Slide 7: Scientific Caveats & Honesty (30 sec)
- Coarse 5.5 km precipitation resolution.
- Cloud coverage during active downpours necessitates optical pre-event baselines.
- Prototype disclaimer: Designed for decision support, not certified civil emergency broadcasting.

### Slide 8: Future Roadmap & Impact (30 sec)
- Sentinel-1 SAR integration for all-weather radar monitoring.
- Scalable deployment across all 34 districts of Assam with ASDMA API integration.
