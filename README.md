# Parking Allocation Using Linear Algebra

A university Linear Algebra mini-project web application demonstrating urban parking allocation grounded in real-world sensor data from the **UCI "Parking Birmingham"** dataset (id 482, CC BY 4.0, Stolfi 2017).

Every core linear algebra routine is implemented **from scratch** in pure Python using NumPy arrays (`np.linalg` is used solely in test suites for cross-validation). Every mathematical stage provides explicit **Concept**, **Purpose**, and **Outcome** explanations.

---

## 1. Quick Start (Single-Command Run)

### Prerequisites
- Python 3.10+ (tested on Python 3.11.9)
- Node.js 18+ (tested on Node.js v24.18.0)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```
*(Packages: `fastapi`, `uvicorn`, `numpy`, `pandas==2.2.3`, `pytest`, `sympy`)*

### Step 2: Build the React Frontend
```bash
cd frontend
npm install
npm run build
cd ..
```

### Step 3: Run the Single-Command Unified Server
```bash
python run.py
```
Open your browser at: **`http://127.0.0.1:8000`**

FastAPI automatically serves both the JSON API and the built React single-page application from `frontend/dist/`.

---

## 2. Running Automated Tests

To cross-validate all custom from-scratch linear algebra algorithms against NumPy and SymPy:
```bash
python -m pytest backend/tests/test_linalg.py -v
```

All 9 test suites verify:
- Gaussian elimination & RREF with partial pivoting vs SymPy `Matrix.rref()`
- $PA = LU$ decomposition & triangular solvers vs `np.linalg.solve`
- Rank and Nullity theorem ($\text{rank} + \text{nullity} = n$) vs SymPy `nullspace()`
- Classical & Modified Gram-Schmidt QR vs `np.linalg.qr` (up to column sign)
- Orthogonal projection residual orthogonality $\|A^T (b - \hat{b})\|_\infty < 10^{-10}$
- Ordinary Least Squares autoregressive regression vs `np.linalg.lstsq` (diff $< 10^{-12}$)
- Power iteration & Hotelling's deflation top eigenpairs vs `np.linalg.eigh`
- Cyclic Jacobi complete symmetric diagonalization & Frobenius reconstruction monotonicity
- Vector norms ($L_1$ Manhattan, $L_2$ Euclidean, $L_\infty$ Chebyshev)

---

## 3. Real Data vs. Modelled System

| Dimension | Real Data (Grounding) | Our Spatial Model |
|---|---|---|
| **Data Source** | UCI Machine Learning Repository (Dataset 482, Stolfi 2017). 35,717 records across 30 car parks, 30-min intervals. | 10 $\times$ 12 parking lot spatial grid (120 total stalls). |
| **Occupancy Type** | Analyzed against capacities (220–4675); confirmed occupancy is an **integer vehicle count** (mean 642, max 4327), not a fractional rate. | Spot state matrix $L \in \{0, 1, 2\}^{10 \times 12}$ (0=Free, 1=Occupied, 2=Blocked). |
| **Car Parks** | 4 most complete car parks: Bullring Markets (cap 577), Town Hall (cap 387), Broad Street (cap 470), Mailbox (cap 687). Total capacity = 2,121. | 4 Zones sized proportionally to real capacities: Zone A (33 spots), Zone B (22 spots), Zone C (26 spots), Zone D (39 spots). Sum = 120 spots. |
| **Missing Values** | Only 1 missing record in Broad Street across 1,307 timestamps; handled via forward-fill (`ffill`) for temporal continuity. | Fixed obstacles / structural columns placed at (3,3), (3,8), (6,3), (6,8). |
| **Spatial Geometry** | The dataset contains no GPS or stall coordinates. | Stalls assigned coordinate matrix $P \in \mathbb{R}^{120 \times 2}$. Vehicle entrance at $(0,0)$; pedestrian destinations at $(9,1), (0,11), (9,11), (5,6)$. |
| **Amenities** | Not recorded in raw sensor data. | Spot features $F \in \mathbb{R}^{120 \times 4}$ (Standard, Compact, EV Charging, Accessible/Disabled). |

---

## 4. 10-Stage Linear Algebra Mathematical Pipeline

1. **Matrix Representation**: Encapsulates time series into observation matrix $X \in \mathbb{R}^{1307 \times 5}$ (4 car parks + derived `Total`), lot grid $L \in \mathbb{R}^{10 \times 12}$, and coordinate matrix $P \in \mathbb{R}^{120 \times 2}$.
2. **RREF & LU Factorization**: From-scratch Gaussian elimination with partial row pivoting. Factors square gram systems $PA = LU$ and solves $Ax = b$ via triangular substitution without explicit matrix inversion.
3. **Rank, Nullity & Null Space**: Demonstrates the Rank-Nullity Theorem: $\text{rank}(X) = 4, \text{nullity}(X) = 1$ ($4 + 1 = 5$). Null vector $v = (1, 1, 1, 1, -1)^T$ proves linear dependency is strictly **by construction** because $\text{Total} = \sum_{i=1}^4 c_i$.
4. **Column Space Basis**: Extracts pivot columns $[0, 1, 2, 3]$ to form full column-rank basis matrix $A \in \mathbb{R}^{1307 \times 4}$, eliminating multicollinearity.
5. **Gram-Schmidt QR**: Compares Classical Gram-Schmidt (defect $2.07 \times 10^{-14}$) and Modified Gram-Schmidt (defect $2.60 \times 10^{-15}$), proving MGS's numerical stability.
6. **Orthogonal Projection**: Solves normal equations $(A^T A) \hat{x} = A^T b$ via custom LU solver to project external demand $b$ onto $\text{col}(A)$. Verifies residual orthogonality $\|A^T (b - \hat{b})\|_\infty < 10^{-7} \approx 0$.
7. **Least Squares Autoregressive Forecasting**: Time-ordered 80/20 train/test split. Forecasts next 30-min occupancy from previous slots ($t-1, t-2$). Mean test RMSE = 48.37 vehicles. Matches `np.linalg.lstsq` within $7.39 \times 10^{-13}$. Outputs predicted zone loads $\hat{h}$.
8. **Eigendecomposition**: Power iteration with Hotelling's deflation extracts top eigenvalues $\lambda_1 = 35053.96$ and $\lambda_2 = 11451.78$ from sample covariance $S$, matching `np.linalg.eigh` within machine precision.
9. **Diagonalization & Spectral Reconstruction**: Diagonalizes $S = V \Lambda V^T$ via cyclic Jacobi algorithm. Shows $k=2$ components explain $90.23\%$ of total system variance, reducing Frobenius error by $70\%$.
10. **Multi-Norm Cost Allocation**: Formulates cost functional $C(v, s) = \alpha \|p_s - d\|_1 + \beta \|p_s - e\|_2 + \gamma \hat{h}[\text{zone}(s)]$. Verifies vehicle compatibility (Standard, Compact, EV, Accessible) and selects $\text{argmin}_s C(v, s)$. Compares $L_1$ (Manhattan), $L_2$ (Euclidean), and $L_\infty$ (Chebyshev) metrics.

---

## 5. Allocation Strategy Benchmark

Monte Carlo simulation across identical pseudo-random vehicle arrival streams evaluates three operational policies:
- **S1 (Nearest Entrance)**: $\alpha=0, \beta=1, \gamma=0$. Minimizes driving distance ($L_2$ Euclidean) from gate. Result: Long pedestrian walks (mean 10.68 units) and heavy perimeter congestion.
- **S2 (Nearest Destination)**: $\alpha=1, \beta=0, \gamma=0$. Minimizes walking distance ($L_1$ Manhattan). Result: Short walking initially (mean 4.12 units) but severe destination quadrant saturation (zone std 28.5%).
- **S3 (Weighted Cost + Stage 7 Prediction)**: $\alpha=1, \beta=0.4, \gamma=2.5$. Balances walk, drive, and Stage 7 forecast load $\hat{h}$. Result: Low walking distance (mean 5.34 units) and balanced zone utilization (zone std 14.8%, a $50\%$ improvement).

---

## 6. Project Structure

```
Math_Project/
├── backend/
│   ├── data/
│   │   └── dataset.csv          # Committed copy of UCI Parking Birmingham dataset
│   ├── core/
│   │   ├── linalg.py            # FROM-SCRATCH linear algebra routines (no np.linalg)
│   │   ├── data_processor.py    # Long-to-wide pivot, imputation, 4 car parks + Total
│   │   ├── lot_model.py         # 10x12 grid, zones, compatibility, cost & simulation
│   │   └── pipeline.py          # 10-stage mathematical pipeline executor & cache
│   ├── tests/
│   │   └── test_linalg.py       # Pytest suite cross-checking against NumPy & SymPy
│   └── main.py                  # FastAPI app & static file SPA server
├── docs/
│   └── viva.md                  # Examiner viva defense guide, real run figures & 3-min script
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LiveLot.jsx             # Page 1: Interactive grid & spot allocator
│   │   │   ├── MathPipeline.jsx        # Page 2: 10-stage stepper with Concept/Purpose/Outcome
│   │   │   └── StrategyComparison.jsx  # Page 3: Monte Carlo benchmark & bar charts
│   │   ├── App.jsx                     # Root navigation & state manager
│   │   └── index.css                   # Custom modern dark-theme styles
│   ├── package.json
│   └── vite.config.js
├── run.py                       # Single-command launcher
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 7. Citation & Attribution

Data: **Stolfi, D. H. (2017). Parking Birmingham Dataset. UCI Machine Learning Repository.** Available under Creative Commons Attribution 4.0 International (CC BY 4.0).

---

## 8. Contributors

See [CONTRIBUTORS.md](./CONTRIBUTORS.md) for the full list of contributors and how to contribute to this project.
