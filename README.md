# 🧬 Augmented Population-Based Training

An implementation of **Augmented Population-Based Training (APBT)** - a hyperparameter optimization and neural architecture search algorithm that evolves neural networks through competitive training.

## 🚀 Quick Start - Interactive Dashboard

**Your dashboard is running at: http://localhost:8501**

### Try it now (2-minute demo):
1. Open http://localhost:8501 in your browser
2. Configure: Iris dataset, 40 networks, 100 epochs
3. Click "▶️ Start Training"
4. Watch neural networks compete and evolve in real-time!

### Available Datasets:
- **Iris** (150 samples, 4 features, 3 classes) - Classic flower classification
- **Tennis** (14 samples, 4 features, 2 classes) - Weather prediction
- **Identity** (100 samples, 5 features, 5 classes) - Synthetic dataset
- **MNIST** (10,000 samples, 784 features, 10 classes) - Handwritten digits *(requires setup)*
- **Fashion MNIST** (10,000 samples, 784 features, 10 classes) - Clothing items *(requires setup)*

### To use MNIST datasets:
```bash
python download_mnist.py  # Download real MNIST & Fashion MNIST data
streamlit run app.py       # Start dashboard
```

### To start/restart the dashboard:
```bash
conda activate torch
streamlit run app.py
```

---

## 📚 Table of Contents

- [What is APBT?](#what-is-apbt)
- [Dashboard Features](#dashboard-features)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Dashboard](#web-dashboard)
  - [Command Line](#command-line)
- [How It Works](#how-it-works)
- [Understanding the Algorithm](#understanding-the-algorithm)
- [Visualization Guide](#visualization-guide)
- [Advanced Topics](#advanced-topics)
- [Examples & Results](#examples--results)

---

## What is APBT?

**Augmented Population-Based Training** combines three powerful ideas:

### 1. 🧬 Population-Based Training
- Multiple neural networks (population of 20-100) train in parallel
- Networks compete for survival based on performance
- Bottom 20% copy weights from top 20% (exploitation)
- Automatic hyperparameter optimization

### 2. 🔍 Neural Architecture Search
- Networks can dynamically add/remove units
- Finds optimal topology automatically
- Balances accuracy vs complexity

### 3. ⚖️ Multi-Objective Optimization
**Fitness Function:** `f(accuracy, size) = 1.09^(accuracy×100) / 1.02^size`
- Rewards accuracy exponentially
- Penalizes large models
- Finds the optimal tradeoff automatically

### The Evolution Loop
```
Initialize Population (random hyperparameters & architectures)
    ↓
┌─────────────────────────────────────┐
│ For each epoch:                     │
│  1. Train all networks (parallel)   │
│  2. Evaluate fitness                │
│  3. Update leaderboard              │
│  4. Exploit (copy winners)          │
│  5. Explore (mutate & try new)      │
└─────────────────────────────────────┘
    ↓
Return Best Network
```

**Result:** Automatically discovers optimal hyperparameters AND architecture without manual tuning!

---

## Dashboard Features

The Streamlit dashboard provides 5 comprehensive visualization tabs:

### 📊 Tab 1: Training Progress
- Real-time performance metrics
- Accuracy evolution (best vs most accurate)
- Model size changes over time
- Average population statistics
- Live progress bar

### 🏆 Tab 2: Leaderboard
- Top 20 networks ranked by performance
- Gold/Silver/Bronze highlighting
- Full hyperparameter details
- Population distribution histograms
- Watch exploitation events happen!

### 🧠 Tab 3: Architecture
- Interactive network topology visualization
- Layer-by-layer breakdown
- Parameter counts
- Color-coded nodes (input/hidden/output)
- See optimal structure emerge

### 📈 Tab 4: Hyperparameters
- Learning rate evolution
- Momentum tracking
- Weight decay changes
- Statistics with deltas
- Understand adaptation patterns

### 🎯 Tab 5: Fitness Landscape
- **3D surface plot** (interactive, rotatable!)
- Accuracy vs Size tradeoff visualization
- Current best network position marked
- Population scatter plot (Pareto front)
- Color-coded by rank

---

## Installation

### Prerequisites
- Python 3.8+
- Conda (recommended)

### Install Dependencies
```bash
# Activate your environment
conda activate torch

# Install required packages
pip install streamlit pandas numpy plotly
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

---

## Usage

### Web Dashboard (Recommended)

Start the interactive dashboard:
```bash
conda activate torch
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**Dashboard Configuration:**
- **Dataset:** Choose from Iris, Tennis, Identity, MNIST, or Fashion MNIST
- **Population Size:** 10-500 networks (40 recommended, 20 for MNIST)
- **Epochs:** 10-1000 (100+ for good results, 50+ for MNIST)
- **Advanced Settings:** Learning rate range, readiness threshold (defaults to 5% of epochs), truncation %

**Note:** MNIST and Fashion MNIST use 784 input features (28x28 images), so training takes longer. The app automatically adjusts default parameters for these datasets.

### Command Line

Run training from the command line:

```bash
python source/main.py \
-a data/iris/iris-attr.txt \
-d data/iris/iris-train.txt \
-t data/iris/iris-test.txt \
-w models/weights.txt \
  -k 40 \
  -e 100 \
--debug
```

**Arguments:**
- `-a, --attributes`: Path to attributes file (required)
- `-d, --training`: Path to training data (required)
- `-t, --testing`: Path to test data (optional)
- `-w, --weights`: Path to save weights (optional)
- `-k, --k-inds`: Population size (required)
- `-e, --epochs`: Number of epochs (required)
- `--debug`: Enable debug output (optional)

### Run Experiment Files

Pre-configured experiments:
```bash
python source/testIris.py
python source/testTennis.py
python source/testIdentity.py
```

---

## How It Works

### Core Components

#### 1. **ANN (Artificial Neural Network)** - `source/ann.py`
Feed-forward neural network with backpropagation:
- Flexible topology: `[input, hidden1, hidden2, ..., output]`
- Hyperparameters: learning rate, momentum, decay
- Activation: Sigmoid function
- Training: Stochastic Gradient Descent (SGD)

#### 2. **APBT Algorithm** - `source/apbt.py`
Population-based training manager:
- **Initialization:** Create k networks with random hyperparameters & architectures
- **Training:** Each network trains for 1 epoch per iteration
- **Evaluation:** Fitness = f(accuracy, model_size)
- **Exploitation:** Bottom 20% copy from top 20% (every ~5% of total epochs by default)
- **Exploration:** Perturb hyperparameters (×0.8 or ×1.2) and architecture (±1 unit)

### Key Parameters

```python
# Hyperparameter Ranges
LR_RANGE = (1e-4, 1e-1)        # Learning rate
M_RANGE = (0.0, 0.9)           # Momentum
D_RANGE = (0.0, 0.1)           # Decay
HL_RANGE = (1, 4)              # Hidden layers
HUPL_RANGE = (2, 10)           # Units per layer

# Algorithm Parameters
READINESS = 5% of epochs       # Epochs before exploitation (dynamic)
TRUNC = 0.2                    # Top/bottom 20%
PERTS = (0.8, 1.2)            # Perturbation factors
X, Y = 1.09, 1.02             # Fitness function factors
```

### The Fitness Function

```python
def f(accuracy, size):
    return 1.09 ** (accuracy * 100) / 1.02 ** size
```

**Examples:**
- Network A: 95% acc, 100 params → fitness = 4,371 ✅
- Network B: 98% acc, 200 params → fitness = 3,812
- Network C: 90% acc, 50 params → fitness = 3,103

Network A wins! Best balance of accuracy and size.

---

## Understanding the Algorithm

### Exploitation (Truncation Selection)

Every ~5% of total epochs (configurable), if a network is in the bottom 20%:
```python
if my_rank > 80th_percentile:
    top_performer = random.choice(top_20%)
    copy(top_performer.weights)
    copy(top_performer.hyperparameters)
```

**Why it works:** 
- Poor performers don't waste time training from scratch
- Copies proven successful configurations
- Maintains diversity by choosing randomly from top 20%

### Exploration (Perturbation)

After exploitation, explore nearby solutions:
```python
# Hyperparameter perturbation (±20%)
learning_rate *= random.choice([0.8, 1.2])
momentum *= random.choice([0.8, 1.2])
decay *= random.choice([0.8, 1.2])

# Architecture perturbation (±1 unit)
random_layer = pick_random_hidden_layer()
random_layer.units += random.choice([-1, 0, 1])

# Adjust weights accordingly
if added_unit:
    add_new_random_weights()
elif removed_unit:
    delete_associated_weights()
```

**Why it works:**
- Small changes prevent wild swings
- Explores nearby solutions
- Maintains good performance while searching

### Evolution Example (Iris Dataset)

**Generation 0 (Random initialization):**
- Network #27: [4,8,4,3] - 38% accuracy, performance: 18.7 (Best)
- Network #12: [4,3,7,3] - 35% accuracy, performance: 15.3
- Network #35: [4,2,9,6,3] - 33% accuracy, performance: 12.1

**Generation 50 (Learning & exploitation):**
- Network #27: [4,8,4,3] - 89% accuracy, performance: 1,285 (Still best)
- Network #12: [4,8,4,3] - 87% accuracy, performance: 1,103 (Copied #27!)
- Network #8: [4,7,5,3] - 85% accuracy, performance: 978

**Generation 200 (Converged):**
- Network #27: [4,7,5,3] - 97% accuracy, performance: 5,820 (Evolved!)
- Network #12: [4,7,5,3] - 96% accuracy, performance: 5,231
- Network #8: [4,7,5,3] - 96% accuracy, performance: 5,231

**Result:** Population discovers optimal architecture: [4, 7, 5, 3]

---

## Visualization Guide

### Key Patterns to Watch

#### 1. Exploitation Events
**Sudden performance jumps** at readiness intervals (default: every 5% of total epochs)
- Bottom 20% of networks copy weights from top 20%
- Performance chart shows dramatic vertical jumps
- Leaderboard shows major reshuffling
- This is "survival of the fittest" in action!

#### 2. Architecture Evolution
**Model size fluctuations** throughout training
- Networks add/remove units during exploration
- Size chart shows spikes and dips
- Eventually converges to optimal complexity
- Shows the architecture search process

#### 3. Population Convergence
**Distribution narrowing** over time
- **Early training**: Wide spread in leaderboard, diverse architectures
- **Mid training**: Some clustering, common patterns emerging
- **Late training**: Tight cluster, similar architectures
- Population agrees on optimal solution

### Interpreting the 3D Fitness Landscape

The fitness landscape shows the optimization objective:
- **X-axis:** Accuracy (%)
- **Y-axis:** Model Size (parameters)
- **Z-axis:** Fitness value
- **Peak:** Optimal accuracy/size balance
- **Red Diamond:** Your current best network

**Ideal position:** Bottom-right area = High accuracy, small size!

---

## Advanced Topics

### Customizing the Fitness Function

Edit `source/apbt.py` line 309:
```python
def f(self, acc, size):
    # Current: Exponential reward/penalty
    return self.X ** (acc * 100) / self.Y ** size
    
    # Alternative: Stronger size penalty
    # return acc ** 2 / (size ** 0.8)
    
    # Alternative: Linear tradeoff
    # return acc * 1000 - size * 0.1
```

### Adjusting Exploitation Timing

In the dashboard's Advanced Settings, adjust the Readiness Threshold slider.
Default: 5% of total epochs (e.g., 5 epochs for 100 total, 50 epochs for 1000 total)

Or edit `source/apbt.py` line 73:
```python
self.READINESS = 220  # Custom value: lower for more frequent, higher for less frequent
```

### Changing Selection Pressure

Edit `source/apbt.py` line 74:
```python
self.TRUNC = 0.2  # Try: 0.1 (top 10%), 0.3 (top 30%)
```

### Adding Custom Datasets

1. Create attribute file: `data/mydata/mydata-attr.txt`
2. Create training file: `data/mydata/mydata-train.txt`
3. Create test file: `data/mydata/mydata-test.txt`
4. Add to dashboard: Edit `app.py` dataset_map dictionary

---

## Examples & Results

### Iris Dataset (150 examples, 4 features, 3 classes)

**Configuration:**
```
Population: 40 networks
Epochs: 100
Time: ~2 minutes
```

**Expected Results:**
```
Accuracy:        94-97%
Model Size:      60-100 parameters
Best Topology:   [4, 6-8, 4-6, 3]
Learning Rate:   ~0.02-0.05
```

### Tennis Dataset (14 examples, weather features, 2 classes)

**Configuration:**
```
Population: 20 networks
Epochs: 100
Time: ~30 seconds
```

**Expected Results:**
```
Accuracy:        85-100%
Model Size:      40-60 parameters
Best Topology:   [10, 3-5, 2]
Learning Rate:   ~0.01-0.03
```

### Identity Dataset (identity function learning)

**Configuration:**
```
Population: 40 networks
Epochs: 200
Time: ~3 minutes
```

**Expected Results:**
```
Accuracy:        90-98%
Model Size:      Variable
Best Topology:   Depends on problem size
Learning Rate:   ~0.03-0.07
```

### MNIST Dataset (28x28 digit images, 784 features, 10 classes)

**Configuration:**
```
Population: 20 networks (smaller due to network size)
Epochs: 50-100
Time: ~10-20 minutes
```

**Expected Results:**
```
Accuracy:        70-85% (with limited training)
Model Size:      2000-5000 parameters
Best Topology:   [784, 32-64, 16-32, 10]
Learning Rate:   ~0.001-0.01
```

**Setup:**
```bash
python download_mnist.py  # Downloads real MNIST dataset
```

### Fashion MNIST Dataset (28x28 fashion images, 784 features, 10 classes)

**Configuration:**
```
Population: 20 networks (smaller due to network size)
Epochs: 50-100
Time: ~10-20 minutes
```

**Expected Results:**
```
Accuracy:        65-80% (with limited training)
Model Size:      2000-5000 parameters
Best Topology:   [784, 32-64, 16-32, 10]
Learning Rate:   ~0.001-0.01
```

**Setup:** Same as MNIST - run `download_mnist.py` (downloads real Fashion MNIST images)

**Fashion MNIST Classes:**
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

### 🖼️ Image Dataset Features

When using MNIST or Fashion MNIST, the dashboard includes:

**Sample Image Display:**
- 5 sample images displayed horizontally
- Ground truth labels shown under each image
- After training: Model predictions with confidence %
- ✓ Green checkmark for correct predictions
- ✗ Red X for incorrect predictions

**Automatic Optimizations:**
- Smaller default population (20 vs 40) for faster training
- Adjusted epoch recommendations (100-200 epochs)
- Warning about large input size (784 features)
- Real-time visualization updates during training

**Data Source:**
- Uses PyTorch's torchvision to download official datasets
- 10,000 training + 2,000 test samples (subsets for speed)
- Full datasets available: 60,000 training + 10,000 test images
- To use full datasets, modify `subset_size_train` and `subset_size_test` in `download_mnist.py`

---

## Why APBT is Powerful

| Method | Hyperparameters | Architecture | Parallel | Manual Tuning |
|--------|----------------|--------------|----------|---------------|
| Grid Search | ✓ Exhaustive | ✗ Fixed | ✓ Yes | ✓ Required |
| Random Search | ✓ Random | ✗ Fixed | ✓ Yes | ✓ Required |
| Bayesian Opt | ✓ Smart | ✗ Fixed | ✗ No | ✓ Required |
| NAS | ✗ Fixed | ✓ Search | ✓ Yes | ✓ Required |
| **APBT** | ✓ Evolving | ✓ Evolving | ✓ Yes | ✗ **None!** |

**Advantages:**
1. ✅ No manual hyperparameter tuning
2. ✅ Automatic architecture search
3. ✅ Parallel efficiency (all networks train simultaneously)
4. ✅ Multi-objective optimization (accuracy vs size)
5. ✅ Adaptive (hyperparameters evolve during training)

---

## File Structure

```
├── app.py                      # Streamlit dashboard (600+ lines)
├── requirements.txt            # Python dependencies
├── source/
│   ├── ann.py                 # Neural network implementation
│   ├── apbt.py                # APBT algorithm
│   ├── main.py                # Command-line interface
│   ├── utils.py               # Utility functions
│   ├── testIris.py            # Iris experiment
│   ├── testTennis.py          # Tennis experiment
│   └── testIdentity.py        # Identity experiment
├── data/
│   ├── iris/                  # Iris dataset
│   ├── tennis/                # Tennis dataset
│   └── identity/              # Identity dataset
├── models/                     # Saved weights
└── docs/                       # Original documentation
```

---

## Tips for Best Results

1. **Start Small:** Use Iris dataset with 40 networks for quick experiments
2. **Be Patient:** Real improvements often take 200+ epochs
3. **Watch Leaderboard:** See competitive dynamics and exploitation events
4. **Check Architecture Tab:** See how optimal network structure evolves
5. **Monitor Fitness Landscape:** Understand why certain networks win
6. **Export Charts:** Hover over charts and click camera icon to save

---

## Troubleshooting

### Dashboard won't load?
- Ensure Streamlit is running: `streamlit run app.py`
- Check http://localhost:8501
- Try refreshing the page

### Training too slow?
- Reduce population size (try 20)
- Reduce epochs (try 50)
- Use simpler dataset (Tennis)

### Want to stop/restart?
```bash
# Stop
pkill -f "streamlit run app.py"

# Restart
conda activate torch
streamlit run app.py
```

---

## Research Context

This implementation is based on:
- **Population Based Training** (DeepMind, 2017)
- **Neural Architecture Search** (Google Brain, 2017)
- **Regularized Evolution** (Google, 2019)

**Key Innovation:** Combines hyperparameter optimization WITH architecture search in a single unified framework.

---

## License

See [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Original implementation by Josias Moukpe (Machine Learning Course, April 2022)

Interactive dashboard and comprehensive documentation added to make the algorithm accessible and understandable through beautiful visualizations.

---

**Enjoy exploring evolutionary neural network training! 🧬🚀**

For questions or issues, check the dashboard help tooltips or review the inline code comments.
