# GCN-based Recommendation System

A Graph Convolutional Network (GCN) implementation for collaborative filtering recommendation, specifically designed for user-item interaction prediction. This project uses the Last.fm dataset to recommend items (music artists) to users based on their listening history.

**Refactored:** The entire codebase has been consolidated into a single, clean `main.py` file (~391 lines) for improved readability and maintainability.

## Table of Contents
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Dataset Description](#dataset-description)
- [GCN Architecture](#gcn-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

This project implements a recommendation system using Graph Convolutional Networks (GCN) with Bayesian Personalized Ranking (BPR) loss. The system learns user and item embeddings by propagating information through a bipartite user-item interaction graph, enabling accurate recommendation predictions.

**Key Features:**
- Graph-based collaborative filtering
- Multi-layer GCN for embedding propagation
- BPR loss optimization for implicit feedback
- Comprehensive evaluation metrics (Recall, Precision, NDCG)
- **Single-file implementation** for easy understanding and deployment

## File Structure

The entire project is now consolidated into a single `main.py` file (~391 lines), organized into clear sections:

### `main.py` - Complete Implementation

#### **Section 1: Configuration (Lines 1-50)**
- Command-line argument parsing (replaces parse.py)
- Global configuration setup (replaces world.py)
- Device selection (CUDA/CPU)
- Utility functions: `cprint()`, `set_seed()`

#### **Section 2: Utils (Lines 51-155)**
- `BPRLoss`: Class wrapping BPR loss computation and Adam optimizer
- `UniformSample_original()`: Samples training triplets (user, positive_item, negative_item)
- `minibatch()`: Generator for creating mini-batches
- `timer`: Context manager for timing operations
- Metric functions: `RecallPrecision_ATk()`, `NDCGatK_r()`, `getLabel()`

#### **Section 3: Dataset (Lines 156-235)**
- `BasicDataset`: Abstract base class
- `LastFM`: Last.fm dataset implementation
  - Loads training data from `data/lastfm/train1.txt` (42,135 interactions)
  - Loads testing data from `data/lastfm/test1.txt` (10,533 interactions)
  - Creates sparse user-item interaction matrix (1892 × 4489)
  - `getSparseGraph()`: Constructs normalized adjacency matrix with symmetric normalization: `D^(-1/2) * A * D^(-1/2)`

#### **Section 4: Model (Lines 236-315)**
- `GCN`: Graph Convolutional Network model
  - Initializes user and item embeddings (normal distribution, std=0.1)
  - `__dropout_x()`: Implements edge dropout for regularization
  - `computer()`: Core GCN propagation performing `n_layers` rounds of graph convolution: `X_(i+1) = A * X_i`
  - `getUsersRating()`: Computes prediction scores using dot product and sigmoid activation
  - `bpr_loss()`: Calculates BPR loss and L2 regularization
  - `getEmbedding()`: Retrieves embeddings for users and items

#### **Section 5: Training & Testing (Lines 316-375)**
- `BPR_train_original()`: Training procedure
  - Samples training triplets with uniform sampling
  - Creates mini-batches (default: 2048)
  - Performs backpropagation and parameter updates
  - Returns average loss
- `Test()`: Evaluation procedure
  - Generates top-K recommendations for test users
  - Excludes training items from recommendations
  - Computes evaluation metrics (Precision, Recall, NDCG)

#### **Section 6: Main Execution (Lines 376-391)**
- Sets random seed for reproducibility
- Initializes dataset, model, and optimizer
- Runs training loop for specified epochs
- Performs testing every 10 epochs
- `minibatch()`: Generator for creating mini-batches
- `shuffle()`: Shuffles training data
- `timer`: Context manager for timing code blocks
- Evaluation metrics:
  - `RecallPrecision_ATk()`: Computes recall and precision at k
  - `NDCGatK_r()`: Computes Normalized Discounted Cumulative Gain at k
  - `getLabel()`: Creates binary relevance labels for predictions

#### `world.py`
**Purpose:** Global configuration and hyperparameters.

**Configuration Parameters:**
- `bpr_batch_size`: Batch size for BPR training (default: 2048)
- `latent_dim_rec`: Embedding dimension (default: 64)
- `GCN_n_layers`: Number of GCN layers (default: 3)
- `dropout`: Whether to use dropout (default: 0/False)
- `keep_prob`: Probability of keeping edges during dropout (default: 0.6)
- `test_u_batch_size`: Batch size for testing (default: 100)
- `lr`: Learning rate (default: 0.001)
- `decay`: L2 regularization weight (default: 1e-4)
- `TRAIN_epochs`: Number of training epochs (default: 500)
- `minibatch()`: Generator for mini-batches
- `shuffle()`: Shuffles arrays while maintaining correspondence
- `timer`: Context manager for timing operations
- `RecallPrecision_ATk()`: Computes Recall and Precision at K
- `NDCGatK_r()`: Computes Normalized Discounted Cumulative Gain at K
- `getLabel()`: Converts predictions to binary labels

### Supported Command-Line Arguments

All configuration is handled through command-line arguments (previously in parse.py):

- `--bpr_batch`: BPR batch size (default: 2048)
- `--recdim`: Embedding dimension (default: 64)
- `--layer`: Number of GCN layers (default: 3)
- `--lr`: Learning rate (default: 0.001)
- `--decay`: L2 regularization weight (default: 1e-4)
- `--dropout`: Enable/disable dropout (default: 0)
- `--keepprob`: Edge keep probability (default: 0.6)
- `--testbatch`: Test batch size (default: 100)
- `--dataset`: Dataset name (default: 'lastfm')
- `--topks`: Top-K values for evaluation (default: "[20]")
- `--epochs`: Number of training epochs (default: 500)
- `--seed`: Random seed (default: 2022)

#### `requirements.txt`
**Purpose:** Lists Python dependencies required to run the project.

**Dependencies:**
- `torch==1.4.0`: Deep learning framework
- `pandas==0.24.2`: Data manipulation
- `scipy==1.3.0`: Sparse matrix operations
- `numpy==1.22.0`: Numerical computations
- `tensorboardX==1.8`: Visualization (optional)
- `scikit-learn==0.23.2`: Machine learning utilities
- `tqdm==4.48.2`: Progress bars

## Dataset Description

### Data Location
The dataset is located in the `data/lastfm/` directory.

### Data Format

#### `train1.txt`
- **Size:** 42,135 interactions
- **Format:** Three tab-separated columns per line
  - Column 1: User ID (0-1891, representing 1892 unique users)
  - Column 2: Item ID (0-4488, representing 4489 unique items/artists)
  - Column 3: Interaction count (not used in the current implementation)

**Example:**
```
424    4076    993
1311   1574    957
1671   410     146
```

#### `test1.txt`
- **Size:** 10,533 interactions
- **Format:** Same as `train1.txt`
- **Purpose:** Held-out data for evaluating recommendation quality

### Dataset Statistics

- **Users:** 1,892 unique users
- **Items:** 4,489 unique items (music artists)
- **Training Interactions:** 42,135
- **Test Interactions:** 10,533
- **Total Interactions:** 52,668
- **Sparsity:** ~0.62% (very sparse, typical for recommendation systems)

### Data Representation

The data is represented as a bipartite graph:
- **Nodes:** Users and items form two disjoint sets
- **Edges:** User-item interactions (listening history)
- **Adjacency Matrix:** A sparse 1892 × 4489 binary matrix where `UserItemNet[u, i] = 1` if user `u` interacted with item `i`

## GCN Architecture

### Overview

Graph Convolutional Networks (GCN) for recommendations leverage the graph structure of user-item interactions to learn better embeddings. The key idea is that users and items connected in the graph should have similar embeddings.

### How GCN Works

#### 1. Graph Construction
- **Bipartite Graph:** Users and items form a bipartite graph where edges represent interactions
- **Adjacency Matrix:** 
  ```
  A = [[0,      R    ],
       [R^T,    0    ]]
  ```
  where `R` is the user-item interaction matrix (1892 × 4489)
  
- **Normalization:** Apply symmetric normalization to prevent scale issues:
  ```
  A_norm = D^(-1/2) * A * D^(-1/2)
  ```
  where `D` is the degree matrix

#### 2. Embedding Initialization
- **User Embeddings:** `E_u ∈ ℝ^(n_users × d)` initialized with Normal(0, 0.1)
- **Item Embeddings:** `E_i ∈ ℝ^(n_items × d)` initialized with Normal(0, 0.1)
- **Dimension:** d = 64 by default
- **Combined:** `E^(0) = [E_u; E_i] ∈ ℝ^((n_users + n_items) × d)`

#### 3. Graph Convolution Layers

The model performs **3 layers** of graph convolution (configurable via `GCN_n_layers`):

**Layer-wise Propagation:**
```
E^(l+1) = A_norm * E^(l)
```

This operation aggregates embeddings from neighbors:
- Each user embedding is updated based on items they interacted with
- Each item embedding is updated based on users who interacted with it

**After 3 layers:**
- `E^(0)`: Initial embeddings
- `E^(1)`: 1-hop neighbor aggregation
- `E^(2)`: 2-hop neighbor aggregation
- `E^(3)`: 3-hop neighbor aggregation

#### 4. Layer Aggregation
- **Combine all layers:** Average embeddings across all layers
  ```
  E_final = mean([E^(0), E^(1), E^(2), E^(3)])
  ```
- **Benefit:** Captures information at multiple scales (local and global structure)

#### 5. Prediction
- **Split embeddings:** Separate final embeddings into users and items
- **Score computation:** For user `u` and item `i`:
  ```
  score(u, i) = σ(E_u^final · E_i^final)
  ```
  where `σ` is the sigmoid function and `·` is the dot product

#### 6. Regularization (Optional)
- **Edge Dropout:** During training, randomly drop edges with probability `1 - keep_prob` (default: 0.4)
- **Purpose:** Prevents overfitting and improves generalization

### Why GCN Works for Recommendations

1. **Collaborative Signal:** GCN propagates collaborative signals through the graph, allowing information to flow between users with similar tastes
2. **High-order Connectivity:** Multiple layers capture transitive relationships (e.g., users who liked similar items but haven't directly interacted)
3. **Parameter Efficiency:** Shares parameters across all nodes, reducing overfitting
4. **Sparse Data Handling:** Effectively handles sparse interaction data through graph structure

## Training and Evaluation

### Training Process

#### 1. Loss Function: Bayesian Personalized Ranking (BPR)

BPR is designed for implicit feedback (e.g., clicks, purchases, plays) where we only observe positive interactions.

**Core Assumption:** Users prefer items they interacted with over items they didn't.

**BPR Loss:**
```
Loss_BPR = Σ -log(σ(score_pos - score_neg))
```

where:
- `score_pos = E_u · E_i_pos`: Score for positive (observed) item
- `score_neg = E_u · E_i_neg`: Score for negative (unobserved) item
- `σ`: Sigmoid function

**Implementation:**
```python
loss = mean(softplus(neg_scores - pos_scores))
```
where `softplus(x) = log(1 + exp(x))` is a smooth approximation of `max(0, x)`

#### 2. Regularization

**L2 Regularization:** Prevents overfitting by penalizing large embedding values
```
Loss_reg = (||E_u^(0)||^2 + ||E_i_pos^(0)||^2 + ||E_i_neg^(0)||^2) / (2 * batch_size)
```

**Total Loss:**
```
Loss_total = Loss_BPR + λ * Loss_reg
```
where `λ = decay = 1e-4`

#### 3. Optimization

- **Optimizer:** Adam with learning rate `lr = 0.001`
- **Batch Size:** 2048 user-item-negative_item triplets per batch
- **Sampling Strategy:** Uniform random sampling
  - For each user, randomly select one positive item they interacted with
  - Randomly sample one negative item they haven't interacted with
- **Training Epochs:** 500 (configurable)

#### 4. Training Loop

For each epoch:
1. Sample training triplets: (user, positive_item, negative_item)
2. Shuffle triplets
3. Create mini-batches
4. For each batch:
   - Forward pass: Compute embeddings via GCN
   - Compute BPR loss and regularization loss
   - Backward pass: Compute gradients
   - Update parameters using Adam optimizer
5. Report average loss for the epoch

### Evaluation Process

#### 1. Evaluation Frequency
- Test every **10 epochs** during training
- Final evaluation after training completes

#### 2. Evaluation Metrics

##### Recall@K
**Definition:** Proportion of relevant items that appear in the top-K recommendations

```
Recall@K = (# relevant items in top-K) / (# total relevant items)
```

**Example:** If a user has 5 relevant items in test set and 3 appear in top-20, Recall@20 = 3/5 = 0.6

##### Precision@K
**Definition:** Proportion of recommended items that are relevant

```
Precision@K = (# relevant items in top-K) / K
```

**Example:** If 3 out of top-20 recommendations are relevant, Precision@20 = 3/20 = 0.15

##### NDCG@K (Normalized Discounted Cumulative Gain)
**Definition:** Measures ranking quality with position-based discount

```
DCG@K = Σ (rel_i / log_2(i + 1))  for i in 1..K
NDCG@K = DCG@K / IDCG@K
```

where:
- `rel_i = 1` if item at position i is relevant, 0 otherwise
- `IDCG@K`: Ideal DCG (best possible ranking)

**Interpretation:** NDCG@K ∈ [0, 1], higher is better. Penalizes relevant items appearing at lower ranks.

#### 3. Evaluation Procedure

For each test user:
1. **Get embeddings:** Compute user and all item embeddings using trained GCN
2. **Compute scores:** Calculate scores for all items: `score = σ(E_u · E_i)`
3. **Exclude training items:** Set scores of training items to very low value (-1024)
4. **Rank items:** Select top-K items with highest scores (default K=20)
5. **Compare with ground truth:** Check which top-K items appear in the test set
6. **Compute metrics:** Calculate Recall@K, Precision@K, NDCG@K

**Final Scores:** Average metrics across all test users

### Accuracy Interpretation

**What the metrics tell us:**

- **Recall@20 = 0.15** means the model finds 15% of relevant items in top-20 recommendations
- **Precision@20 = 0.08** means 8% of top-20 recommendations are actually relevant
- **NDCG@20 = 0.12** indicates moderate ranking quality (0=worst, 1=perfect)

**Trade-offs:**
- Higher K increases Recall but decreases Precision
- BPR loss optimizes for ranking quality, which aligns well with NDCG
- Sparse data makes achieving high precision challenging

## Installation

### Prerequisites
- Python 3.6 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd recommend-GCN-missing
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** You may need to adjust PyTorch version based on your CUDA version:
   ```bash
   # For CUDA 10.1
   pip install torch==1.4.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
   
   # For CPU only
   pip install torch==1.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```

## Usage

### Basic Training

Run with default parameters:
```bash
python main.py
```

### Custom Configuration

**Example 1:** Change embedding dimension and learning rate
```bash
python main.py --recdim 128 --lr 0.0005
```

**Example 2:** Use dropout regularization
```bash
python main.py --dropout 1 --keepprob 0.6
```

**Example 3:** Train for fewer epochs with larger batch size
```bash
python main.py --epochs 200 --bpr_batch 4096
```

**Example 4:** Evaluate at multiple K values
```bash
python main.py --topks "[10,20,50]"
```

### Complete Example with All Parameters
```bash
python main.py \
  --bpr_batch 2048 \
  --recdim 64 \
  --layer 3 \
  --lr 0.001 \
  --decay 0.0001 \
  --dropout 0 \
  --keepprob 0.6 \
  --testbatch 100 \
  --epochs 500 \
  --topks "[20]" \
  --seed 2022
```

### Expected Output

During training, you'll see output like:
```
>>SEED: 2022
loading [last fm]
LastFm Sparsity : 0.006219...
gcn is already to go(dropout:0)
EPOCH[1/500]loss1.234-|Sample:2.45|
EPOCH[2/500]loss1.156-|Sample:2.38|
...
EPOCH[10/500]loss0.892-|Sample:2.41|
[TEST]
{'precision': array([0.0123]), 'recall': array([0.0456]), 'ndcg': array([0.0678])}
```

### Monitoring Training

- **Loss decrease:** Should gradually decrease over epochs (initial ~1.2 → final ~0.5-0.8)
- **Test metrics:** Should gradually improve over epochs
- **Training time:** ~5-10 seconds per epoch on GPU, ~30-60 seconds on CPU

### Saving Results

To save results, modify `main.py` or redirect output:
```bash
python main.py > training_log.txt 2>&1
```

## Understanding the Code

### Key Code Sections with Missing Implementations

The original project had placeholders for students to implement:

1. **In `dataloader.py`:** Loading Last.fm data and constructing UserItemNet
2. **In `model.py`:** 
   - GCN graph convolution in `computer()`
   - BPR loss calculation in `bpr_loss()`
   - User rating computation in `getUsersRating()`
3. **In `main.py`:** Training and testing loop

These sections are now implemented and can serve as a reference for understanding GCN-based recommendations.

## References

### Papers
- **LightGCN:** *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* (SIGIR 2020)
- **BPR:** *BPR: Bayesian Personalized Ranking from Implicit Feedback* (UAI 2009)
- **GCN:** *Semi-Supervised Classification with Graph Convolutional Networks* (ICLR 2017)

### Concepts
- **Collaborative Filtering:** Recommending items based on user-item interaction patterns
- **Implicit Feedback:** Learning from positive-only signals (clicks, views) without explicit ratings
- **Graph Neural Networks:** Neural networks that operate on graph-structured data
- **Pairwise Ranking:** Learning to rank by comparing pairs of items

## License

This project is for educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
