# 🎯 Graph-Based Recommendation System

This project implements a recommendation engine using Graph Neural Networks (GNN), specifically **GraphSAGE**, to model complex user-item interactions as a bipartite graph.

## 🚀 Features
- User-item graph creation using NetworkX
- Graph conversion to PyTorch Geometric format
- Node feature engineering for users and movies
- GNN model training and evaluation with classification metrics

## 🧰 Tech Stack
- Python
- PyTorch & PyTorch Geometric
- NetworkX
- Pandas, NumPy
- Matplotlib
- Scikit-learn

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. **Create a virtual environment and activate it**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install the dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the scripts in order**
- `build_graph.py`: Constructs the NetworkX bipartite graph from MovieLens data
- `recommend.py`: Converts it to PyTorch Geometric HeteroData format
- `algorithm.py`: Trains and evaluates the GraphSAGE-based GNN model

## 📊 Dataset Used
[MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/100k/)

## 🧠 Author
**Ragam Vishal**  
Feel free to connect and explore this graph-based learning project!
