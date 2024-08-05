

# Unsupervised Clustering on Mall Customers

## Project Overview
This project uses unsupervised machine learning (KMeans clustering) to segment customers based on their shopping data. The dataset, `mall_customers.csv`, includes customer attributes such as age, annual income, and spending scores. The project aims to identify distinct groups or clusters of customers to tailor marketing strategies effectively.

## Getting Started

### Prerequisites
To run this project, you need Python installed along with several packages for data handling and machine learning. The primary libraries used are:
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

You can install these using pip:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Installation
Clone the repository to get started with the project:

```bash
git clone https://github.com/dhrupadDJ/Unsupervised-Clustering
cd unsupervised_clustering
```

### File Structure
- `data_preprocessing/`
  - `load_data.py` - Module for loading data.
  - `preprocessdata.py` - Module for data preprocessing utilities including statistics and visualizations.
- `models/`
  - `kmeans_clustering.py` - Contains functions related to KMeans clustering.
- `evaluation/`
  - `evaluation.py` - Evaluation metrics and plotting functions for cluster analysis.
- `mall_customers.csv` - Dataset used for clustering.

### Running the Code
To execute the main script, navigate to the project directory and run:

```bash
python main.py
```

Ensure `main.py` is replaced with the actual name of your script if it's different.

## Usage
This analysis helps in understanding customer behavior by segmenting them into clusters based on similarities in their shopping patterns. Such insights can aid businesses in crafting personalized marketing strategies to target specific customer groups more effectively.

## Contributing
Contributions to this project are welcome. To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature_branch`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature_branch`).
5. Open a new Pull Request.

