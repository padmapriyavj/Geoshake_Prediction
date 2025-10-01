# 🌍 GeoShake Prediction

A machine learning pipeline for **earthquake data analysis and prediction** using **graph-based deep learning**. This project retrieves earthquake event data from the **USGS Earthquake API**, constructs a **spatial graph** of earthquake occurrences, and prepares the data for predictive modeling using **PyTorch Geometric**.

## 📌 Features

* 🔗 Fetch real-time earthquake data (GeoJSON) from **USGS**
* 📊 Convert raw data into a structured **DataFrame**
* 🌐 Build a **K-Nearest Neighbors (KNN) Graph** to connect earthquake events based on spatial proximity
* ⚡ Normalize features (latitude, longitude, depth, timestamp, magnitude) for ML training
* 🧠 Prepare graph datasets for use with **Graph Neural Networks (GNNs)** via **PyTorch Geometric**
* 📈 Scalable pipeline for **earthquake prediction research**

## 🛠️ Tech Stack

* **Python 3**
* **Pandas / Fireducks** (data handling)
* **NumPy**
* **Requests** (API calls)
* **Matplotlib** (visualization)
* **Scikit-learn** (scaling + KNN graph)
* **NetworkX** (graph creation)
* **PyTorch + PyTorch Geometric** (GNNs)

## 📂 Workflow

1. **Install Dependencies**

   ```bash
   pip install pandas numpy requests matplotlib networkx torch torch-geometric scikit-learn fireducks
   ```

2. **Fetch Earthquake Data (USGS API)**

   ```python
   url = "https://earthquake.usgs.gov/fdsnws/event/1/query.geojson?...params..."
   response = requests.get(url)
   earthquake_data = response.json()
   ```

3. **Convert Data → DataFrame**

   * Extract longitude, latitude, depth, magnitude, and time
   * Convert timestamps to UNIX time

4. **Graph Construction**

   * Build KNN graph (`K=5`)
   * Nodes = earthquake events
   * Edges = nearest neighbor connections

5. **Graph Neural Network Input**

   * Normalize features with `StandardScaler` / `MinMaxScaler`
   * Create `torch_geometric.data.Data` object

   ```python
   from torch_geometric.data import Data
   data = Data(x=x, edge_index=edge_index, y=y)
   ```

## 📊 Example Output

* **Nodes (earthquakes):** ~11,883
* **Edges (spatial connections):** ~38,483
* Data object ready for GNN models:

  ```
  Data(x=[11883, 4], edge_index=[2, 38483], y=[11883])
  ```

## 🚀 Future Improvements

* Train GNN models to predict earthquake **magnitude** or **occurrence likelihood**
* Add **temporal modeling** for time-series predictions
* Visualize earthquake networks interactively

## 📜 Data Source

* [USGS Earthquake API](https://earthquake.usgs.gov/fdsnws/event/1/)
