# BD Analizer

**BD Analizer** is a Python-based Shiny web application for data analysis and visualization. This app allows users to load data from CSV files or SQL databases and perform various analyses, including clustering, time series forecasting, classification, regression, and deep learning. It also provides dynamic visualizations and benchmarking.

---

## **Features**
- Load data from:
  - **CSV**: Upload CSV files with customizable parsing options (separator, decimal symbol, row names).
  - **SQL**: Connect to a database and query data from specified tables.
- Perform advanced data analysis, including:
  - **Clustering**: PCA, Hierarchical, K-means, T-SNE, and UMAP methods.
  - **Time Series Forecasting**: SARIMAX and Exponential Smoothing.
  - **Supervised Learning**: Classification, Regression, and Deep Learning.
- Dynamic and interactive visualizations:
  - Scatter plots, radar charts, and time series graphs.
- Error handling with detailed feedback for incorrect inputs or processing errors.

---

## **Technologies Used**
```plaintext
- Python (Shiny for Python)
- Pandas, Numpy
- Scipy, Sklearn
- Matplotlib, Plotly, Seaborn
- MySQL Connector, SQLAlchemy
```

---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/KainnT/BD-Analizer.git
   cd BD-Analizer
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   shiny run app.py
   ```

4. Open your browser and navigate to:
   ```plaintext
   http://127.0.0.1:8000
   ```

---

## **File Structure**
```plaintext
BD-Analizer/
├── app.py               # Main application file
├── clases_app.py        # Custom classes for processing and models
├── requirements.txt     # Python dependencies
├── static/              # Static files (e.g., CSS, JS, images)
├── templates/           # HTML templates (if needed)
└── README.md            # Documentation
```

---

## **How to Use**

### **1. Load Data**
You can load data in two ways:
- **CSV**
  - Upload a `.csv` file.
  - Specify separator (`;` or `,`), decimal symbol (`.` or `,`), and row names.
- **SQL**
  - Provide database credentials:
    ```plaintext
    - Host: Server address (e.g., localhost)
    - User: Database username
    - Password: Database password
    - Database: Name of the database
    - Table: Table to query
    ```

### **2. Select a Model**
Choose the type of analysis from the **Type of model** options:
- **Time Series Forecast**
  - Select the date column, frequency (daily, monthly, etc.), and prediction horizon.
- **Clustering**
  - Choose the number of clusters and clustering model (PCA, Hierarchical, etc.).
- **Classification / Regression / Deep Learning**
  - Configure validation method and model parameters.

### **3. View Results**
- Explore the processed dataset in an interactive data table.
- View plots and benchmarking results dynamically based on your selections.

---

## **Key Code Snippets**

### **Data Loading**
```python
@reactive.Effect
@reactive.event(input.cargar)
def cargar_datos():
    data_source = input.data_source()
    if data_source == "CSV":
        ruta = input.archivo()[0]["datapath"]
        df = pd.read_csv(ruta, sep=input.sep(), decimal=input.dec(), index_col=input.rownames())
        data.set(df)
    elif data_source == "SQL":
        engine = create_engine(
            f"mysql+pymysql://{input.sql_user()}:{input.sql_password()}@{input.sql_host()}/{input.sql_db()}"
        )
        df = pd.read_sql(f"SELECT * FROM {input.sql_table()}", engine)
        data.set(df)
```

### **Time Series Forecasting**
```python
@reactive.Effect
@reactive.event(input.update)
def process_dates():
    if input.pro() == 'Time Series Forecast':
        date_column = input.da_time()
        df = data.get()
        df[date_column] = pd.to_datetime(df[date_column])
        ts_model = cp.TimeSeriesModel(df, input.pred_time())
        ts_model.fit_exponential_smoothing()
        time_plot.set(ts_model.plot_results())
```

### **Clustering**
```python
@reactive.Effect
@reactive.event(input.update, input.clus_mod)
def clustering_plots():
    if input.pro() == 'Clustering':
        ns = cp.NoSupervisados(data.get(), n_clusters=input.nclusters())
        if input.clus_mod() == "PCA":
            dis_plot.set(ns.plot_pca())
        elif input.clus_mod() == "Kmeans":
            rad_plot.set(ns.Kmeans_graph_radar())
            dis_plot.set(ns.Kmeans_graph_disper())
```

---

## **Future Enhancements**
- Add more clustering algorithms such as DBSCAN.
- Integrate advanced time series models like Prophet.
- Improve UI responsiveness for mobile devices.

---

## **License**
This project is licensed under the [MIT License](LICENSE).

---

Let me know if you'd like to adjust any sections or add specific examples!
