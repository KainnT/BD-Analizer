import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shiny as sh
from shiny import App, render, ui, reactive
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
import plotly.express as px
import seaborn as sns
import pandas as pd
from sqlalchemy import create_engine
import clases_app as cp
from shinyswatch import theme


app_ui = ui.page_navbar(
    ui.nav(
        "Load Data",
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_radio_buttons(
                    id="data_source",
                    label="Select:",
                    choices=["CSV", "SQL"],
                    selected="CSV",  # Default to CSV
                ),
                ui.panel_conditional(
                    "input.data_source == 'CSV'",
                    ui.input_file(
                        id="archivo",
                        label="Upload an CSV:",
                        accept=[".csv"],
                    ),
                    ui.input_radio_buttons(
                        id="sep",
                        label="Select the separator:",
                        choices=[";", ","],
                        inline=True,
                    ),
                    ui.input_radio_buttons(
                        id="dec",
                        label="Select the decimal symbol:",
                        choices=[",", "."],
                        inline=True,
                    ),
                    ui.input_switch(
                        id="rownames",
                        label="¿Row names?",
                        value=True,
                    ),
                ),
                ui.panel_conditional(
                    "input.data_source == 'SQL'",
                    ui.div(
                        {"class": "panel panel-default", "style": "padding: 10px;"},
                        ui.input_text(
                            id="sql_host",
                            label="Host:",
                            value="localhost",
                            placeholder="Enter the server hostname or IP address"
                        ),
                        ui.input_text(
                            id="sql_user",
                            label="Usuario:",
                            value="root",
                            placeholder="Enter your username"
                        ),
                        ui.input_password(
                            id="sql_password",
                            label="Contraseña:",
                            placeholder="Enter your password"
                        ),
                        ui.input_text(
                            id="sql_db",
                            label="Base de datos:",
                            value="btc_usd_2024",
                            placeholder="Enter the database name"
                        ),
                        ui.input_text(
                            id="sql_table",
                            label="Tabla:",
                            value="btc_usd_2024",
                            placeholder="Enter the table name"
                        ),
                    ),
                ),
                ui.input_action_button("cargar", "Load"),
                ui.input_radio_buttons(
                    id="pro",
                    label="Type of model:",
                    choices=["Time Series Forecast", "Classification", "Regression", "Deep Learning", "Clustering" ],
                    inline=True,
                )
            ),
            ui.card(ui.output_data_frame("tabladatos"), height="400px"),
            ui.panel_conditional(
                "input.pro != 'Clustering'",
                ui.card(
                    ui.output_data_frame("bech_deploy"),
                    style="width: 75%; height: 400px; margin-left: 0px;"
                ),
            ),
            ui.panel_conditional(
                "input.pro == 'Time Series Forecast'",
                ui.card(
                    ui.output_plot("graphTime"),
                    style="width: 75%; height: 400px; margin-left: 0px;"
                ),
            ),
            ui.panel_conditional(
                "input.pro == 'Clustering'",
                ui.card(
                    ui.output_plot("graphradar"),
                    style="width: 75%; height: 400px; margin-left: 0px;"
                ),
                ui.card(
                    ui.output_plot("graphdis"),
                    style="width: 75%; height: 400px; margin-left: 0px;"
                )
            ),
            ui.panel_conditional(
                "input.pro == 'Clustering'",
                ui.div(
                    ui.h3("Clustering preparation"),
                    ui.input_slider("nclusters", "Número de clusters:", min=2, max=10, value=2, step=1),
                    ui.input_select("clus_mod", "Clustering Model", choices=[None, "PCA", "Hierarchical", "Kmeans", "T-SNE", "UMAP"], width="70%"
                    ),
                    style="""
                        padding: 10px; 
                        margin: 0; 
                        margin-top: -795px;
                        margin-left: auto; 
                        background-color: #343a40; 
                        color: white;              
                        border-radius: 8px;
                        width: 20%;
                    """
                )
            ),
            ui.panel_conditional(
                "input.pro == 'Time Series Forecast'",
                ui.div(
                    ui.h3("Time Series preparation"),
                    ui.input_select("da_time", "Date column", choices=[], width="70%"),
                    ui.input_select("frequency", "Frequency", choices=[None, "B", "D", "W", "M", "Y"], width="70%"),
                    ui.input_numeric(
                        id="pred_time",
                        label="Forecast time",
                        value=1,
                        min=1,
                        max=31,
                        width="50%"
                    ),
                    ui.input_select("pred_col", "Column to Predict", choices=[], width="70%"),
                    style="""
                        padding: 10px; 
                        margin: 0; 
                        margin-top: -780px;
                        margin-left: auto; 
                        background-color: #343a40; 
                        color: white;              
                        border-radius: 8px;
                        width: 20%;   
                    """
                )
            ),
            ui.panel_conditional(
                "input.pro != 'Clustering' && input.pro != 'Time Series Forecast'",
                ui.div(
                    ui.h3("Class, Reg, Deep models preparation"),
                    ui.input_select("val_train", "Validation Method", choices=[None, "Cross Validation", "TrainTest 0.75"], width="70%"),                    
                    ui.panel_conditional(
                        "input.pro != 'Deep Learning' && input.val_train == 'Cross Validation'",
                        ui.div(
                            ui.input_numeric(
                                id="c_val_split",
                                label="K fold split",
                                value=1,
                                min=1,
                                max=5,
                                width="50%"
                            ),
                        )
                    ),
                    style="""
                        padding: 20px; 
                        margin: 0; 
                        margin-top: -360px;
                        margin-left: auto; 
                        background-color: #343a40; 
                        color: white;              
                        border-radius: 8px; 
                        width: 20%; 
                        height: 40%
                    """
                )
            ),
            ui.div(
            ui.input_select("del_col", "Delete any column?", choices=[], width="70%"),
            ui.input_action_button("update", "Generate", style="background-color:#B22222; color: white; padding: 10px"),
            style="""
                padding: 10px; 
                margin: 0; 
                margin-top: 20px;
                margin-left: auto; 
                background-color: #343a40; 
                color: white;              
                border-radius: 8px; 
                width: 20%; 
                height: 20%
            """
            )
        )   
    ),
    title="BD Analizer",
    bg="#D3D3D3",
    theme = theme.darkly(),
    
)

from shiny import App, reactive, render, session
def server(input, output, session):
    data = reactive.Value(None)
    data_time= reactive.Value(None)
    error_message = reactive.Value("")
    benchmark = reactive.Value(None)
    dis_plot = reactive.Value(None)
    rad_plot =  reactive.Value(None)
    time_plot = reactive.Value(None)
    @reactive.Effect
    @reactive.event(input.cargar)
    def cargar_datos():
        data_source = input.data_source()
        df = pd.DataFrame()

        if data_source == "CSV":
            file_info = input.archivo()
            if file_info is None:
                error_message.set("No file selected")
                return

            try:
                ruta = input.archivo()[0]["datapath"]
                sep = input.sep()
                dec = input.dec()
                rownames = input.rownames()
                if rownames:
                    rownames = 0

                df = pd.read_csv(ruta, sep=sep, decimal=dec, index_col=rownames)
                data.set(df)
                ui.update_select("varx", choices=df.columns.tolist(), session=session)
                ui.update_select("vary", choices=df.columns.tolist(), session=session)
            except Exception as e:
                ui.notification_show(str(e), duration=5)

        elif data_source == "SQL":
            try:
                engine = create_engine(
                    f"mysql+pymysql://{input.sql_user()}:{input.sql_password()}@{input.sql_host()}/{input.sql_db()}"
                )
                query = f"SELECT * FROM {input.sql_table()}"
                df = pd.read_sql(query, engine)
                data.set(df)
                ui.update_select("varx", choices=df.columns.tolist(), session=session)
                ui.update_select("vary", choices=df.columns.tolist(), session=session)
            except Exception as e:
                error_message.set(f"SQL Error: {e}")
                return

        error_message.set("")
        column_choices = ["None"] + list(df.columns.values)
        
        ui.update_select("da_time", choices=column_choices, selected=None, session=session)
        ui.update_select("pred_col", choices=column_choices, selected=None, session=session)        
        ui.update_select("del_col", choices=column_choices, selected=None, session=session)
    
    @reactive.Effect   
    @reactive.event(input.del_col)
    def delete():
        df = data.get()
        if df is None:
            error_message.set("Data frame is None. Please ensure data is loaded correctly.")
            return 
        del_col = input.del_col()
        if not del_col:
            error_message.set("No column specified for deletion.")
            return

        if del_col not in df.columns:
            error_message.set(f"Column not found: {del_col}")
            return

        try:
            df = df.drop(del_col, axis=1)
            print(df.head())
            print(f"Borrada {del_col}")
            data.set(df)
        except Exception as e:
            error_message.set(f"Error occurred: {e}")

    @reactive.Effect                      
    @reactive.event(input.update)
    def process_dates():
        problema = input.pro()
        pred_column = input.pred_col()
        df = data.get()

        if problema == 'Time Series Forecast':
            date_column = input.da_time()
            freq = input.frequency()
            fore_time = input.pred_time()
            if df is None or date_column not in df.columns:
                error_message.set("Date column not found in data")
                return
            
            try:
                df[date_column] = pd.to_datetime(df[date_column])
                df_series = df.copy()
                series_result = cp.prepa_series(df_series, date_column, pred_column, freq)  
                ts_model = cp.TimeSeriesModel(series_result, fore_time)
                ts_model.fit_exponential_smoothing()
                ts_model.fit_sarimax()

                graph = ts_model.plot_results() 
                time_plot.set(graph)
                # Benchmark errors
                benchmark_model = ts_model.benchmark()
                benchmark.set(benchmark_model)
                '''
                print("\nProcessed time series:")
                print(series_result.head())

                data_time(series_result)
                error_message.set("")'''
                
            except Exception as e:
                error_message.set(f"Error processing date column: {e}")
                return

        elif problema == 'Classification':
            try:
                metod= None
                val_splits = input.c_val_split()
                if val_splits is not None:
                    metod = input.val_train()
                print(metod)
                bench = cp.Benchmark_supervisados(df, metodo= metod, n_splits= val_splits)
                bench.fit()
                benchmark_model = bench.to_pandas()
                benchmark.set(benchmark_model)
                error_message.set("")  # Clear any previous error messages
            except Exception as e:
                error_message.set(f"Error in classification benchmark: {e}")
                return

        elif problema == 'Regression':
            try:
                metod= None
                val_splits = input.c_val_split()
                if val_splits is not None:
                    metod = input.val_train()
                print(metod)
                bench = cp.Benchmark_regresion(df, metodo= metod, n_splits= val_splits)
                benchmark_model = bench.to_pandas()  
                benchmark.set(benchmark_model)
                print(benchmark_model)
                error_message.set("")
            except Exception as e:
                error_message.set(f"Error in regression benchmark: {e}")
                return

        elif problema == 'Deep Learning':
            try:
                bench = cp.Benchmark_deep_models(df)
                bench.fit()
                benchmark_model = bench.to_pandas()
                benchmark.set(benchmark_model)
                error_message.set("")
            except Exception as e:
                error_message.set(f"Error in classification benchmark: {e}")
                return

    @reactive.Effect
    @reactive.event(input.update, input.clus_mod)
    def clustering_plots():
        problema = input.pro()
        clus_mod = input.clus_mod()
        df = data.get()
        clusters = input.nclusters()
        if problema == 'Clustering' and df is not None:
            ns = cp.NoSupervisados(df, n_clusters= clusters)
            if clus_mod == "PCA":
                dis = ns.plot_pca()
                dis_plot.set(dis)
            elif clus_mod == "Hierarchical":
                dis = ns.Jerarquico_graph_disper()
                rad = ns.Jerarquico_graph_radar()
                rad_plot.set(rad)
                dis_plot.set(dis)
            elif clus_mod == "Kmeans":
                dis = ns.Kmeans_graph_disper()
                rad = ns.Kmeans_graph_radar()
                rad_plot.set(rad)
                dis_plot.set(dis)
            elif clus_mod == "T-SNE":
                dis = ns.T_SNE_graph_disper()
                rad = ns.T_SNE_graph_radar()
                rad_plot.set(rad)
                dis_plot.set(dis)
            elif clus_mod == "UMAP":
                dis = ns.UMAP_graph_disper()
                rad = ns.UMAP_graph_radar()
                rad_plot.set(rad)
                dis_plot.set(dis)


    @output
    @render.text
    def error_message_output():
        return error_message.get()

    @output
    @render.data_frame
    def tabladatos():
        df = data.get()
        if df is None:
            return None
        return render.DataGrid(df, height="350px", width="100%", filters=True)

    @output
    @render.data_frame
    def bech_deploy():
        bench = benchmark.get()
        if bench is None:
            return None
        return render.DataGrid(bench, height="290px", width="100%", filters=False)

    @output
    @render.plot
    def graphdis():
        return dis_plot.get()  

    @output
    @render.plot
    def graphradar():
        return rad_plot.get()  
    
    @output
    @render.plot   
    def graphTime():
        return time_plot.get()  
    
app = App(app_ui, server)

    
    

