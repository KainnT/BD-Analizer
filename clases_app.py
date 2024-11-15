from math import pi
import math
import os
from random import randint
import statistics
import warnings
from webbrowser import get
import graphviz
import pandas as pd
import numpy  as np
from sklearn import tree
import plotly.express as px
import matplotlib.pyplot as plt
from pandas import DataFrame, get_dummies
from collections import OrderedDict
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, ward, single, complete,average,linkage, fcluster
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from numpy import corrcoef
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
#________________________________________________________________________________________

# Time series preparation

def prepa_series(df_series, date_column, pred_column, frequency):
    df_series[pred_column] = df_series[pred_column].astype(float)
    
    fechas = pd.DatetimeIndex(df_series[date_column].values)
    df_series[date_column] = fechas
    fecha_inicio = df_series[date_column].min()
    fecha_final = df_series[date_column].max()
    total_fechas = pd.date_range(start = fecha_inicio, end = fecha_final, freq=frequency).tolist()
    faltan_fechas = [x for x in total_fechas if x not in df_series[date_column].tolist()]
    df_series = pd.concat([df_series, pd.DataFrame({date_column: faltan_fechas})], ignore_index=True)
    df_suavizado = df_series.copy()
    df_suavizado 
    df_suavizado[pred_column]= df_suavizado[pred_column].rolling(5, min_periods=1, center=True).mean()
    df_series.loc[faltan_fechas, pred_column] = df_suavizado.loc[faltan_fechas, pred_column]
    fechas = pd.date_range(start=fecha_inicio, end=fecha_final, freq=frequency)
    series_result  = pd.Series(df_series[pred_column].values, index=fechas)

    return series_result

#________________________________________________________________________________________

# Time series Forecast models

class ts_error:
    def __init__(self, preds, real, nombres=None):
        self.__preds = preds
        self.__real = real
        self.__nombres = nombres

    @property
    def preds(self):
        return self.__preds

    @preds.setter
    def preds(self, preds):
        if isinstance(preds, (pd.Series, np.ndarray)):
            self.__preds = [preds]
        elif isinstance(preds, list):
            self.__preds = preds
        else:
            warnings.warn('ERROR: El parámetro preds debe ser una serie de tiempo o una lista de series de tiempo.')

    @property
    def real(self):
        return self.__real

    @real.setter
    def real(self, real):
        self.__real = real

    @property
    def nombres(self):
        return self.__nombres

    @nombres.setter
    def nombres(self, nombres):
        if isinstance(nombres, str):
            nombres = [nombres]
        if len(nombres) == len(self.__preds):
            self.__nombres = nombres
        else:
            warnings.warn('ERROR: Los nombres no calzan con la cantidad de métodos.')

    def RSS(self):
        res = []
        for pred in self.preds:
            res.append(sum((pred - self.real)**2))
        return res

    def MSE(self):
        return [pred / len(self.real) for pred in self.RSS()]

    def RMSE(self):
        return [math.sqrt(pred) for pred in self.MSE()]

    def RE(self):
        res = []
        for pred in self.preds:
            res.append(sum(abs(self.real - pred)) / sum(abs(self.real)))
        return res

    def CORR(self):
        res = []
        for pred in self.preds:
            corr = corrcoef(self.real, pred)[0, 1]
            res.append(0 if math.isnan(corr) else corr)
        return res

    def df_errores(self):
        res = pd.DataFrame({'MSE': self.MSE(), 'RMSE': self.RMSE(), 'RE': self.RE(), 'CORR': self.CORR()})
        if self.nombres is not None:
            res.index = self.nombres
        return res

class TimeSeriesModel:
    def __init__(self, data, test_size=15):
        self.data = data
        self.test_size = test_size
        self.train_data = self.data.head(len(self.data) - test_size)
        self.test_data = self.data.tail(test_size)
        self.pred_hw = None
        self.pred_arima = None
        self.__modelo = {}

    def fit_exponential_smoothing(self, trend='add', seasonal='add'):
        model = ExponentialSmoothing(self.train_data, trend=trend, seasonal=seasonal)
        model_fit = model.fit()
        self.pred_hw = model_fit.forecast(self.test_size)

    def fit_sarimax(self, order=(5, 1, 0), seasonal_order=(1, 0, 1, 7)):
        auto_arima(self.train_data, m=7)  # Optional: Can be used to determine the best parameters
        model = SARIMAX(self.train_data, order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        self.pred_arima = model_fit.forecast(self.test_size)

    def plot_results(self):
        series = [self.train_data, self.test_data, self.pred_hw, self.pred_arima]
        names = ["Training", "Test", "Holt-Winters", "SARIMAX"]

        fig = plt.figure(figsize=(12, 6))
        for serie, name in zip(series, names):
            if serie is not None:
                plt.plot(serie.index, serie.values, marker='o', label=name)

        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Time Series Forecasts')
        plt.legend()
        return fig
 

    def benchmark(self):
        errores = ts_error([self.pred_hw, self.pred_arima], self.test_data, ["Holt-Winters", "ARIMA"])
        self.__modelo = errores.df_errores()
        return self.__modelo

    def to_pandas(self):
        return self.__modelo


#________________________________________________________________________________________

# Unsupervised learning models 

class NoSupervisados:
    def __init__(self, df, n_componentes=2, n_clusters = 3):
        self.df = df
        self.n_componentes = n_componentes
        self.n_clusters = n_clusters
        self.datos_scaled = None
        self.grupos = None
        self.centros = None
    # Estandarizacion y hotcoded columns

    def estandarizacion(self):
        datos = self.df.select_dtypes(include=[np.number])
        datos = datos.fillna(datos.mean())
        scaler = StandardScaler()
        df_numericos_scaled = pd.DataFrame(scaler.fit_transform(datos), columns=datos.columns, index=self.df.index)
        return df_numericos_scaled
    
    def get_dummies(self):
        df_numericos_scaled = self.estandarizacion()
        categorical_columns = self.df.select_dtypes(exclude=[np.number])
        df_categorical = pd.get_dummies(categorical_columns)
        self.datos_scaled = pd.concat([df_numericos_scaled, df_categorical], axis=1)
        return self.datos_scaled

    #_________________________________________________________________________
    # Modelo ACP 

    def ACP(self):
        if self.datos_scaled is None:
            self.get_dummies()
        pca = PCA(n_components=self.n_componentes)
        componentes = pca.fit_transform(self.datos_scaled)
        return componentes
    
    # Grafico de dispersion 

    def plot_pca(self):
        componentes = self.ACP()
        fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        for i in range(componentes.shape[0]):
            x, y = componentes[i, 0], componentes[i, 1]
            ax.scatter(x, y)
        return fig
    #_________________________________________________________________________
    # Modelo Clustering Jerarquico

    def Jerarquico(self):
        if self.datos_scaled is None:
            self.get_dummies()
        self.grupos = fcluster(linkage(self.datos_scaled, method='ward'), self.n_clusters, criterion='maxclust') - 1
        self.centros = np.array([self.centroide(i) for i in range(self.n_clusters)])
        self.df['grupos'] = self.grupos

    def centroide(self, num_cluster):
        ind = self.grupos == num_cluster
        return pd.DataFrame(self.datos_scaled[ind].mean()).T


    # Llamar grafico radar  
    def Jerarquico_graph_radar(self):
        jerar = self.Jerarquico()
        if self.centros is not None:
            jerar = self.radar_plot(self.centros, self.datos_scaled.columns)
        return jerar
    # Llamar grafico dispersion
    def Jerarquico_graph_disper(self):
        jerar = self.Jerarquico()
        if self.grupos is not None:
            coms = self.ACP()
            jerar = self.plano_dis(coms)
        return jerar
    #_________________________________________________________________________
    # Modelo Kmeans

    def Kmeans(self):
        if self.datos_scaled is None:
                self.get_dummies()
        kmeans = KMeans(n_clusters = self.n_clusters, n_init = 200, max_iter = 2500)
        kmeans.fit(self.datos_scaled)
        print("Inercia Intra-Clase: ", kmeans.inertia_)
        self.grupos = kmeans.predict(self.datos_scaled)
        self.centros = kmeans.cluster_centers_

    def Kmeans_graph_radar(self):
        df_kmea = self.Kmeans()
        if self.centros is not None:
            df_kmea = self.radar_plot(self.centros, self.datos_scaled.columns)
        return df_kmea
    def Kmeans_graph_disper(self):
        df_kmea = self.Kmeans()
        if self.grupos is not None:
            coms = self.ACP()
            df_kmea = self.plano_dis(coms)
        return df_kmea
    #_________________________________________________________________________
    # Modelo T-SNE

    def T_SNE(self):
        if self.datos_scaled is None:
            self.get_dummies()
        tsne_est = TSNE(n_components=self.n_componentes, perplexity=30, learning_rate='auto', init='random').fit_transform(self.datos_scaled)
        df_tsne = pd.DataFrame(tsne_est, columns=["dim1", "dim2"], index=self.datos_scaled.index.values)
        
        kmedias = KMeans(n_clusters=self.n_clusters, max_iter=200, n_init=150)
        kmedias.fit(df_tsne)
        self.grupos = kmedias.predict(df_tsne)
        self.centros = kmedias.cluster_centers_  
        df_tsne["cluster"] = self.grupos
        return df_tsne
    
    def calcular_centros(self):
        if self.datos_scaled is not None and self.grupos is not None:
            return np.array([self.datos_scaled[self.grupos == i].mean(axis=0) for i in range(self.n_clusters)])
        return None
    
    def T_SNE_graph_radar(self):
        df_tsne = self.T_SNE() 
        centroids = self.calcular_centros()
        if centroids is not None:
            df_tsne = self.radar_plot(centroids, self.datos_scaled.columns)
        return df_tsne
    def T_SNE_graph_disper(self):
        df_tsne = self.T_SNE()
        df_tsne = self.plano_dis(df_tsne[['dim1', 'dim2']].values)
        return df_tsne
    #_________________________________________________________________________
    # Modelo UMAP 

    def UMAP(self):
        if self.datos_scaled is None:
            self.get_dummies()
        umap_est = UMAP(n_components=self.n_componentes, n_neighbors = 300).fit_transform(self.datos_scaled)
        df_umap = pd.DataFrame(umap_est, columns=["dim1", "dim2"], index=self.datos_scaled.index.values)
        
        kmedias = KMeans(n_clusters=self.n_clusters, max_iter=200, n_init=150)
        kmedias.fit(df_umap)
        self.grupos = kmedias.predict(df_umap)
        self.centros = kmedias.cluster_centers_  
        df_umap["cluster"] = self.grupos
        return df_umap
    
    def calcular_centros(self):
        if self.datos_scaled is not None and self.grupos is not None:
            return np.array([self.datos_scaled[self.grupos == i].mean(axis=0) for i in range(self.n_clusters)])
        return None
    
    def UMAP_graph_radar(self):
        df_tsne = self.UMAP() 
        centroids = self.calcular_centros()
        if centroids is not None:
            df_tsne = self.radar_plot(centroids, self.datos_scaled.columns)
        return df_tsne
    
    def UMAP_graph_disper(self):
        df_tsne = self.UMAP()
        df_tsne = self.plano_dis(df_tsne[['dim1', 'dim2']].values)
        return df_tsne
    #_________________________________________________________________________
    # Grafico Radar

    def radar_plot(self, centros, labels):
        fig, ax = plt.subplots(figsize=(15, 8), dpi=200, subplot_kw={'polar': True})
        normalized_centros = []
        for n in centros.T:
            if n.max() != n.min():
                normalized = (n - n.min()) / (n.max() - n.min()) * 100
            else:     
                normalized = np.full(n.shape, 50)  
            normalized_centros.append(normalized)       
        normalized_centros = np.array(normalized_centros)
        
        angles = [i / float(len(labels)) * 2 * pi for i in range(len(labels))]
        angles += angles[:1]        
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], labels, color='grey', size=8)
        plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
                ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], color="grey", size=7)
        plt.ylim(0, 100)
        
        for i in range(normalized_centros.shape[1]):
            values = normalized_centros[:, i].tolist()
            values += values[:1]  # ensure the plot is closed
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'Cluster {i + 1}')
            ax.fill(angles, values, alpha=0.3)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        return fig

    # Grafico Dispersion utlizando ACP y Clustering

    def plano_dis(self, componentes):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,6), dpi = 200)
        plt.scatter(componentes[:, 0], componentes[:, 1],
        c=self.grupos, edgecolor='none', alpha=1,
        cmap=plt.cm.get_cmap('tab10', 3))
        plt.xlabel('Componente 1')
        plt.ylabel('Componente 2')
        plt.colorbar()
        return fig


#________________________________________________________________________________________

# Supervised learning models 

class PrediccionBase:
    def __init__(self, datos):
        self.__datos = self.cleaning(datos)
        self.__precisionGlobal = 0
        self.__error_global = 0
        self.__verdaderosNegativos = 0
        self.__falsosPositivos = 0
        self.__falsosNegativos = 0
        self.__verdaderosPositivos = 0
        self.__reporte = 0
        self.__precision_category = 0

    @property
    def datos(self):
        return self.__datos

    @property
    def reporte(self):
        return self.__reporte

    def cleaning(self, datos):
        datos = datos.replace({'x': 0, 'o': 1, 'b': 2})
        return datos

    def entrenamiento(self):
        pass

    def generacionReporte(self, nombreDelModelo):
        dict = {
            "Modelo": [nombreDelModelo],
            "Precision Global": [self.__precisionGlobal],
            "Error Global": [self.__error_global],
            "Verdaderos Positivos": [self.__verdaderosPositivos],
            "Verdaderos Negativos": [self.__verdaderosNegativos],
            "Falsos Negativos": [self.__falsosNegativos],
            "Falsos Positivos": [self.__falsosPositivos]}
        self.__reporte = pd.DataFrame(dict).join(self.__precision_category)

    def analsis(self, MC, modelo):
        self.__verdaderosNegativos, self.__falsosPositivos, self.__falsosNegativos, self.__verdaderosPositivos = MC.ravel()
        self.__precisionGlobal = np.sum(MC.diagonal()) / np.sum(MC)
        self.__error_global = 1 - self.__precisionGlobal
        self.__precision_category = pd.DataFrame(MC.diagonal() / np.sum(MC, axis=1)).T
        self.__precision_category.columns = ["Precision Positiva (PP)", "Precision Negativa (PN)"]

        self.generacionReporte(modelo)
        return {"Matriz de Confusión": MC,
                "Precisión Global": self.__precisionGlobal,
                "Error Global": self.__error_global,
                "Precisión por categoría": self.__precision_category}

# __________________________________________________________________________________________

# Modelo de Kneighbors

class PrediccionKNeighbors(PrediccionBase):
    def __init__(self, datos, metodo, n_splits):
        super().__init__(datos)
        self.__instancia_potenciacion = None
        self.__x_train = []
        self.__metodo = metodo
        self.__n_splits = n_splits    
    
    @property
    def instancia(self):
        return self.__instancia_potenciacion

    @property
    def x_test(self):
        return self.__x_test


    def train(self, X, y, train_size=0.75):
        self.__x_train, self.__x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)
        self.__instancia_potenciacion.fit(self.__x_train, y_train)
        prediccion = self.__instancia_potenciacion.predict(self.__x_test)
        MC = confusion_matrix(y_test, prediccion)
        return MC

    def cros_val(self, X, y):
        best_MC = None
        best_accuracy = 0

        for i in range(5):
            kfold = KFold(n_splits=self.__n_splits, shuffle=True, random_state=0)
            MC = np.zeros((2, 2))
            for train_index, test_index in kfold.split(X):
                modelo = self.__instancia_potenciacion
                modelo.fit(X.iloc[train_index], y.iloc[train_index])
                pred_fold = modelo.predict(X.iloc[test_index])
                MC += confusion_matrix(y.iloc[test_index], pred_fold)
            accuracy = np.trace(MC) / np.sum(MC)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_MC = MC

        return best_MC
    
    def entrenamiento(self, nucleo="auto", n_neighbors=3, train_size=0.80, entrenamiento="KNeighbors"):
        x = self.datos.iloc[:, :-1]
        y = self.datos.iloc[:, -1]
 
        self.__instancia_potenciacion = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=nucleo)
        
        if self.__metodo == "Cross-Validation":
            MC = self.cros_val(X=x, y=y)
            print(MC)
        else:
            MC = self.train(X=x, y=y)

        indices = self.analsis(MC, entrenamiento)
        #for k in indices:
        #    print("\n%s:\n%s" % (k, str(indices[k])))

# __________________________________________________________________________________________

# Modelo XG boosting 

class PrediccionXGBoosting(PrediccionBase):
    def __init__(self, datos, metodo, n_splits):
        super().__init__(datos)
        self.__instancia_potenciacion = None
        self.__x_train = []
        self.__metodo = metodo
        self.__n_splits = n_splits    
    

    @property
    def instancia(self):
        return self.__instancia_potenciacion
    
    @property
    def x_test(self):
        return self.__x_test
    
    def train(self, X, y, train_size=0.75):
        self.__x_train, self.__x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)
        self.__instancia_potenciacion.fit(self.__x_train, y_train)
        prediccion = self.__instancia_potenciacion.predict(self.__x_test)
        MC = confusion_matrix(y_test, prediccion)
        return MC

    def cros_val(self, X, y):
        best_MC = None
        best_accuracy = 0

        for i in range(5):
            kfold = KFold(n_splits=self.__n_splits, shuffle=True, random_state=0)
            MC = np.zeros((2, 2))
            for train_index, test_index in kfold.split(X):
                modelo = self.__instancia_potenciacion
                modelo.fit(X.iloc[train_index], y.iloc[train_index])
                pred_fold = modelo.predict(X.iloc[test_index])
                MC += confusion_matrix(y.iloc[test_index], pred_fold)
            accuracy = np.trace(MC) / np.sum(MC)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_MC = MC

        return best_MC
    
    def obtenerVariablesImportantes(self):
        importancia = self.__instancia_potenciacion.feature_importances_
        print(importancia)
        etiquetas = self.__x_train.columns.values
        y_pos = np.arange(len(etiquetas))
        plt.figure(figsize=(10, 8))
        plt.barh(y_pos, importancia, align='center', alpha=0.5)
        plt.yticks(y_pos, etiquetas)

    def entrenamiento(self, train_size=0.75, n_estimators=300, random_state=0, min_samples_split=2,
                      entrenamiento="Bosques Aleatorios XG Boosting"):
        x = self.datos.iloc[:, :-1]
        y = self.datos.iloc[:, -1]

        self.__instancia_potenciacion = GradientBoostingClassifier(n_estimators=n_estimators, random_state=n_estimators,
                                                                   min_samples_split=min_samples_split, max_depth=None)

        if self.__metodo == "Cross Validation":
            MC = self.cros_val(X=x, y=y)
        else:
            MC = self.train(X=x, y=y)

        indices = self.analsis(MC, entrenamiento)
        #for k in indices:
        #    print("\n%s:\n%s" % (k, str(indices[k])))
# __________________________________________________________________________________________

# Modelo Random Forest 

class PrediccionRandomForest(PrediccionBase):
    def __init__(self, datos, metodo, n_splits):
        super().__init__(datos)
        self.__instancia_potenciacion = None
        self.__x_train = []
        self.__metodo = metodo
        self.__n_splits = n_splits    
    

    @property
    def instancia(self):
        return self.__instancia_potenciacion
    
    @property
    def x_test(self):
        return self.__x_test
    

    def obtenerVariablesImportantes(self):
        importancia = self.__instancia_potenciacion.feature_importances_
        print(importancia)
        etiquetas = self.__x_train.columns.values
        y_pos = np.arange(len(etiquetas))
        plt.figure(figsize=(10, 8))
        plt.barh(y_pos, importancia, align='center', alpha=0.5)
        plt.yticks(y_pos, etiquetas)

    def train(self, X, y, train_size=0.75):
        self.__x_train, self.__x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)
        self.__instancia_potenciacion.fit(self.__x_train, y_train)
        prediccion = self.__instancia_potenciacion.predict(self.__x_test)
        MC = confusion_matrix(y_test, prediccion)
        return MC

    def cros_val(self, X, y):
        best_MC = None
        best_accuracy = 0

        for i in range(5):
            kfold = KFold(n_splits=self.__n_splits, shuffle=True, random_state=0)
            MC = np.zeros((2, 2))
            for train_index, test_index in kfold.split(X):
                modelo = self.__instancia_potenciacion
                modelo.fit(X.iloc[train_index], y.iloc[train_index])
                pred_fold = modelo.predict(X.iloc[test_index])
                MC += confusion_matrix(y.iloc[test_index], pred_fold)
            accuracy = np.trace(MC) / np.sum(MC)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_MC = MC

        return best_MC
    
    def entrenamiento(self, train_size=0.75, n_estimators=10, entrenamiento="Bosques Aleatorios"):
        x = self.datos.iloc[:, :-1]
        y = self.datos.iloc[:, -1]

        self.__instancia_potenciacion = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

        if self.__metodo == "Cross Validation":
            MC = self.cros_val(X=x, y=y)
        else:
            MC = self.train(X=x, y=y)

        indices = self.analsis(MC, entrenamiento)
        #for k in indices:
        #    print("\n%s:\n%s" % (k, str(indices[k])))
# __________________________________________________________________________________________

# Modelo Ada Boosting

class PrediccionADABoosting(PrediccionBase):
    def __init__(self, datos, metodo, n_splits):
        super().__init__(datos)
        self.__instancia_potenciacion = None
        self.__x_train = []
        self.__metodo = metodo
        self.__n_splits = n_splits    
    
    @property
    def instancia(self):
        return self.__instancia_potenciacion

    @property
    def x_test(self):
        return self.__x_test

    def obtenerVariablesImportantes(self):
        importancia = self.__instancia_potenciacion.feature_importances_
        print(importancia)
        etiquetas = self.__x_train.columns.values
        y_pos = np.arange(len(etiquetas))
        plt.figure(figsize=(10, 8))
        plt.barh(y_pos, importancia, align='center', alpha=0.5)
        plt.yticks(y_pos, etiquetas)

    def train(self, X, y, train_size=0.75):
        self.__x_train, self.__x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)
        self.__instancia_potenciacion.fit(self.__x_train, y_train)
        prediccion = self.__instancia_potenciacion.predict(self.__x_test)
        MC = confusion_matrix(y_test, prediccion)
        return MC

    def cros_val(self, X, y):
        best_MC = None
        best_accuracy = 0

        for i in range(5):
            kfold = KFold(n_splits=self.__n_splits, shuffle=True, random_state=0)
            MC = np.zeros((2, 2))
            for train_index, test_index in kfold.split(X):
                modelo = self.__instancia_potenciacion
                modelo.fit(X.iloc[train_index], y.iloc[train_index])
                pred_fold = modelo.predict(X.iloc[test_index])
                MC += confusion_matrix(y.iloc[test_index], pred_fold)
            accuracy = np.trace(MC) / np.sum(MC)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_MC = MC

        return best_MC
    
    def entrenamiento(self, train_size=0.75, criterion="gini", splitter="best", min_samples_split=2,
                      entretenamiento="Bosques Aleatorios ADA Boosting"):
        x = self.datos.iloc[:, :-1]
        y = self.datos.iloc[:, -1]

        instancia_tree = DecisionTreeClassifier(min_samples_split= min_samples_split, max_depth=None,
                                                  criterion=criterion, splitter=splitter)
        
        self.__instancia_potenciacion = AdaBoostClassifier(instancia_tree,
                                                           n_estimators=100, random_state=0)
        
        if self.__metodo == "Cross Validation":
            MC = self.cros_val(X=x, y=y)
        else:
            MC = self.train(X=x, y=y)

        indices = self.analsis(MC, entretenamiento)
        #for k in indices:
        #    print("\n%s:\n%s" % (k, str(indices[k])))
# __________________________________________________________________________________________

# Modelo Arbol

class PrediccionArbolBinario(PrediccionBase):
    def __init__(self, datos, metodo, n_splits):
        super().__init__(datos)
        self.__instancia_potenciacion = None
        self.__x_train = []
        self.__metodo = metodo
        self.__n_splits = n_splits    
    
    
    @property
    def instancia(self):
        return self.__instancia_potenciacion
    
    @property
    def x_test(self):
        return self.__x_test
    
    def obtenerVariablesImportantes(self):
        importancia = self.__instancia_potenciacion.feature_importances_
        print(importancia)
        etiquetas = self.__x_train.columns.values
        y_pos = np.arange(len(etiquetas))
        plt.figure(figsize=(10, 8))
        plt.barh(y_pos, importancia, align='center', alpha=0.5)
        plt.yticks(y_pos, etiquetas)

    def train(self, X, y, train_size=0.75):
        self.__x_train, self.__x_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)
        self.__instancia_potenciacion.fit(self.__x_train, y_train)
        prediccion = self.__instancia_potenciacion.predict(self.__x_test)
        MC = confusion_matrix(y_test, prediccion)
        return MC

    def cros_val(self, X, y):
        best_MC = None
        best_accuracy = 0

        for i in range(5):
            kfold = KFold(n_splits=self.__n_splits, shuffle=True, random_state=0)
            MC = np.zeros((2, 2))
            for train_index, test_index in kfold.split(X):
                modelo = self.__instancia_potenciacion
                modelo.fit(X.iloc[train_index], y.iloc[train_index])
                pred_fold = modelo.predict(X.iloc[test_index])
                MC += confusion_matrix(y.iloc[test_index], pred_fold)
            accuracy = np.trace(MC) / np.sum(MC)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_MC = MC

        return best_MC

    def entrenamiento(self, entrenamiento="Decision Tree"):
        x = self.datos.iloc[:, :-1]
        y = self.datos.iloc[:, -1]
        self.__instancia_potenciacion = DecisionTreeClassifier()

        if self.__metodo == "Cross Validation":
            MC = self.cros_val(X=x, y=y)
        else:
            MC = self.train(X=x, y=y)

        indices = self.analsis(MC, entrenamiento)

        #for k in indices:
        #    print("\n%s:\n%s" % (k, str(indices[k])))

# __________________________________________________________________________________________

class Benchmark_supervisados:
    def __init__(self, datos, metodo = str(None), n_splits = None):
        self.__datos = datos
        self.__modelo = {}
        self.__datos_scaled = None
        self.__metodo = metodo
        self.__n_splits = n_splits
        self.get_dummies()  
    @property
    def modelo(self):
        return self.__modelo
    
    def get_dummies(self):
        categorical_columns = self.__datos.iloc[:, :-1].select_dtypes(include=['object']).columns
        self.__datos_scaled = pd.get_dummies(self.__datos.iloc[:, :-1], columns=categorical_columns)
        self.__datos_scaled = pd.concat([self.__datos_scaled, self.__datos.iloc[:, -1]], axis=1)
        return self.__datos_scaled

    def generarListaDeRandomIndex(self):
        listaElementosAleatorios = []
        for x in range(0, self.__datos_scaled.shape[0] - 1):
            listaElementosAleatorios.append(randint(0, self.__datos_scaled.shape[0] - 1))
        return listaElementosAleatorios

    def llenar(self):
        muestra = []
        seleccionRandom = self.generarListaDeRandomIndex()
        for x in seleccionRandom:
            muestra.append(self.__datos_scaled.iloc[x])
        return pd.DataFrame(muestra)

    def fit(self):
        prediccionADABosting = PrediccionADABoosting(datos=self.llenar(),  metodo= self.__metodo, n_splits = self.__n_splits)
        prediccionXGBoosting = PrediccionXGBoosting(datos=self.llenar(),  metodo= self.__metodo, n_splits = self.__n_splits)
        prediccionRandomForest = PrediccionRandomForest(datos=self.llenar(),  metodo= self.__metodo, n_splits = self.__n_splits)
        prediccionKNeighbors = PrediccionKNeighbors(datos=self.llenar(),  metodo= self.__metodo, n_splits = self.__n_splits)
        prediccionArbolBinario = PrediccionArbolBinario(datos=self.llenar(), metodo= self.__metodo, n_splits = self.__n_splits)

        prediccionADABosting.entrenamiento(entretenamiento="ADA Bosting")
        prediccionXGBoosting.entrenamiento(entrenamiento="XG Bosting")
        prediccionRandomForest.entrenamiento(entrenamiento="Random Forest")
        prediccionKNeighbors.entrenamiento(entrenamiento="KNeighbors")
        prediccionArbolBinario.entrenamiento(entrenamiento="Decision Tree")
 
        comparacion = pd.concat([
            prediccionADABosting.reporte,
            prediccionXGBoosting.reporte,
            prediccionRandomForest.reporte,
            prediccionKNeighbors.reporte,
            prediccionArbolBinario.reporte
        ])

        comparacion = comparacion.sort_values(by='Precision Global', ascending=False)
        comparacion.reset_index(inplace=True, drop=True)
        comparacion.columns = ['Modelo', 'Precision Global', 'Error Global', 'Verdaderos Positivos', 'Verdaderos Negativos', 'Falsos Positivos', 'Falsos Negativos', 'Precision Positiva (PP)', "Precision Negativa (PN)" ]

        self.__modelo = comparacion
        self.__modelo = comparacion
        
    def predict(self, x_test):
        print("El modelo es: ", self.__modelo.iloc[0].Modelo)
        return self.__modelo.iloc[0].instancia.predict(x_test)

    def to_pandas(self):
        return self.__modelo
    
'''    
    
datos_s = pd.read_csv(
    "C://Users//axels//Desktop//datasets//Clasificacion//potabilidad_V2.csv",delimiter=',',decimal=".",index_col=0)

df_s =  pd.read_csv("C://Users//axels//Desktop//datasets//Clasificacion//diabetes_V2.csv", delimiter = ',', decimal = ".", index_col=0)
    
    
benchmark = Benchmark_supervisados(datos_s, "Cross-Validation", n_splits= 2)
benchmark.fit()
print(benchmark.modelo)
'''


#________________________________________________________________________________________

# Regression Models

def ResVarPred(VarPred): 
  Cuartil = statistics.quantiles(VarPred)
  val_max = np.max(VarPred)
  val_min = np.min(VarPred)
  return {"Máximo": val_max,
                "Cuartiles": Cuartil,
                "Mínimo": val_min}





class Analisis_Predictivo:
    def __init__(self, datos: pd.DataFrame, predecir: str, predictoras=None, 
                 modelo=None, estandarizar=True, train_size=0.75, metodo=None, 
                 n_splits=None, random_state=None):
        self.datos = datos
        self.predecir = predecir
        self.predictoras = predictoras if predictoras else []
        self.nombre_clases = list(np.unique(self.datos[predecir].values))
        self.modelo = modelo
        self.random_state = random_state
        self.__metodo = metodo
        self.__n_splits = n_splits
        self.train_size = train_size

        if self.modelo is not None:
            self.entrenamiento(estandarizar)

    def entrenamiento(self, estandarizar=True):
        if not self.predictoras:
            X = self.datos.drop(columns=[self.predecir])
            self.predictoras = list(X.columns.values)
        else:
            X = self.datos[self.predictoras]

        if estandarizar:
            X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        
        y = self.datos[self.predecir].values
        
        if self.__metodo == "Cross Validation":
            y_predicted, y_test = self.cross_val(X=X, y=y)
        else:
            y_predicted = self.train_test(X=X, y=y)
            y_test = self.y_test
        return y_predicted, y_test

    def train_test(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=self.train_size, random_state=self.random_state
        )
        if(self.modelo != None):
            self.modelo.fit(self.X_train, self.y_train)
            return self.modelo.predict(self.X_test)

    def cross_val(self, X, y):
        best_rmse = []
        best_model = None

        for _ in range(5):
            kfold = KFold(n_splits=self.__n_splits, shuffle=True, random_state=self.random_state)
            rmse_total = 0
            
            for train, test in kfold.split(X, y):
                tam_test = y[test].size
                modelo = self.modelo
                modelo.fit(X.iloc[train], y[train])
                pred_fold = modelo.predict(X.iloc[test])
                rmse_total= rmse_total + np.sqrt((1/tam_test)*np.sum(np.square(y[test] - pred_fold)))
            rmse_total = rmse_total / self.__n_splits
            
            best_rmse.append(rmse_total)
            
        rmse_total = np.round(np.sum(best_rmse)/5,decimals=2)

        print(f"Best RMSE: {best_rmse}")
        return pred_fold, y[test]

    def fit_predict_resultados(self, imprimir=True):
        if self.modelo is not None:
            indices = self.indices_general(self.nombre_clases)
            
            if imprimir:
                for k, v in indices.items():
                    print(f"\n{k}:\n{v}")
            
            return indices

    def indices_general(self, nombres=None):
        if isinstance(self.datos[self.predecir].iloc[0], (float, int)):

            y_predicted, y_test = self.entrenamiento()

            MSE0 = np.sum(np.square(y_test - y_predicted))
            RMSE = np.sqrt(MSE0 / len(y_test))
            MAE0 = np.sum(np.abs(y_test - y_predicted))
            MAE = MAE0 / len(y_test)
            EN = np.sum(np.abs(y_test - y_predicted))
            ED = np.sum(np.abs(y_test))
            ER = EN / ED
            return {"RMSE": RMSE,
                    "MAE": MAE,
                    "ER": ER}
        else:
            prediccion = self.fit_predict()
            MC = confusion_matrix(self.y_test, prediccion, labels= self.nombre_clases)
            precision_global = np.sum(MC.diagonal()) / np.sum(MC)
            error_global = 1 - precision_global
            precision_categoria = pd.DataFrame(MC.diagonal()/np.sum(MC, axis=1)).T
            if nombres != None:
                precision_categoria.columns = nombres
            return {"Matriz de Confusión": MC,
                    "Precisión Global": precision_global,
                    "Error Global": error_global,
                    "Precisión por categoría": precision_categoria}


class Benchmark_regresion:
    def __init__(self, datos, metodo = str(None), n_splits = None):
        self.__datos = datos
        self.__modelo = {}
        self.__datos_scaled = self.preprocess_data() 
        self.__metodo = metodo
        self.__n_splits = n_splits
    @property
    def modelo(self):
        return self.__modelo
    
    def preprocess_data(self):
        datos_with_dummies = self.catcodes()
        datos_preprocessed = self.nans(datos_with_dummies)
        return datos_preprocessed
    
    def catcodes(self):
        categorical_columns = self.__datos.select_dtypes(exclude=[np.number]).astype('category')
        coded_df = pd.DataFrame()
        for col in categorical_columns.columns:
            coded_df[col + '_coded'] = categorical_columns[col].cat.codes
        datos_scaled = pd.concat([self.__datos, coded_df], axis=1)
        datos_scaled.drop(categorical_columns.columns, axis=1, inplace=True)
        return datos_scaled
    
    def nans(self, datos_scaled):
        mis_val_count  = datos_scaled.isna().sum().sum() 
        if mis_val_count > 0:
            imputer = IterativeImputer(max_iter=15)
            datos_imputed = imputer.fit_transform(datos_scaled)
            self.__datos_scaled = pd.DataFrame(datos_imputed, columns=datos_scaled.columns)  
        else:
            self.__datos_scaled = datos_scaled

        return self.__datos_scaled
    
    def linear_multi(self):
        regresion_lineal = LinearRegression() 
        analisis_Boston = Analisis_Predictivo(self.__datos_scaled, predecir = "precio",  modelo   = regresion_lineal, train_size = 0.75, metodo= self.__metodo, n_splits= self.__n_splits)
        resultados_rl = analisis_Boston.fit_predict_resultados()
        return resultados_rl

    def lasso(self):
        modelo_lasso = linear_model.Lasso(alpha=0.1)
        analisis_Boston = Analisis_Predictivo(self.__datos_scaled, predecir = "precio",  modelo   = modelo_lasso, train_size = 0.75, metodo= self.__metodo, n_splits= self.__n_splits )
        resultados_rll = analisis_Boston.fit_predict_resultados()
        return resultados_rll
    
    def lassoCV(self):
        modelo_lasso = linear_model.LassoCV()
        analisis_Boston = Analisis_Predictivo(self.__datos_scaled, predecir = "precio",  modelo   = modelo_lasso, train_size = 0.75, metodo= self.__metodo, n_splits=  self.__n_splits)
        resultados_rllCv = analisis_Boston.fit_predict_resultados()
        return resultados_rllCv
    
    def ridge(self):
        modelo_Ridge = linear_model.Ridge(alpha = 1.0)
        analisis_Boston = Analisis_Predictivo(self.__datos_scaled, predecir = "precio",  modelo   = modelo_Ridge, train_size = 0.75, metodo= self.__metodo, n_splits= self.__n_splits)
        resultados_rlr = analisis_Boston.fit_predict_resultados()
        return resultados_rlr
    
    def RidgeCV(self):
        modelo_Ridge = linear_model.RidgeCV()
        analisis_Boston = Analisis_Predictivo(self.__datos_scaled, predecir = "precio",  modelo   = modelo_Ridge, train_size = 0.75, metodo= self.__metodo, n_splits= self.__n_splits)
        resultados_rlrCv = analisis_Boston.fit_predict_resultados()
        return resultados_rlrCv
    
    def svr(self, kernel):
        modelo_msv = make_pipeline(StandardScaler(),SVR(kernel= kernel, C=100, epsilon=0.1)) 
        analisis_Boston = Analisis_Predictivo(self.__datos_scaled, predecir = "precio",  modelo   = modelo_msv, train_size = 0.75, metodo= self.__metodo, n_splits= self.__n_splits)
        resultados_msv = analisis_Boston.fit_predict_resultados()
        return resultados_msv
    
    def decisionTreeReg(self):
        modelo_arbol = DecisionTreeRegressor(
            max_depth = 3, 
            random_state = 123 
          )
        analisis_Boston = Analisis_Predictivo(self.__datos_scaled, predecir = "precio",  modelo   = modelo_arbol, train_size = 0.75, metodo= self.__metodo, n_splits= self.__n_splits)
        resultados_arbol = analisis_Boston.fit_predict_resultados()
        return resultados_arbol
    
    def randomForestReg(self):
        modelo_bosque = RandomForestRegressor(max_depth=2, random_state=0)
        analisis_Boston = Analisis_Predictivo(self.__datos_scaled, predecir = "precio",  modelo   = modelo_bosque, train_size = 0.75, metodo = self.__metodo, n_splits= self.__n_splits)
        resultados_bosque = analisis_Boston.fit_predict_resultados()
        return resultados_bosque
    
    def potenciador(self):
        params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}
        modelo_potenciacion = ensemble.GradientBoostingRegressor(**params)
        analisis_Boston = Analisis_Predictivo(self.__datos_scaled, predecir = "precio",  modelo   = modelo_potenciacion, train_size = 0.75,  metodo = self.__metodo, n_splits= self.__n_splits)
        resultados_potenciacion = analisis_Boston.fit_predict_resultados()
        return resultados_potenciacion
    
    def beckmark(self):
        resultados_rl = self.linear_multi()
        resultados_rll = self.lasso()
        resultados_rllCv = self.lassoCV()
        resultados_rlr = self.ridge()
        resultados_rlrCv = self.RidgeCV()
        resultados_msv_linear = self.svr("linear")
        resultados_msv_poly = self.svr("poly")  
        resultados_msv_rbf = self.svr("rbf")    
        resultados_arbol = self.decisionTreeReg()
        resultados_bosque = self.randomForestReg()
        resultados_potenciacion = self.potenciador()

        comparacion = pd.DataFrame({
            'Múltiple': resultados_rl,
            'Lasso': resultados_rll,
            'Lasso CV': resultados_rllCv,
            'Ridge': resultados_rlr,
            'Ridge CV': resultados_rlrCv,
            'SVM Linear': resultados_msv_linear,  
            'SVM Poly': resultados_msv_poly,      
            'SVM RBF': resultados_msv_rbf,        
            'Árbol': resultados_arbol,
            'Bosques': resultados_bosque,
            'Potenciación': resultados_potenciacion
        }).T  


        resumen_numerico = ResVarPred(self.__datos.precio)
        print(f"Resumen numerico del Dataset: {resumen_numerico}, Importante para medir el rendimiento de los modelos")


        comparacion = comparacion.sort_values(by='RMSE', ascending=True)
        comparacion.reset_index(inplace=True)
        comparacion.rename(columns={'index': 'Modelo'}, inplace=True)
        comparacion.columns = ['Modelo', 'RMSE', 'MAE', 'ER']

        return comparacion
    def to_pandas(self):
        return self.beckmark()

# __________________________________________________________________________________________

# Deep Learning Models

class PrediccionBaseN:
    def __init__(self, datos):
        self.__datos = self.cleaning(datos)
        self.__precisionGlobal = 0.0
        self.__error_global = 0.0
        self.__verdaderosNegativos = 0
        self.__falsosPositivos = 0
        self.__falsosNegativos = 0
        self.__verdaderosPositivos = 0
        self.__reporte = pd.DataFrame()
        self.__precision_category = pd.DataFrame()

    @property
    def datos(self):
        return self.__datos

    @property
    def reporte(self):
        return self.__reporte

    def cleaning(self, datos):
        datos = datos.replace({'x': 0, 'o': 1, 'b': 2})
        return datos

    def entrenamiento(self):
        pass

    def generacionReporte(self, nombreDelModelo: str):

        report_dict = {
            "Modelo": [nombreDelModelo],
            "Precision Global": [self.__precisionGlobal],
            "Error Global": [self.__error_global],
            "Verdaderos Positivos": [self.__verdaderosPositivos],
            "Verdaderos Negativos": [self.__verdaderosNegativos],
            "Falsos Negativos": [self.__falsosNegativos],
            "Falsos Positivos": [self.__falsosPositivos]
        }
        if not self.__precision_category.empty:
            self.__reporte = pd.DataFrame(report_dict).join(self.__precision_category)
        else:
            self.__reporte = pd.DataFrame(report_dict)

    def analsis(self, MC, nombres=None, modelo=""):
        "Método para calcular los índices de calidad de la predicción"
        self.__precisionGlobal = np.sum(MC.diagonal()) / np.sum(MC)
        self.__error_global = 1 - self.__precisionGlobal
        self.__precision_category = pd.DataFrame(MC.diagonal() / np.sum(MC, axis=1)).T
        if nombres is not None:
            self.__precision_category.columns = nombres
            print(nombres)
            self.generacionReporte(modelo)
        return {
            "Matriz de Confusión": MC,
            "Precision Global": self.__precisionGlobal,
            "Error Global": self.__error_global,
            "Precisión por categoría": self.__precision_category}
        
class Deep_Models1(PrediccionBaseN):
    def __init__(self, datos):
        super().__init__(datos)
        self.__instancia_potenciacion = None
        self.__x_train = []
        self.__x_test = []
        self.__y_train = []
        self.__y_test = []

    @property
    def instancia(self):
        return self.__instancia_potenciacion

    @property
    def x_test(self):
        return self.__x_test

    def entrenamiento(self, train_size=0.75, entrenamiento="Models Class"):
        x = self.datos.iloc[:, :-1]
        y = self.datos.iloc[:, -1]
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x, y, train_size=train_size, random_state=0)
        
        self.__instancia_potenciacion = self.build_model_1()
        
        self.__instancia_potenciacion.fit(self.__x_train, self.__y_train, epochs=100, batch_size=16, verbose=0)
        prediccion = self.__instancia_potenciacion.predict(self.__x_test)
        prediccion = [1 if p > 0.5 else 0 for p in prediccion]
        prediccion = encoder.inverse_transform(prediccion)

        # Recodificamos las etiquetas de y_test
        y_test_classes = encoder.inverse_transform(self.__y_test)
        
        MC = confusion_matrix(prediccion, y_test_classes, labels = encoder.classes_)
        indices = self.analsis(MC, list(encoder.classes_), entrenamiento)

    def build_model_1(self):
        model = Sequential()
        model.add(Dense(units=60, activation="tanh"))
        model.add(Dense(units=30, activation="tanh"))
        model.add(Dense(units=15, activation="relu"))
        model.add(Dense(units=1, activation="sigmoid"))

        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        return model


class Deep_Models2(PrediccionBaseN):
    def __init__(self, datos):
        super().__init__(datos)
        self.__instancia_potenciacion = None
        self.__x_train = []
        self.__x_test = []
        self.__y_train = []
        self.__y_test = []

    @property
    def instancia(self):
        return self.__instancia_potenciacion

    @property
    def x_test(self):
        return self.__x_test

    def entrenamiento(self, train_size=0.75, entrenamiento="Models Class"):
        x = self.datos.iloc[:, :-1]
        y = self.datos.iloc[:, -1]
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x, y, train_size=train_size, random_state=0)
        
        self.__instancia_potenciacion = self.build_model_2()
        
        self.__instancia_potenciacion.fit(self.__x_train, self.__y_train, epochs=100, batch_size=16, verbose=0)
        prediccion = self.__instancia_potenciacion.predict(self.__x_test)
        prediccion = [1 if p > 0.5 else 0 for p in prediccion]
        prediccion = encoder.inverse_transform(prediccion)

        # Recodificamos las etiquetas de y_test
        y_test_classes = encoder.inverse_transform(self.__y_test)
        
        MC = confusion_matrix(prediccion, y_test_classes, labels = encoder.classes_)
        indices = self.analsis(MC, list(encoder.classes_), entrenamiento)
        
        
    def build_model_2(self):
        model = Sequential()
        model.add(Dense(units=20, activation="tanh"))
        model.add(Dense(units=15, activation="relu"))
        model.add(Dense(units=10, activation="relu"))
        model.add(Dense(units=1, activation="sigmoid"))

        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    
    
class Deep_Models3(PrediccionBaseN):
    def __init__(self, datos):
        super().__init__(datos)
        self.__instancia_potenciacion = None
        self.__x_train = []
        self.__x_test = []
        self.__y_train = []
        self.__y_test = []

    @property
    def instancia(self):
        return self.__instancia_potenciacion

    @property
    def x_test(self):
        return self.__x_test

    def entrenamiento(self, train_size=0.75, entrenamiento="Models Class"):
        x = self.datos.iloc[:, :-1]
        y = self.datos.iloc[:, -1]
        encoder = LabelEncoder()
        y = encoder.fit_transform(y)
        
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(x, y, train_size=train_size, random_state=0)
        
        self.__instancia_potenciacion = self.build_model_3()
        
        self.__instancia_potenciacion.fit(self.__x_train, self.__y_train, epochs=100, batch_size=16, verbose=0)
        prediccion = self.__instancia_potenciacion.predict(self.__x_test)
        prediccion = [1 if p > 0.5 else 0 for p in prediccion]
        prediccion = encoder.inverse_transform(prediccion)

        # Recodificamos las etiquetas de y_test
        y_test_classes = encoder.inverse_transform(self.__y_test)
        
        MC = confusion_matrix(prediccion, y_test_classes, labels = encoder.classes_)
        indices = self.analsis(MC, list(encoder.classes_), entrenamiento)


    def build_model_3(self):
        model = Sequential()
        model.add(Dense(units=36, activation="relu"))
        model.add(Dense(units=16, activation="relu"))
        model.add(Dense(units=12, activation="relu"))
        model.add(Dense(units=1, activation="sigmoid"))

        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    
    
class Benchmark_deep_models:
    def __init__(self, datos):
        self.__datos = datos
        self.__modelo = {}
        self.__datos_scaled = None
        self.get_dummies()

    @property
    def modelo(self):
        return self.__modelo

    def get_dummies(self):
        categorical_columns = self.__datos.iloc[:, :-1].select_dtypes(include=['object']).columns
        self.__datos_scaled = pd.get_dummies(self.__datos.iloc[:, :-1], columns=categorical_columns)
        self.__datos_scaled = pd.concat([self.__datos_scaled, self.__datos.iloc[:, -1]], axis=1)
        self.__datos_scaled = self.__datos_scaled.astype({col: 'float64' for col in self.__datos_scaled.select_dtypes(include=['int64']).columns})
        return self.__datos_scaled

    def generarListaDeRandomIndex(self):
        listaElementosAleatorios = []
        for x in range(0, self.__datos_scaled.shape[0] - 1):
            listaElementosAleatorios.append(randint(0, self.__datos_scaled.shape[0] - 1))
        return listaElementosAleatorios

    def llenar(self):
        muestra = []
        seleccionRandom = self.generarListaDeRandomIndex()
        for x in seleccionRandom:
            muestra.append(self.__datos_scaled.iloc[x])
        return pd.DataFrame(muestra)

    def fit(self):
        prediccionDeep1 = Deep_Models1(datos=self.llenar())
        prediccionDeep2 = Deep_Models2(datos=self.llenar())
        prediccionDeep3 = Deep_Models3(datos=self.llenar())

        prediccionDeep1.entrenamiento(entrenamiento="Modelo 1")
        prediccionDeep2.entrenamiento(entrenamiento="Modelo 2")
        prediccionDeep3.entrenamiento(entrenamiento="Modelo 3")

        reports = [
            prediccionDeep1.reporte if isinstance(prediccionDeep1.reporte, pd.DataFrame) else pd.DataFrame(),
            prediccionDeep2.reporte if isinstance(prediccionDeep2.reporte, pd.DataFrame) else pd.DataFrame(),
            prediccionDeep3.reporte if isinstance(prediccionDeep3.reporte, pd.DataFrame) else pd.DataFrame()
        ]


        comparacion = pd.concat(reports)

        comparacion = comparacion.sort_values(by='Precision Global', ascending=False)
        comparacion.reset_index(inplace=True, drop=True)
        comparacion.columns = ['Modelo', 'Precision Global', 'Error Global', 'Verdaderos Positivos', 'Verdaderos Negativos', 'Falsos Positivos', 'Falsos Negativos', 'Precision Positiva (PP)', "Precision Negativa (PN)" ]

        comparacion = comparacion.loc[:, (comparacion != 0).any(axis=0) & comparacion.notna().all(axis=0)]

        self.__modelo = comparacion
    def predict(self, x_test):
        print("El modelo es: ", self.__modelo.iloc[0].Modelo)
        return self.__modelo.iloc[0].instancia.predict(x_test)

    def to_pandas(self):
        return self.__modelo



