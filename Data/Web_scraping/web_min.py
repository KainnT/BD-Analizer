import pandas as pd
import numpy as np
from selenium import webdriver
import time
import re 
from selenium.webdriver.common.by import By
from selenium import webdriver

from selenium.common.exceptions import NoSuchElementException, TimeoutException

browser = webdriver.Firefox()


browser.get("https://tiendadeaventuracr.com/collections/todos-nuestros-productos")



base_url = "https://tiendadeaventuracr.com"
prueba = '/collections/todos-nuestros-productos?page=1&grid_list'
lista = []
todos_productos = []

todos_productos = [base_url + re.sub(r"\d+", str(i), prueba) for i in range(1, 28)]

tablaProductos = pd.DataFrame(columns=["nombre", "precio actual", "precio descuento"])

for page_url in todos_productos:
    browser.get(page_url)
    
    time.sleep(5)
    
    productos = browser.find_elements(By.CSS_SELECTOR, "div.productitem--info")
    
    for producto in productos:
        try:
            etqNombre = producto.find_element(By.CSS_SELECTOR, "h2.productitem--title a").text.strip()
        except:
            etqNombre = "Nombre no disponible"
        
        try:
            etqPrecio = producto.find_element(By.CSS_SELECTOR, "div.price__current.price__current--emphasize span.money").text.strip()
        except:
            etqPrecio = "Precio no disponible"
        
        try:
            etqPrecio_descuento_element = producto.find_element(By.CSS_SELECTOR, "div.price__compare-at.visible span.money.price__compare-at--single")
            etqPrecio_descuento = etqPrecio_descuento_element.text.strip() if etqPrecio_descuento_element else "No hay descuento"
        except:
            etqPrecio_descuento = "No hay descuento"
        
        nuevoProducto = pd.DataFrame({
            "nombre": [etqNombre],
            "precio actual": [etqPrecio],
            "precio descuento": [etqPrecio_descuento]
        })
        
        tablaProductos = pd.concat([tablaProductos, nuevoProducto], ignore_index=True)
browser.quit()

tablaProductos.to_csv('productos.csv', index=False, encoding='utf-8')


