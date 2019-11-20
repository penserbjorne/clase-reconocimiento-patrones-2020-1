#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Universidad Nacional Autónoma de México
# Facultad de Ingeniería
# Reconocimiento de patrones
# 2020-1
# Aguilar Enriquez Paul Sebastian
# Ejemplo de t-SNE
# Basado en https://towardsdatascience.com/t-sne-python-example-1ded9953f26


# In[2]:


import numpy as np
from sklearn.datasets import load_digits    # Dataset

# Para implementar t-SNE a mano
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform

# t-SNE implementado en sklearn
from sklearn.manifold import TSNE

# Para graficación
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)


# In[3]:


# Cargamos el dataset
X, y = load_digits(return_X_y=True)


# In[4]:


# Parametros para t-SNE

## Epsilon / Learning Rate
MACHINE_EPSILON = np.finfo(np.double).eps
## Dimnesiones que tendremos
n_components = 2
## Perplejidad por default en la versión de sklearn
perplexity = 30


# **Espacio de alta dimensionalidad**
# 
# t-SNE primero calcula las probabilidad p_ij que son proporionales a la similitud de los objetos x_i y x_j mediante:
# 
# ![./imgs/01.png](./imgs/01.png)
# 
# La similitud de un punto x_j a un punto x_i es la probabilidad condicional p(j|i), esto es, x_i tomará x_j como su vecino.
# 
# ![./imgs/02.png](./imgs/02.png)
# 
# **Grados de libertad**
# 
# Una `Student t-distribution` con un grado de libertad es lo mismo que una `Cauchy distribution` utilizada para medir similitud entre punto de baja dimensionalidad.
# 
# ![./imgs/04.png](./imgs/04.png)

# In[5]:


# Transforma nuestro set de datos, se apoya en la función _tsne
# que esta definida más abajo

def fit(X):
    # Almacenamos el número de muestras para futura referencia
    n_samples = X.shape[0]
    
    # Distancia euclideana
    distances = pairwise_distances(X, metric='euclidean', squared=True)
    
    # Probabilidades conjuntas p_ij de las distancias
    P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)
    
    # Los embeddings son inicializados con iid muetras de Gaussianos con desviación estander 1e-4.
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)
    
    # degrees_of_freedom = n_components - 1 viene de 
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)
    
    return _tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded)


# In[6]:


# Función _tsne implementada a mano
def _tsne(P, degrees_of_freedom, n_samples, X_embedded):
    
    # Convertirmos nuestro vector en uno de 1-D
    params = X_embedded.ravel()
    
    # Generamos un objeto que procese la divergencia KL
    obj_func = _kl_divergence
    
    # Generamos un objeto que procese el gradiente descendiente
    # con los parametros definifos para el t-SNE
    # Vasicamente utilizamos nuestro gradiente para minimizar
    # la divergencia KL
    params = _gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, n_components])
    
    # Regresamos la transformación ya procesada pero ahora en 2D
    X_embedded = params.reshape(n_samples, n_components)
    
    return X_embedded


# **Espacio de baja dimensionalidad**
# 
# Calculamos la distribución de probabilidad de los puntos del espacio de baja dimensionalidad.
# 
# ![./imgs/03.png](./imgs/03.png)
# 
# En el paper se modifica el calculo añadiendo los grados de libertad (α) para la `Student-t distribution`
# 
# ![./imgs/05.png](./imgs/05.png)
# 
# **Divergencia KL**
# 
# Calculamos la divergencia KL (veamos a `np.dot` como una suma.
# 
# ![./imgs/06.png](./imgs/06.png)
# 
# **Gradiente (derivadas parciales)**
# 
# `dist` es yi — yj en:
# 
# ![./imgs/07.png](./imgs/07.png)
# 
# Una vez más, en el paper se modifica la ecuación diferencial para incluir los grados de libertad.
# 
# ![./imgs/08.png](./imgs/08.png)

# In[7]:


# Implementación de la divergencia KL
def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components):
    X_embedded = params.reshape(n_samples, n_components)
    
    # Calculamos Q
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    
    # Divergencia Kullback-Leibler de P y Q
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    
    # Gradiente: dC/dY
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    
    return kl_divergence, grad


# **Gradiente descendente**
# 
# El gradiente descendente actualiza los valores de la función minimizando la divergencia KL. 
# 
# Nos detenemos cuando la norma del gradiente está por debajo del umbral o cuando alcanzamos el número máximo de iteraciones sin progresar.
# 

# In[8]:


# Definición del gradiente descendiente

def _gradient_descent(obj_func, p0, args, it=0, n_iter=1000,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7):
    
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it
    
    for i in range(it, n_iter):
        error, grad = obj_func(p, *args)
        grad_norm = linalg.norm(grad)
        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update
        
        print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      % (i + 1, error, grad_norm))
        
        if error < best_error:
                best_error = error
                best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break
        
        if grad_norm <= min_grad_norm:
            break
    return p


# In[9]:


# Tenemos todo listo, usemoslo!
X_embedded = fit(X)

# Grafiquemos
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)


# In[10]:


# Veamos la versión ya implementada e.e
tsne = TSNE()
X_embedded = tsne.fit_transform(X)
# Grafiquemos
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)


# In[ ]:




