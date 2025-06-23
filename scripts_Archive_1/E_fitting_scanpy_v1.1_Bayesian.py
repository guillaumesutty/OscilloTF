import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from utils_plot import *

def fourier_basis(theta, num_harmonics=3):
    """
    Generates a Fourier basis matrix using sine and cosine components
    """
    basis = np.zeros((2*num_harmonics,len(theta)))
    for i in range(num_harmonics):
        basis[2*i,:] = np.cos(2*(i+1)*np.pi*(theta))
        basis[2*i+1,:] = np.sin(2*(i+1)*np.pi*(theta))

    return basis

def fourier_fit(data, theta_smooth, num_harmonics=3):
    # Generate Fourier basis for fitting (for original thetas)
    fourier_matrix = fourier_basis(thetas, num_harmonics)  # Shape: (2*num_harmonics+1, len(thetas))

    # Solve for all TFs using least squares
    params, _, _, _ = np.linalg.lstsq(fourier_matrix.T, data.T, rcond=None)  # Shape: (2*num_harmonics+1, TFs)

    # Generate Fourier basis for smooth theta values
    fourier_matrix_smooth = fourier_basis(theta_smooth, num_harmonics)  # Shape: (2*num_harmonics+1, 100)

    # Compute A_smooth for all TFs
    data_smooth = params.T @ fourier_matrix_smooth  # Shape: (TFs, 100)
    
    return data_smooth

def performance_metrics(rates, reconstructed_rates):
    """ Compute MSE between rates and reconstructed rates from model """
    return mean_squared_error(rates, reconstructed_rates)

filedeepcycle = "/shared/space2/molina/Data/mESCs_2iLIF/SRR13790993/deepcycle/deepcycle.h5ad"
fileBSM = 'Data/data_binding_site_matrix.txt'
# Generate smooth theta values
theta_smooth = np.linspace(0, 1, 100)

# Reading binding site matrix and targets' rates
N = pd.read_csv(fileBSM, sep="\t",index_col=0)

adata = sc.read_h5ad(filedeepcycle)
adata.var_names_make_unique()
thetas = adata.obs['cell_cycle_theta']
E = adata.layers['Mu'].T

#PREPROCESSING
#Removal of targets that does not contain any known binding site.
tf_names = N.columns
targetnames_N = N.index
targetnames_E = adata.var_names
targetnames = targetnames_E.intersection(targetnames_N)
E = E[[targetnames_E.get_loc(name) for name in list(targetnames)], :]

# Calculates amplitude over noise of rates around the cell cycle
E_smooth = fourier_fit(E, theta_smooth) #Fit E = f(theta)
sigm = E.std(axis=1)
ampl = (E_smooth.max(axis=1)-E_smooth.min(axis=1))/2
zval = ampl/sigm
ind = zval > 2
E = E[ind,:]
targetnames = targetnames[ind]

# Filtering the arrays
N = N.loc[targetnames].to_numpy()

N = N - N.mean(axis=0,keepdims=True)
E = E - E.mean(axis=1,keepdims=True) - E.mean(axis=0,keepdims=True) + E.mean()
print("Kept genes :", E.shape[0])


from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.multioutput import MultiOutputRegressor

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store MSE results
MSE_train_all = []
MSE_test_all = []

# Create a structured NumPy array with named fields
dtype = [
    ("A", "O"),  # Object type for matrices (arrays of different shapes)
    ("alpha_1", "f8"), ("alpha_2", "f8"),
    ("lambda_1", "f8"), ("lambda_2", "f8")
]

for train_index, test_index in kf.split(N):
    N_train, N_test = N[train_index], N[test_index]
    E_train, E_test = E[train_index], E[test_index]

    # Wrap the bayesian regressor in a MultiOutputRegressor
    multi_bayesian = MultiOutputRegressor(BayesianRidge())
    multi_bayesian.fit(N_train, E_train)

    # Get coefficient matrix from each output regressor
    A = np.array([reg.coef_.T for reg in multi_bayesian.estimators_])  # Shape (TFs, cells)
    alpha_1 = [estimator.alpha_1 for estimator in multi_bayesian.estimators_]
    alpha_2 = [estimator.alpha_2 for estimator in multi_bayesian.estimators_]
    lambda_1 = [estimator.lambda_1 for estimator in multi_bayesian.estimators_]
    lambda_2 = [estimator.lambda_2 for estimator in multi_bayesian.estimators_]

    # Predict
    print(A.T.shape)
    print(N_train.shape)
    R_train = N_train @ A.T  # Transpose to match correct shape
    R_test = N_test @ A.T

    # Compute MSE
    MSE_train = performance_metrics(E_train, R_train)
    MSE_test = performance_metrics(E_test, R_test)
    print("MSE_train : ", MSE_train, "MSE_test : ", MSE_test)

    # Store results
    MSE_train_all.append(MSE_train)
    MSE_test_all.append(MSE_test)

# Compute average results across folds
MSE_train_mean = np.mean(MSE_train_all)
MSE_test_mean = np.mean(MSE_test_all)

# Print results
print("Bayesian Regression MSE Train Mean:", MSE_train_mean)
print("Bayesian Regression MSE Test Mean:", MSE_test_mean)
print("alpha_1 : ", alpha_1)
print("alpha_2 : ", alpha_2)
print("lambda_1 : ", lambda_1)
print("lambda_2 : ", lambda_2)
