import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

#plt.style.use("nature.mplstyle") #use Nature style

def plot_CV_results(lambda_values, MSE_train_all, MSE_test_all, metrics="MSE", filename="plot.png", plot_mean=True, ylim=False):
    """ Plots the results of the 5-fold cross-validation for different lambda values. """

    plt.figure(figsize=(7, 5))

    if plot_mean:
        # Compute mean & standard deviation for Train & Test MSE across folds
        MSE_train_mean = [np.mean(MSE_train_all[lmbda]) for lmbda in lambda_values]
        MSE_test_mean = [np.mean(MSE_test_all[lmbda]) for lmbda in lambda_values]
        # Plot Mean Curve with Confidence Interval
        plt.plot(lambda_values, MSE_train_mean, label="Train Mean", color="blue", linewidth=2)
        plt.plot(lambda_values, MSE_test_mean, label="Test Mean", color="orange", linewidth=2)
    else:
        # Plot Individual Folds
        for i in range(len(MSE_train_all[list(MSE_train_all)[-1]])):
            train_mse_fold = [MSE_train_all[lmbda][i] for lmbda in lambda_values]
            test_mse_fold = [MSE_test_all[lmbda][i] for lmbda in lambda_values]

            plt.plot(lambda_values, train_mse_fold, linestyle="--", color="blue", alpha=0.5, label="Train Fold" if i == 0 else "")
            plt.plot(lambda_values, test_mse_fold, linestyle="--", color="orange", alpha=0.5, label="Test Fold" if i == 0 else "")

    # Final plot settings
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel(metrics)
    plt.legend()
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.grid(True)
    plt.title(metrics + " vs Lambda (5-fold CV)")
    #plt.savefig("plots/" + filename)
    plt.show()

def plot_target(rates, process, theta, name: str):
    """ Shows the activity (expression, degradation) of the target (gene, RNA...) along cell cycle. """
    num_estimates = len(rates.loc[name]) #rate is predicted multiple times for each target
    plt.title(name + " " + process[0] + " rate along cell cycle")
    plt.xlabel("\u03B8")
    plt.ylabel(process[0] + " rate")
    plt.plot(theta,rates.loc[name].mean(axis=0)+rates.loc[name].std(axis=0)/np.sqrt(num_estimates), label="\u03B1 + <\u03C3>", color="red", linestyle='dotted')
    plt.plot(theta,rates.loc[name].mean(axis=0), label="\u03B1", color="green")
    plt.plot(theta,rates.loc[name].mean(axis=0)-rates.loc[name].std(axis=0)/np.sqrt(num_estimates), label="\u03B1 - <\u03C3>", color="red", linestyle='dotted')
    plt.legend()
    plt.show()


def plot_rate_comparison(names, rates, reconstructed_rates, process, theta, target_nb=1, std_a=None, std_R=None):
    """ Compares the process rates inferred from experimental data with reconstructed data from activities """
    if std_a is None:
        std_a = np.zeros_like(rates[target_nb, :])
    if std_R is None:
        std_R = np.zeros_like(reconstructed_rates[target_nb, :])
    plt.figure(figsize=(6, 4))
    plt.title(names[target_nb])
    plt.xlabel("\u03B8")
    plt.ylabel("Transcription rate (mean-centered)")
    #plt.ylim(-0.75, 0.75)
    plt.plot(theta,rates[target_nb,:], label="scRNA-seq", color="green")
    plt.plot(theta,reconstructed_rates[target_nb,:], label="Reconstructed", color="red")
    plt.fill_between(theta, rates[target_nb,:]-std_a, rates[target_nb,:]+std_a, color="green", alpha=0.2, edgecolor=None)
    plt.fill_between(theta, reconstructed_rates[target_nb,:]-std_R, reconstructed_rates[target_nb,:]+std_R, color="red", alpha=0.2, edgecolor=None)
    plt.legend()
    plt.show()


def plot_binding_protein_activity(names, activities, process, theta, BP_nb=1, std=None):
    """ Shows the inferred activity of a binding protein (TF, RBP) """
    A = activities[BP_nb,:]
    if std is None:
        std = np.zeros_like(A)
    plt.figure(figsize=(6, 4))
    plt.title(names[BP_nb])
    plt.xlabel("\u03B8")
    plt.ylabel("Activity (mean-centered)")
    plt.plot(theta,A, color="green")
    plt.fill_between(theta, A-std, A+std, color="green", alpha=0.2, edgecolor=None)
    plt.show()


def plot_heatmap(activity, ylabels=None, display_limit=None, cmap='RdBu_r', title=""):
    """ Heatmap of the activity (expression, degradation, or TF, RBP...) throughout cell cycle (sorted by maximum along theta) """
    if (display_limit is not None):
        #We display only the biggest amplitudes until display_limit is reached.
        amp = activity.max(axis=1)-activity.min(axis=1) #Amplitude of TFs
        ind = np.flip(np.argsort(amp)) #Sort by amplitude from highest to lowest
        S = activity[ind[0:display_limit],:] #Take TFs with highest amplitude
        if (ylabels is not None):
            ylabels = np.array(ylabels[ind[0:display_limit]])
    else:
        S = activity
    if (ylabels is None):
        ylabels = False #'auto' for numbers

    I = np.argsort(np.argmax(S,axis=1)) #Sort by max activity along theta

    if isinstance(ylabels, np.ndarray):  # Ensure ylabels is an array before indexing
        ylabels = ylabels[I]
    
    H = S[I,:]
    H = (H-H.mean(axis=1,keepdims=True))/H.std(axis=1,keepdims=True) #Normalization
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    if (title != ""):
        plt.title(title)
    ax = sns.heatmap(H, cmap=cmap, cbar=True, annot=False, yticklabels=ylabels,xticklabels=False)

    # Add vertical dashed lines at positions theta1 and theta2
    theta1 = 25 #G1 -> S
    theta2 = 63 #S -> G2/M
    ax.axvline(x=theta1, color=[0.7,0.7,0.7], linestyle='--', linewidth=4)
    ax.axvline(x=theta2, color=[0.7,0.7,0.7], linestyle='--', linewidth=4)

    # Set ticks and labels for the subset
    ax.set_xticks(np.linspace(0,100,11));  # Specify tick positions
    thetalabels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    ax.set_xticklabels(thetalabels);#,fontsize=14);  # Set tick labels
    ax.tick_params(axis='y')#, labelsize=12)

    # Label
    ax.set_xlabel('Cell-cycle phase')#, fontsize=14)  # Adjust fontsize as needed
    cbar = ax.collections[0].colorbar
    cbar.set_label('Activity (z-score)')#, fontsize=14)  # Adjust label as needed

    #plt.savefig("heatmap_degradation.png")  # Save as PNG
    #plt.savefig("heatmap_degradation.pdf")  # Save as PDF
    #plt.savefig("heatmap_degradation.svg")  # Save as SVG
    
    return ylabels

def plot_heatmap_list(A, tf_names, ylabels, clip=False):
    indices = [list(tf_names).index(x) for x in ylabels]
    S = A[indices, :]

    I = np.argsort(np.argmax(S,axis=1)) #Sort by max activity along theta

    ylabels = np.array(ylabels)[I]
    H = S[I,:]
    H = (H-H.mean(axis=1,keepdims=True))/H.std(axis=1,keepdims=True) #Normalization (z-score)
    if clip:
        H = np.clip(H, -2, 2) #Prevent outlier to distort the color scale
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    #plt.title("Heatmap of transcription factors activities")
    ax = sns.heatmap(H, cmap='RdBu_r', cbar=True, annot=False, yticklabels=ylabels,xticklabels=False)

    # Add vertical dashed lines at positions theta1 and theta2
    theta1 = 25 #G1 -> S
    theta2 = 63 #S -> G2/M
    ax.axvline(x=theta1, color=[0.7,0.7,0.7], linestyle='--', linewidth=3)
    ax.axvline(x=theta2, color=[0.7,0.7,0.7], linestyle='--', linewidth=3 )

    # Set ticks and labels for the subset
    ax.set_xticks(np.linspace(0,100,11));  # Specify tick positions
    thetalabels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    ax.set_xticklabels(thetalabels);#,fontsize=14);  # Set tick labels
    ax.tick_params(axis='y')#, labelsize=12)

    # Label
    ax.set_xlabel('Cell-cycle phase')#, fontsize=14)  # Adjust fontsize as needed
    cbar = ax.collections[0].colorbar
    cbar.set_label('Activity (z-score)')#, fontsize=14)  # Adjust label as needed

    #plt.savefig("heatmap_degradation.png")  # Save as PNG
    #plt.savefig("heatmap_degradation.pdf")  # Save as PDF
    #plt.savefig("heatmap_degradation.svg")  # Save as SVG
    
    return ylabels


def plot_TF_exp_activity(theta_smooth, alpha, A, tf_names, key_tfs, tf, std1=None, std2=None):
    
    if std1 is None:
        std1 = np.zeros_like(alpha[tf, :])
    if std2 is None:
        std2 = np.zeros_like(A[list(tf_names).index(key_tfs[tf]), :])

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Plot TF expression on the left y-axis
    ax1.set_xlabel("Cell Cycle Phase (θ)")
    ax1.set_ylabel("TF Expression (α)", color='tab:red')
    
    line1, = ax1.plot(theta_smooth, alpha[tf], color='tab:red', label="snRNA-seq")
    ax1.fill_between(theta_smooth, alpha[tf]-std1, alpha[tf]+std1, color="red", alpha=0.2, edgecolor=None)

    ax1.tick_params(axis='y')

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("TF Activity (A)", color='tab:green')
    
    line2, = ax2.plot(theta_smooth, A[list(tf_names).index(key_tfs[tf])], color='tab:green', label="TF activity")
    ax2.fill_between(theta_smooth, A[list(tf_names).index(key_tfs[tf])]-std2, A[list(tf_names).index(key_tfs[tf])]+std2, color="green", alpha=0.2, edgecolor=None)
    
    ax2.tick_params(axis='y')

    # Combine all handles from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]

    # Set the legend on the first axis
    ax1.legend(lines, labels, loc="upper right")

    plt.title(key_tfs[tf])
    fig.tight_layout()
    plt.show()

    
def contiguous_segments(indices):
    """
    Given a sorted 1D array of indices, return a list of numpy arrays,
    each containing a contiguous block.
    """
    segments = []
    if len(indices) == 0:
        return segments
    current = [indices[0]]
    for idx in indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            segments.append(np.array(current))
            current = [idx]
    segments.append(np.array(current))
    return segments

def sample_contiguous_outside_mean(A, outside_idx, n_inside, n_samples, rng):
    """
    Given a activity and the outside indices (sorted),
    randomly sample n_samples contiguous blocks of length n_inside
    (only from segments where this is possible) and return the average mean and std.
    """
    segments = contiguous_segments(np.sort(outside_idx))
    # Keep only segments long enough:
    valid_segs = [seg for seg in segments if len(seg) >= n_inside]
    if not valid_segs:
        # fallback: if no contiguous segment is long enough, use the first n_inside indices from outside_idx.
        candidate = np.sort(outside_idx)
        block = candidate[:n_inside]
        return np.mean(A[block]), np.std(A[block], ddof=1)
    sample_means = []
    sample_stds = []
    for _ in range(n_samples):
        seg = rng.choice(valid_segs)
        start_range = len(seg) - n_inside + 1
        start_idx = rng.integers(low=0, high=start_range)
        block = seg[start_idx:start_idx + n_inside]
        sample_means.append(np.mean(A[block]))
        sample_stds.append(np.std(A[block], ddof=1))
    return np.mean(sample_means), np.mean(sample_stds)

def compute_tf_activity_difference(A, theta, expected_ranges, inhibitory=False, n_samples=3, random_state=42):
    """
    Given a 1D activity for a TF and theta values (assumed sorted),
    compute the mean activity in the 'inside' region defined by expected_ranges,
    and compare it to the mean over a randomly chosen contiguous block from outside,
    where the block size equals the number of theta points inside.
    """
    theta = np.array(theta)
    # Build inside mask as union of all ranges.
    inside_mask = np.zeros(theta.shape, dtype=bool)
    for (start, end) in expected_ranges:
        inside_mask |= (theta >= start) & (theta <= end)
    inside_idx = np.where(inside_mask)[0]
    outside_idx = np.where(~inside_mask)[0]
    n_inside = len(inside_idx)
    inside_mean = np.mean(A[inside_idx])
    s_inside = np.std(A[inside_idx], ddof=1)
    
    # Use our helper to sample a contiguous block from outside.
    rng = np.random.default_rng(random_state)
    outside_mean, s_outside = sample_contiguous_outside_mean(A, outside_idx, n_inside, n_samples, rng)
    
    # Pooled standard error (using standard error of the mean for each block)
    pooled_std = np.sqrt((s_inside**2)/n_inside + (s_outside**2)/n_inside)
    z_value = (inside_mean - outside_mean) / pooled_std if pooled_std > 0 else np.nan
    if inhibitory:
        z_value = -z_value
    #return inside_mean, outside_mean, pooled_std, z_value, n_inside
    return z_value

def plot_peak_theta_vs_index(alpha, A):
    # Normalize theta scale: assume 100 theta points between 0 and 1
    theta = np.linspace(0, 1, alpha.shape[1])

    # Get the index of theta with max activity
    gene_peak_indices = np.argmax(alpha, axis=1)
    tf_peak_indices = np.argmax(A, axis=1)

    gene_peak_theta = theta[gene_peak_indices]
    tf_peak_theta = theta[tf_peak_indices]

    # Sort for visual clarity
    gene_sort_idx = np.argsort(gene_peak_theta)
    tf_sort_idx = np.argsort(tf_peak_theta)

    sorted_gene_theta = gene_peak_theta[gene_sort_idx]
    sorted_tf_theta = tf_peak_theta[tf_sort_idx]

    sorted_gene_index = np.arange(len(sorted_gene_theta))
    sorted_tf_index = np.arange(len(sorted_tf_theta))

    # Plotting
    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(sorted_gene_theta, sorted_gene_index, label='Genes', color='tab:red', linewidth=2)
    ax1.set_ylabel('Gene count', color='tab:red')
    ax1.set_xlabel('Theta')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Add second axis for TFs
    ax2 = ax1.twinx()
    ax2.plot(sorted_tf_theta, sorted_tf_index, label='Transcription Factors', color='tab:green', linewidth=2)
    ax2.set_ylabel('Transcription factors count', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    plt.title('Transcription and TF activity dynamics across the cell cycle')
    fig.tight_layout()
    plt.grid(True)
    plt.show()
    
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
    fourier_matrix = fourier_basis(theta_smooth, num_harmonics)  # Shape: (2*num_harmonics+1, 100)

    # Solve for all TFs using least squares
    params, _, _, _ = np.linalg.lstsq(fourier_matrix.T, data.T, rcond=None)  # Shape: (2*num_harmonics+1, TFs)

    # Compute A_smooth for all TFs
    data_smooth = params.T @ fourier_matrix  # Shape: (TFs, 100)
    
    return data_smooth

def compute_reproducibility(A_star1, A_star2, alpha1_norm, alpha2_norm, metric="TF activities"):
    """
    Compare reproducibility of inferred TF activities vs raw gene expression.
    
    Inputs:
        A_star1, A_star2: (TFs x theta) activity matrices from replicate 1 and 2
        alpha1_norm, alpha2_norm: (genes x theta) normalized expression data

    Outputs:
        corrs_A: Pearson correlations per TF (array of length TFs)
        corrs_E: Pearson correlations per gene (array of length genes)
    """
    n_tfs = A_star1.shape[0]
    n_genes = alpha1_norm.shape[0]

    corrs_A = np.zeros(n_tfs)
    for m in range(n_tfs):
        corrs_A[m], _ = pearsonr(A_star1[m, :], A_star2[m, :])

    corrs_E = np.zeros(n_genes)
    for g in range(n_genes):
        corrs_E[g], _ = pearsonr(alpha1_norm[g, :], alpha2_norm[g, :])

    # Plot
    plt.figure(figsize=(6, 4))
    plt.hist(corrs_E, bins=30, alpha=0.6, label='Gene Expression', color='tab:blue', density=True)
    plt.hist(corrs_A, bins=30, alpha=0.6, label=metric, color='tab:orange', density=True)
    plt.xlabel('Pearson Correlation Across Replicates')
    plt.ylabel('Density')
    plt.title('Reproducibility of '+ metric +' vs Gene Expression')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Print stats
    print("Mean correlation of expression profiles: ", round(np.mean(corrs_E), 3))
    print("Median correlation of expression:        ", round(np.median(corrs_E), 3))
    print("Mean correlation of "+ metric +":       ", round(np.mean(corrs_A), 3))
    print("Median correlation of "+ metric +":     ", round(np.median(corrs_A), 3))
    
    
def W_key_TF(W, tf_names, top_k=20, y_bars=None):
    # --- Step 1: Compute total weight per TF (W is >= 0) ---
    tf_importance = np.sum(W, axis=0)  # shape: (n_tfs,)
    
    # --- Step 2: Create and sort DataFrame ---
    df = pd.DataFrame({'sum_W': tf_importance}, index=tf_names)
    df = df.sort_values('sum_W', ascending=False)
    
    # --- Step 3: Plot top_k TFs ---
    top_df = df.head(top_k)
    
    #plt.figure(figsize=(9, 5))
    plt.bar(np.arange(top_k), top_df['sum_W'], yerr=y_bars, color='red')
    plt.xticks(np.arange(top_k), top_df.index, rotation=45, ha='right')
    plt.ylabel("Sum of W across genes")
    #plt.title(f"Top {top_k} TFs by ∑W (After amplitude standardization)")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.show()

    # --- Step 4: Print top TF names ---
    print(list(top_df.index))

    # --- Step 5: Return full sorted DataFrame ---
    return df

def W_key_targets_per_TF(W, tf, targetnames, top_k=20, y_bars=None):
    # --- Step 1: Compute total weight per TF (W is >= 0) ---
    tf_importance = np.sum(W, axis=0)  # shape: (n_tfs,)
    
    # --- Step 2: Create and sort DataFrame ---
    df = pd.DataFrame({'sum_W': tf_importance}, index=tf_names)
    df = df.sort_values('sum_W', ascending=False)
    
    # --- Step 3: Plot top_k TFs ---
    top_df = df.head(top_k)
    
    #plt.figure(figsize=(9, 5))
    plt.bar(np.arange(top_k), top_df['sum_W'], yerr=y_bars, color='red')
    plt.xticks(np.arange(top_k), top_df.index, rotation=45, ha='right')
    plt.ylabel("Sum of W across genes")
    #plt.title(f"Top {top_k} TFs by ∑W (After amplitude standardization)")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.show()

    # --- Step 4: Print top TF names ---
    print(list(top_df.index))

    # --- Step 5: Return full sorted DataFrame ---
    return df

def W_key_gene(W, gene_names, top_k=20):
    # --- Step 1: Compute total weight per gene ---
    gene_importance = np.sum(W, axis=1)  # shape: (n_genes,)
    
    # --- Step 2: Create and sort DataFrame ---
    df = pd.DataFrame({'sum_W': gene_importance}, index=gene_names)
    df = df.sort_values('sum_W', ascending=False)
    
    # --- Step 3: Plot top_k genes ---
    top_df = df.head(top_k)
    
    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(top_k), top_df['sum_W'], color='red')
    plt.xticks(np.arange(top_k), top_df.index, rotation=45, ha='right')
    plt.ylabel("Sum of W across TFs")
    #plt.title(f"Top {top_k} Genes by ∑W (After amplitude standardization)")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.show()

    # --- Step 4: Print top gene names ---
    #print(list(top_df.index))

    # --- Step 5: Return full sorted DataFrame ---
    return df