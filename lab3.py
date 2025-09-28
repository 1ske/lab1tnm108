from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer.DESCR)
print(len(cancer.data[cancer.target==1]))

import numpy as np
import matplotlib.pyplot as plt

fig,axes =plt.subplots(10,3, figsize=(12, 9)) # 3 columns each containing 10 figures, total 30 features
malignant=cancer.data[cancer.target==0] # define malignant
benign=cancer.data[cancer.target==1] # define benign
ax=axes.ravel()# flat axes with numpy ravel
for i in range(30):
    _,bins=np.histogram(cancer.data[:,i],bins=40)
    ax[i].hist(malignant[:,i],bins=bins,color='r',alpha=.5)  # red color for malignant class
    ax[i].hist(benign[:,i],bins=bins,color='g',alpha=0.3  )# alpha is for transparency in the overlapped region
    ax[i].set_title(cancer.feature_names[i],fontsize=9)
    ax[i].axes.get_xaxis().set_visible(False) # the x-axis coordinates are not so useful, as we just want to look how well separated the histograms are
    ax[i].set_yticks(())

ax[0].legend(['malignant','benign'],loc='best',fontsize=8)
plt.tight_layout()# let's make good plots
plt.show()




import pandas as pd
cancer_df=pd.DataFrame(cancer.data,columns=cancer.feature_names)# just convert the scikit learn data-set to pandas data-frame.
plt.subplot(1,2,1)#fisrt plot
plt.scatter(cancer_df['worst symmetry'], cancer_df['worst texture'],
s=cancer_df['worst area']*0.05, color='magenta', label='check',alpha=0.3)
plt.xlabel('Worst Symmetry',fontsize=12)
plt.ylabel('Worst Texture',fontsize=12)
plt.subplot(1,2,2)# 2nd plot
plt.scatter(cancer_df['mean radius'], cancer_df['mean concave points'], s=cancer_df['mean area']*0.05, color='purple',label='check', alpha=0.3)
plt.xlabel('Mean Radius',fontsize=12)
plt.ylabel('Mean Concave Points',fontsize=12)
plt.tight_layout()
plt.show()


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()#instantiate
scaler.fit(cancer.data) # compute the mean and standard which will be used in the next command
X_scaled=scaler.transform(cancer.data)# fit and transform can be applied together and I leave that for simple exercise
# we can check the minimum and maximum of the scaled features which we expect to be 0 and 1
print("after scaling minimum", X_scaled.min(axis=0))




from sklearn.decomposition import PCA

# Configuration: Change this number to use different amounts of components
N_COMPONENTS = 4

pca=PCA(n_components=N_COMPONENTS)
pca.fit(X_scaled)
X_pca=pca.transform(X_scaled)
#let's check the shape of X_pca array
print("shape of X_pca", X_pca.shape)



ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)




# Xax=X_pca[:,0]
# Yax=X_pca[:,1]
Xax=X_pca[:,0]
Yax=X_pca[:,1]  # Changed from X_pca[:,2] to X_pca[:,1]
labels=cancer.target
cdict={0:'red',1:'green'}
labl={0:'Malignant',1:'Benign'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5}
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
    ix=np.where(labels==l)
    ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,label=labl[l],marker=marker[l],alpha=alpha[l])
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()



plt.matshow(pca.components_,cmap='viridis')
plt.yticks(range(N_COMPONENTS),[f'{i+1}st Comp' if i==0 else f'{i+1}nd Comp' if i==1 else f'{i+1}rd Comp' for i in range(N_COMPONENTS)],fontsize=10)
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),cancer.feature_names,rotation=65,ha='left')
plt.tight_layout()
plt.show()





feature_worst=list(cancer_df.columns[20:31]) # select the 'worst' features
import seaborn as sns
s=sns.heatmap(cancer_df[feature_worst].corr(),cmap='coolwarm')
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
plt.show()


# Create comparison plots for all component pairs
labels=cancer.target
cdict={0:'red',1:'green'}
labl={0:'Malignant',1:'Benign'}
marker={0:'*',1:'o'}
alpha={0:.3, 1:.5}

# Create subplots for all possible pairs
n_pairs = N_COMPONENTS * (N_COMPONENTS - 1) // 2
fig, axes = plt.subplots(1, n_pairs, figsize=(6*n_pairs, 5))
if n_pairs == 1:
    axes = [axes]  # Make it iterable for single subplot
fig.patch.set_facecolor('white')

plot_idx = 0
for i in range(N_COMPONENTS):
    for j in range(i+1, N_COMPONENTS):
        ax = axes[plot_idx]
        for l in np.unique(labels):
            ix=np.where(labels==l)
            ax.scatter(X_pca[ix,i], X_pca[ix,j], c=cdict[l], s=40, label=labl[l], 
                      marker=marker[l], alpha=alpha[l])
        ax.set_xlabel(f"Principal Component {i+1}", fontsize=12)
        ax.set_ylabel(f"Principal Component {j+1}", fontsize=12)
        ax.set_title(f"PC{i+1} vs PC{j+1}", fontsize=14)
        ax.legend()
        plot_idx += 1

plt.tight_layout()
plt.show()

# Calculate and display explained variance ratios
print("Explained variance ratios:")
for i in range(N_COMPONENTS):
    print(f"PC{i+1}: {pca.explained_variance_ratio_[i]:.4f}")
print(f"Total variance explained by first {N_COMPONENTS} components: {np.sum(pca.explained_variance_ratio_[:N_COMPONENTS]):.4f}")

# Calculate separation metrics for each pair
from sklearn.metrics import silhouette_score

print("\nSeparation Analysis:")
for i in range(N_COMPONENTS):
    for j in range(i+1, N_COMPONENTS):
        print(f"PC{i+1} vs PC{j+1} - Silhouette Score: {silhouette_score(X_pca[:, [i,j]], labels):.4f}")

# Calculate within-class and between-class distances for each pair
def calculate_separation_ratio(data, labels):
    class_0 = data[labels == 0]
    class_1 = data[labels == 1]
    
    within_class_0 = np.mean([np.linalg.norm(p - np.mean(class_0, axis=0)) for p in class_0])
    within_class_1 = np.mean([np.linalg.norm(p - np.mean(class_1, axis=0)) for p in class_1])
    between_class = np.linalg.norm(np.mean(class_0, axis=0) - np.mean(class_1, axis=0))
    
    separation_ratio = between_class / ((within_class_0 + within_class_1) / 2)
    return separation_ratio

print(f"\nSeparation Ratios (higher is better):")
for i in range(N_COMPONENTS):
    for j in range(i+1, N_COMPONENTS):
        print(f"PC{i+1} vs PC{j+1}: {calculate_separation_ratio(X_pca[:, [i,j]], labels):.4f}")