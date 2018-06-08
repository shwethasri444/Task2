
# coding: utf-8

# In[3]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# In[4]:


df= pd.read_csv("C:\\Users\\admin\\Desktop\\shwetha\\Task2\\turbine_data.csv", error_bad_lines=False,lineterminator='\n', encoding="utf-8",low_memory=False)
df= df.dropna()
df= df.drop(df.columns[[0]],axis=1)
variables=pd.DataFrame()
#variables=np.array((1,1038))
total_columns=1038
variables=list(df)
# for i in range(total_columns):
#     print(variables[i])
#     print(type(variables[i]))
print(type(variables))

#print(df.head(5))


# In[5]:


scaler = StandardScaler()
scaler.fit(df)


# In[6]:


data = scaler.transform(df)
print(data[0:5,:])


# In[7]:


pca=PCA(0.95)
principalComponents = pca.fit_transform(data)
print(principalComponents[0:5,:])


# In[8]:


principal_df = pd.DataFrame(data=principalComponents)


# In[9]:


pca.explained_variance_ratio_


# In[10]:


pca.components_


# In[11]:


pca.explained_variance_


# In[12]:


no_of_pc=3
loadings=np.zeros((4,1038))
# the values are initialised to 0
for i in range(no_of_pc):
    loadings[i+1] = pca.components_[i]* np.sqrt(pca.explained_variance_[i])       


# In[13]:


tag_desc = pd.read_csv("C:\\Users\\admin\\Desktop\\shwetha\\Task2\\Tag_desc.csv",encoding='utf-8')
tag_dic = dict(zip(tag_desc["Tag"],tag_desc["Description"]))
str(tag_dic)
#len(tag_dic)


# In[39]:


loadings = pd.DataFrame(loadings)
loadings_transpose = loadings.transpose()
loadings_transpose.columns = ["Variables","PC1","PC2","PC3"]
count=1038
variables1 = pd.DataFrame(np.nan, index=range(0,1037), columns=['A'])
for i in range(count-1):
    variables1.loc[i,'A']=tag_dic[variables[i]]
loadings_transpose["Variables"]=variables1
print(loadings_transpose)

# #establish the direct or inverse relationship between them
# #yay you are done


# In[33]:


pc1 = pd.DataFrame()
pc1["variables"] = loadings_transpose["Variables"]
pc1["PC1"] = loadings_transpose["PC1"]
pc1_low_limit = pc1["PC1"]>0.5
pc1_up_limit = pc1["PC1"]<-0.5
pc1_imp = pc1[pc1_low_limit | pc1_up_limit]
print(len(pc1_imp))
print(pc1_imp)


# In[40]:


pc2 = pd.DataFrame()
pc2["variables"] = loadings_transpose["Variables"]
pc2["PC2"] = loadings_transpose["PC2"]
pc2_low_limit = pc2["PC2"]>0.5
pc2_up_limit = pc2["PC2"]<-0.5
pc2_imp = pc2[pc2_low_limit | pc2_up_limit]
print(len(pc2_imp))
print(pc2_imp)


# In[41]:


pc3 = pd.DataFrame()
pc3["variables"] = loadings_transpose["Variables"]
pc3["PC3"] = loadings_transpose["PC3"]
pc3_low_limit = pc3["PC3"]>0.5
pc3_up_limit = pc3["PC3"]<-0.5
pc3_imp = pc3[pc3_low_limit | pc3_up_limit]
print(len(pc3_imp))
print(pc3_imp)

