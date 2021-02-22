import numpy as np
import math
import pandas as pd
import numpy.matlib as b
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
import time
from sklearn.model_selection import KFold
from entropy_estimators import *

start = time.time()
def Input():
    # Read the data from the txt file
    sample = pd.read_csv('Test.csv',header=None)
    (N, L) = np.shape(sample)  
    dim = L - 1
    
    label1 = sample.iloc[:,L-1]
    label = label1.values
    data = sample.iloc[:,0:dim]   
    # NewData = normalize(data)
    NewData = Pre_Data(data)
    return NewData,label

def FeatureType(data):
    [N,dim] = np.shape(data)
    th = round(N**0.5)
    F_cont = []
    F_disc = []
    for j in range(dim):
        temp_unique = np.unique(data[:,j])
        if len(temp_unique) > th:
            F_cont.append(j)
        else:
            F_disc.append(j)
    return F_cont, F_disc
        
def Pre_Data(data):
    [N,L] = np.shape(data)
    scaler = MinMaxScaler()
    scaler.fit(data)
    NewData = scaler.transform(data)
    return NewData
                
def Distribution_Est(data, dim):
    DC_mean = np.zeros(dim)
    DC_std = np.zeros(dim)
    
    for i in range(dim):
        TempClass = data[:,i]
        DC_mean[i] = np.mean(TempClass)
        DC_std[i] = np.std(TempClass)
        
    return DC_mean,DC_std

def Feature_Dist1(DC_means,DC_std,data,Var,dim,Corr):
    
    DisC = np.zeros((dim,dim))
    Dist = []
    for i in range(dim):
        
        for j in range(i,dim):
            
            DisC[i,j] = KLD_Cal(data,i,j,Var,Corr)
            DisC[j,i] = DisC[i,j]
            Dist.append(DisC[i,j])
    
    return DisC,Dist

def Feature_Dist2(data,dim):
    Dist = []
    DisC = np.zeros((dim,dim))
    
    for i in range(dim): 
        for j in range(i,dim):
            DisC[i,j] = Sym_Cal(data,i,j)
            DisC[j,i] = DisC[i,j]
            Dist.append(DisC[i,j])
    return DisC,Dist

def KLD_Cal(data,i,j,Var,Corr):
    Var1 = Var[i]
    Var2 = Var[j]

    P = Corr[i,j]
    Sim = Var1 + Var2 - ((Var1 + Var2)**2 - 4 * Var1 * Var2 * (1 - P**2))**0.5
    D_KL = Sim / (Var1 + Var2)
    
    return D_KL 

def Sym_Cal(data,i,j):
    I_ij = midd(data[:,i],data[:,j])
    H_I = entropyd(data[:,i])
    H_J = entropyd(data[:,j])

    if (H_I + H_J) == 0:
        D_KL = 1
    else:
        D_KL = 1 - 2*(I_ij)/(H_I + H_J)
    return D_KL

def fitness_cal(DisC, DC_means, DC_std, data, StdF, gamma):
    fitness = np.zeros(len(DC_means))
    # print(np.shape(fitness))
    for i in range(len(DC_means)):
        TempSum = 0
        for j in range(len(DC_means)):
            if j != i:
                D = DisC[i,j]
                TempSum = TempSum + (math.exp(- (D**2) / StdF))**gamma
        fitness[i] = TempSum
    return fitness

def Pseduo_Peaks1(DisC, Dist, DC_Mean, DC_Std, data, fitness, StdF, gamma, Var):
    
    # The temporal sample space in terms of mean and standard deviation
    sample = np.vstack((DC_Mean,DC_Std)).T

    # Search Stage of Pseduo Clusters at the temporal sample space
    NeiRad = 0.01*max(Dist) #0.01
    # NeiRad = (StdF/gamma)
    i = 0
    marked = []
    C_Indices = np.arange(1, len(DC_Mean)+1) # The pseduo Cluster label of features
    PeakIndices = []
    Pfitness = []
    co = []
    F = fitness
    while True:
        
        PeakIndices.append(np.argmax(F))
        Pfitness.append(np.max(F))
    
        indices = NeighborSearch1(DisC, data, sample, PeakIndices[i], marked, NeiRad, Var)
        
        C_Indices[indices] = PeakIndices[i]
        if len(indices) == 0:
            indices=[PeakIndices[i]]
        
        co.append(len(indices)) # Number of samples belong to the current 
    # identified pseduo cluster
        marked = np.concatenate(([marked,indices]))
  
        # Fitness Proportionate Sharing
        F = Sharing(F, indices) 
        
        # Check whether all of samples has been assigned a pseduo cluster label
        if np.sum(co) >= (len(F)):
            
            break
        
        i=i+1 # Expand the size of the pseduo cluster set by 1

    C_Indices = Close_FCluster(PeakIndices,DisC,np.shape(DisC)[0])
    return PeakIndices,Pfitness,C_Indices
def Pseduo_Peaks2(DisC, Dist, DC_Mean, DC_Std, data, fitness, StdF, gamma):
    
    # The temporal sample space in terms of mean and standard deviation
    sample = np.vstack((DC_Mean,DC_Std)).T
    # Spread= np.max(Dist)

    # Search Stage of Pseduo Clusters at the temporal sample space
#    NeiRad = 0.25 * StdF
    NeiRad = 0.01*np.max(Dist)
    # NeiRad = (StdF/gamma)
    i = 0
    marked = []
    C_Indices = np.arange(1, len(DC_Mean)+1) # The pseduo Cluster label of features
    PeakIndices = []
    Pfitness = []
    co = []
    F = fitness
    while True:
        
        PeakIndices.append(np.argmax(F))
        Pfitness.append(np.max(F))
    
        indices = NeighborSearch2(DisC, data, sample, PeakIndices[i], marked, NeiRad)
        
        C_Indices[indices] = PeakIndices[i]
        if len(indices) == 0:
            indices=[PeakIndices[i]]
        
        co.append(len(indices)) # Number of samples belong to the current 
    # identified pseduo cluster
        marked = np.concatenate(([marked,indices]))
  
        # Fitness Proportionate Sharing
        F = Sharing(F, indices) 
        
        # Check whether all of samples has been assigned a pseduo cluster label
        if np.sum(co) >= (len(F)):
            break
        i=i+1 # Expand the size of the pseduo cluster set by 1
    C_Indices = Close_FCluster(PeakIndices, DisC, np.shape(DisC)[0])
    return PeakIndices,Pfitness,C_Indices
def NeighborSearch1(DisC, data, sample, P_indice, marked, radius, Var):
    Cluster = []    
    for i in range(np.shape(sample)[0]):      
        if i not in marked:            
            Dist = DisC[i, P_indice]            
            if Dist <= radius:
                Cluster.append(i)
    Indices = Cluster    
    return Indices

def NeighborSearch2(DisC, data, sample, P_indice, marked, radius):
    Cluster = []   
    for i in range(np.shape(sample)[0]):        
        if i not in marked:            
            Dist = DisC[i, P_indice]            
            if Dist <= radius:
                Cluster.append(i)
    Indices = Cluster   
    return Indices

def Sharing(fitness, indices):
    newfitness = fitness
    sum1 = 0
    for j in range(len(indices)):
        sum1 = sum1 + fitness[indices[j]]
    for th in range(len(indices)):
            newfitness[indices[th]] = fitness[indices[th]] / (1+sum1)
            
    return newfitness
    
def Pseduo_Evolve(DisC, PeakIndices, PseDuoF, C_Indices, DC_Mean, DC_Std, data, fitness, StdF, gamma):
    
    # Initialize the indices of Historical Pseduo Clusters and their fitness values
    HistCluster = PeakIndices
    HistClusterF = PseDuoF
    while True:
        # Call the merge function in each iteration
        [Cluster,Cfitness,F_Indices] = Pseduo_Merge(DisC, HistCluster, HistClusterF, C_Indices, DC_Mean, DC_Std, data, fitness, StdF, gamma)
        # Check for the stablization of clutser evolution and exit the loop
        if len(np.unique(Cluster)) == len(np.unique(HistCluster)):
            break
    
        # Update the feature indices of historical pseduo feature clusters and
        # their corresponding fitness values
    
        HistCluster=Cluster
        HistClusterF=Cfitness
        C_Indices = F_Indices
    # Compute final evolved feature cluster information 
    FCluster = np.unique(Cluster)
    Ffitness = Cfitness
    C_Indices = F_Indices
    
    return FCluster, Ffitness, C_Indices
#----------------------------------------------------------------------------------------------------------
def Pseduo_Merge(DisC, PeakIndices, PseDuoF, C_Indices, DC_Mean, DC_Std, data, fitness, StdF, gamma):
    if len(PeakIndices) == 1:
        FCluster = PeakIndices
        Ffitness = fitness[FCluster]
        F_Indices = Close_FCluster(FCluster, DisC, np.shape(DisC)[0])
        return FCluster, Ffitness, F_Indices
    # Initialize the pseduo feature clusters lables for all features 
    F_Indices = C_Indices
    # Initialize the temporal sample space for feature means and stds
    sample = np.vstack((DC_Mean,DC_Std)).T
    ML = [] # Initialize the merge list as empty
    marked = [] #List of checked Pseduo Clusters Indices 
    Unmarked = [] # List of unmerged Pseduo Clusters Indices 
    for i in range(len(PeakIndices)):
            M = 1 # Set the merge flag as default zero
            MinDist = math.inf # Set the default Minimum distance between two feature clusters as infinite
            MinIndice = -1 # Set the default Neighboring feature cluster indices as zero
            # Check the current Pseduo Feature Cluster has been evaluated or not
            if PeakIndices[i] not in marked:
                for j in range(len(PeakIndices)):
                        if j != i:
                            # Divergence Calculation between two pseduo feature clusters
                            D = DisC[PeakIndices[i], PeakIndices[j]]
                            if MinDist > D:
                                MinDist = D
                                MinIndice = j
                if MinIndice >= 0:
                    # Current feature pseduo cluster under check
                    Current = sample[PeakIndices[i],:]
                    CurrentFit = PseDuoF[i]
                    # Neighboring feature pseduo cluster of the current checked cluster
                    Neighbor = sample[PeakIndices[MinIndice],:]
                    NeighborFit = PseDuoF[MinIndice]
                    
                    # A function to identify the bounady feature instance between two 
                    # neighboring pseduo feature clusters
                    BP=Boundary_Points(DisC, F_Indices,data, PeakIndices[i], PeakIndices[MinIndice])
                    BPF=fitness[BP]
                    if BPF<1*min(CurrentFit,NeighborFit):
                        M=0 # Change the Merge flag
                    
                    if M == 1:
                        ML.append([PeakIndices[i],PeakIndices[MinIndice]])
                        marked.append(PeakIndices[i])
                        marked.append(PeakIndices[MinIndice])
                    else:
                        Unmarked.append(PeakIndices[i])
    NewPI = []
    # Update the pseduo feature clusters list with the obtained mergelist 
    for m in range(np.shape(ML)[0]):
        # print(ML[m][0],ML[m][1])
        if fitness[ML[m][0]] > fitness[ML[m][1]]:
            NewPI.append(ML[m][0])
            F_Indices[C_Indices==ML[m][1]] = ML[m][0]
        else:
            NewPI.append(ML[m][1])
            F_Indices[C_Indices==ML[m][0]] = ML[m][1]
    # Update the pseduo feature clusters list with pseduo clusters that have not appeared in the merge list 
    for n in range(len(PeakIndices)):
        if PeakIndices[n] in Unmarked:
            NewPI.append(PeakIndices[n])

    # Updated pseduo feature clusters information after merging
    FCluster = np.unique(NewPI)
    FCluster = FCluster.astype(int)
    Ffitness = fitness[FCluster]
    F_Indices = Close_FCluster(FCluster, DisC, np.shape(DisC)[0])
    return FCluster, Ffitness, F_Indices

def Boundary_Points(DisC, F_Indices, data, Current, Neighbor):
    
    [N, dim] = np.shape(data)
    TempCluster1 = np.where(F_Indices == Current)
    TempCluster2 = np.where(F_Indices == Neighbor)
    
    TempCluster = np.append(TempCluster1,TempCluster2)
    D = []
#    D = np.inf
#    FI = Current
#    print(len(TempCluster))
    for i in range(len(TempCluster)):
        D1 = DisC[TempCluster[i], Current]
        D2 = DisC[TempCluster[i], Neighbor]
#        if D < abs(D1-D2):
#            D = abs(D1-D2)
#            FI = i
        D.append(abs(D1 - D2))
    if not D:
        BD = Current
    else:
        FI = np.argmin(D)
        BD = TempCluster[FI]
    
    return BD

def PseduoGeneration(PseP,N):
    
    Pse_Mean = PseP[:,0]
    Pse_Std = PseP[:,1]
    
    # Data = (np.zeros((N,len(Pse_Mean))))
    
    Data = np.zeros((N,len(Pse_Mean)))
    
    for i in range(len(Pse_Mean)):
        
        Data[:, i] = (np.repeat(Pse_Mean[i],N) + Pse_Std[i] * np.random.randn(N)).T
    
    return Data

def Psefitness_cal( PseP, sample, data, PseduoData, StdF, gamma):
    
    OriFN = np.shape(sample)[0]
    PN = np.shape(PseP)[0]
    PsePF = np.zeros(PN)
    
    
    for i in range(PN):
        
        TempSum = 0
        
        for j in range(OriFN):
            
            Var1 = np.var(data[:,j])
            Var2 = np.var(PseduoData[:,i])
            
            
            
            P = np.corrcoef(data[:,j],PseduoData[:,i])[0,1]
            
            Sim = Var1 + Var2 - ((Var1 + Var2)**2 - 4 * Var1 * Var2 * (1 - P**2))**0.5
            
            D_KL = Sim / (Var1 + Var2)
            
            TempSum = TempSum + (math.exp(-(D_KL**2)/StdF))**gamma
        PsePF[i] = TempSum
    return PsePF
 
def Close_FCluster(FCluster,DisC,dim):
    F_Indices = np.arange(dim)
    for i in range(dim):
        dist_fcluster = DisC[i,FCluster]
        F_Indices[i] = FCluster[np.argmin(dist_fcluster)]
    return F_Indices 

def ContinousFeatures(data,label,f_cont):
    if len(f_cont) < 1:
        return []
    if len(f_cont) == 1:
        return f_cont
    contin_sample = data[:,f_cont]
    [N1, dim1] = np.shape(contin_sample)
    [DC_means1, DC_std1] = Distribution_Est(contin_sample,dim1)  
    Var1 = np.var(contin_sample,axis=0)
    Corr1 = np.corrcoef(contin_sample.T)
    DisC1,Dist1 =  Feature_Dist1(DC_means1,DC_std1,contin_sample,Var1,dim1,Corr1)
    StdF1 = (np.mean(np.power(Dist1,0.5)))**2
    gamma1 = 5
        
    fitness1 = fitness_cal(DisC1, DC_means1, DC_std1, contin_sample, StdF1, gamma1)
    oldfitness1 = np.copy(fitness1)
    [PeakIndices1,Pfitness1,C_Indices1] = Pseduo_Peaks1(DisC1, Dist1, DC_means1,
    DC_std1,contin_sample,fitness1,StdF1,gamma1, Var1)
            
    fitness1 = oldfitness1
    # Pseduo Clusters Infomormation Extraction
    PseDuo1 = DC_means1[PeakIndices1] # Pseduo Feature Cluster centers
    PseDuoF1 = Pfitness1 # Pseduo Feature Clusters fitness values
    #-------------Check for possible merges among pseduo clusters-----------#
    [FCluster1,Ffitness1,C_Indices1] = Pseduo_Evolve(DisC1, PeakIndices1, 
    PseDuoF1, C_Indices1, DC_means1, DC_std1, contin_sample, fitness1, StdF1, gamma1)
    
    SF1 = []
            
    label = label.reshape(N,)
            
    C_Indices1 = Close_FCluster(FCluster1,DisC1,dim1)
            
    for i in FCluster1:
        tempf_cluster1 = np.where(C_Indices1==i)[0]
        if len(tempf_cluster1) > 1:
            temp_fea1 = data[:,tempf_cluster1]
            f_rel1 = mutual_info_classif(temp_fea1,label)
#               f_rel = []
#               for j in range(len(tempf_cluster)):
#                   temp_fea = data[:,tempf_cluster[j]]
#                   temp_fea = temp_fea.reshape(N,1)
#                   f_rel.append(mutual_info_classif(temp_fea,label))
            SF1.append(tempf_cluster1[np.argmax(f_rel1)])
        else:
            SF1.append(i)
    return f_cont[SF1]

def DiscreteFeatures(data,label,f_disc):
    if len(f_disc) < 1:
        return []
    disct_sample = data[:,f_disc]
    [N2, dim2] = np.shape(disct_sample)
    [DC_means2, DC_std2] = Distribution_Est(disct_sample,dim2)  
    
    DisC2,Dist2 =  Feature_Dist2(disct_sample,dim2)
    StdF2 = max(Dist2)
    gamma2 = 5
        
    fitness2 = fitness_cal(DisC2, DC_means2, DC_std2, disct_sample, StdF2, gamma2)
    oldfitness2 = np.copy(fitness2)
    [PeakIndices2,Pfitness2,C_Indices2] = Pseduo_Peaks2(DisC2, Dist2, DC_means2,
    DC_std2,disct_sample,fitness2,StdF2,gamma2)
            
    fitness2 = oldfitness2
    # Pseduo Clusters Infomormation Extraction
    PseDuo2 = DC_means2[PeakIndices2] # Pseduo Feature Cluster centers
    PseDuoF2 = Pfitness2 # Pseduo Feature Clusters fitness values
    #-------------Check for possible merges among pseduo clusters-----------#
    [FCluster2,Ffitness2,C_Indices2] = Pseduo_Evolve(DisC2, PeakIndices2, 
    PseDuoF2, C_Indices2, DC_means2, DC_std2, disct_sample, fitness2, StdF2, gamma2)
    
    SF2 = []
            
    label = label.reshape(N,)
            
    C_Indices2 = Close_FCluster(FCluster2,DisC2,dim2)
            
    for i in FCluster2:
        tempf_cluster2 = np.where(C_Indices2==i)[0]
        if len(tempf_cluster2) > 1:
            temp_fea2 = data[:,tempf_cluster2]
            f_rel2 = mutual_info_classif(temp_fea2,label)
#               f_rel = []
#               for j in range(len(tempf_cluster)):
#                   temp_fea = data[:,tempf_cluster[j]]
#                   temp_fea = temp_fea.reshape(N,1)
#                   f_rel.append(mutual_info_classif(temp_fea,label))
            SF2.append(tempf_cluster2[np.argmax(f_rel2)])
        else:
            SF2.append(i)
    return f_disc[SF2]
#--------------------------------------------------------------------------------------------------------------  
if __name__ == '__main__':
    [data1,label1] = Input()  
    
    f_cont, f_disc = FeatureType(data1)
    
    f_cont = np.asarray(f_cont)
    f_disc = np.asarray(f_disc)
    
    kf = KFold(n_splits=10,shuffle=True)
    X = data1
    Acc1 = []
    Acc2 = []
    
    for train_index, test_index in kf.split(X):
        data, test_x = X[train_index], X[test_index]
        label, test_y = label1[train_index], label1[test_index]
    
        [N, dim] = np.shape(data)
        # f_cont = np.arange(0,dim)
        
        SF1 = ContinousFeatures(data,label,f_cont)
        SF2 = DiscreteFeatures(data,label,f_disc)
        
        if len(SF2) > 0 and len(SF1) > 0:
            SF = np.concatenate([SF1,SF2])
        elif len(SF1) > 0:
            SF = SF1
        else:
            SF = SF2
        
            
        true_label = label.reshape(N,)
                
        clf1 = KNeighborsClassifier(n_neighbors=3)
        clf2 = KNeighborsClassifier(n_neighbors=3)
            
        clf1 = clf1.fit(data[:,SF],true_label)
        clf2 = clf2.fit(data,true_label)
            
        Acc1.append(clf1.score(test_x[:,SF],test_y)) 
        Acc2.append(clf2.score(test_x,test_y))
        
    
    print("Cross-validated accuracy 1: ", np.mean(Acc1))
    print("Cross-validated accuracy 2: ", np.mean(Acc2))

    print("Number of Selected Features: ", len(SF))
    end = time.time()
    print('The total time in seconds:',end-start)
    
    
        
