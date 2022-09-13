import pandas as pd
import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib as plt
%matplotlib inline

import requests
import urllib.request
from bs4 import BeautifulSoup





def read_UCSC(loc,chrom):

    url = 'http://genome.ucsc.edu/cgi-bin/hgc?hgsid=881074721_LpmxtsYGvqbXIBttNRLbIV3G45mP&c=chr' + str(chrom) + '&l=' + str(loc) + '&r=' + str(loc+1)+ '&o=' + str(loc) + '&t=' + str(loc+1) +'&g=phyloP100wayAll&i=phyloP100wayAll'
    response = requests.get(url)
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    tds = soup.findAll(attrs={'align':'right'})
    
    if tds:
        cons = float(tds[0].text)
    else:
        cons = 0
    print(cons)
    return cons

def Maf_rank(var):
    if var[0] == "1" or var[0] == "2":
        return 1
    elif var[0] == "3" or var[0] == "4":
        return 2
    else:
        return 3
def Cand_rank(var):
    if "SHigh" in var:
        return 5
    
    elif "High" in var:
        return 4
    
    elif "Medium" in var:
        return 3
    elif "Low" in var:
        return 2
    elif "SLow" in var:
        return 1
def kg(var):
    if "." in var or "New" in var:
        return 0
    elif "kw" in var:
        return 1
    else:
        return 2
    
def chb_db(var):
    if var[-1] == "N":
        return 20
    elif var[-1] == "*":
        return chb_db(var[0:-1])
    else:
        return int(var[3:])
#def ex_func(var):
 #   if var[0] == "1" or var[0] == "2" or var[0] == "3"
        


inp = pd.read_excel("C://Users/emani/Desktop/inputs/train_data.xlsx",sheet_name = "High Filters", header = 3)

cols = 	inp.columns



#print(answer.columns)

#Below, we create a dict called fams that stores the family as keys and the gene answer list as the values

def train(inp, answer):
    inp = inp[[str(item)!= "nan" for item in inp[cols[47]]]]
    inp = inp[[not("#" in item) for item in inp["Genes.rank"]]]
    inp = inp.reset_index(drop=True)
    
    fams = {}
    for ind in range(len(answer["Genes.PI"])):
        item = answer["Genes.PI"][ind]
        
        if not("." in item or "no" in item):
            fam = answer["Family"][ind]
            
            geneList = item.split(",")
            
            if fam in inp["Family.ID"].unique():
                fams[fam] = geneList
            #print(geneList)
    
    
    #print(inp["Genes.rank"])
    correct = []
    numb = 0
    for inds in range(len(inp["Genes.rank"])):
        gene = inp["Genes.rank"][inds]
        fam = inp["Family.ID"][inds]
        corr = False
        
        
        for item in fams[fam]:
            #print(item)
            if (item in gene) or (gene in item):
                corr = True
                numb += 1
                break
        correct.append(corr)
        
    inp["correct"] = correct
    inp["correct"]= [int(item) for item in inp["correct"]]
    
    
    
    
    #inp["Conservation"] = [read_UCSC(int(item)) for item in inp["start"]]
    
    
    #Select the proper columns
    key_col = inp[cols[[2,3,12,13,19,23,24,28,47,48,49,54,55,56,132,133]]]
    
    #remove any # from column
    key_col = key_col[[not("#" in item) for item in key_col["Genes.rank"]]]
    
    #cols2 = key_col.columns
    #key_col[cols[1]] = [int(item[0]) for item in inp[cols[1]]]
    key_col[cols[2]] = [int("Y" in item) for item in inp[cols[2]]]
    key_col[cols[2]] = pd.to_numeric(key_col[cols[2]])
    
    
    
        
    key_col[cols[3]] = [Maf_rank(item) for item in inp[cols[3]]]
    #key_col[cols[10]] = pd.to_numeric(key_col[cols[10]])
    
    
    key_col[cols[12]] = [Cand_rank(item) for item in inp[cols[12]]]
   # key_col[cols[13]] = [kg(item) for item in inp[cols[13]]]
    #key_col[cols[18]] = [item.count("Y") for item in inp[cols[18]]]
  #  key_col[cols[19]] = [chb_db(item) for item in inp[cols[19]]]
    #key_col[cols[19]] = pd.to_numeric(key_col[cols[19]])
    key_col[cols[23]] = [int(not("Incomplete" in item)) for item in inp[cols[23]]]
    key_col[cols[28]] = [int(item[0]) for item in inp[cols[28]]]
    key_col[cols[47]] = pd.to_numeric(key_col[cols[47]])
    key_col[cols[48]] = pd.to_numeric(key_col[cols[48]])
    key_col[cols[49]] = pd.to_numeric(key_col[cols[49]])
    
    
    
    
   # key_col[cols[54]] = pd.to_numeric(key_col[cols[54]])
   # key_col[cols[55]] = [int(str(item) != "nan") for item in inp[cols[55]]]
   # key_col[cols[56]] = [int(str(item) != "nan") for item in inp[cols[56]]]
    key_col[cols[132]] = pd.to_numeric(key_col[cols[132]])
    key_col[cols[133]] = pd.to_numeric(key_col[cols[133]])
    
    
    key_col["ans"] = inp["correct"]
    test = key_col
    
    #key_col = key_col[["nan" in item for item in key_col["ExAC.pLI"]]]
    
    #key_col[cols[19]] = pd.to_numeric(key_col[cols[19]])
    
    
    test = test.dropna()
    ans = test["ans"].to_numpy()
    ins = test[cols[[2,3,12,23,28,47,48,49,132,133]]].to_numpy()
    
    
    clf = LogisticRegression(random_state=0, max_iter= 100000).fit(ins,ans)
    fit_func = clf.coef_
    weights = pd.DataFrame()
    weights["Column"] = cols[[2,3,12,23,28,47,48,49,132,133]]
    weights["Weight"] = fit_func[0]
    
    #Use exponential scale to convert values to analog 0-1 range from -inf to inf range
    scaled = [((2.7**(np.dot((clf.coef_), ins[n])))/(2.7**(np.dot((clf.coef_), ins[n])) +1))[0] for n in range(len(ins))] 
    scaled_table = pd.DataFrame({"Gene": test["Genes.rank"], "Prediction" : scaled})
    
    fpr, tpr, thresholds = metrics.roc_curve(ans,scaled, pos_label=1)
    
    fig, ax = plt.pyplot.subplots()
    ax.set_title('ROC Curve Simple Logistic Regression')
    ax.plot(fpr, tpr)
    ax.plot(fpr,fpr)
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    
    # place a text box in upper left in axes coords
    score = metrics.roc_auc_score(ans,scaled)
    ax.text(0.65, 0.15, "Area UC: " + str(round(score, 3)) , transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
    return {"Regression object": clf, "ROC AUC score": score, "Weights": weights, "Number correct": numb}



###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

def test(inp_test, answer_test, clf):

    inp_test = inp_test[[str(item)!= "nan" for item in inp_test[cols[47]]]]
    inp_test = inp_test[[not("#" in item) for item in inp_test["MuPh"]]]
    inp_test = inp_test.reset_index(drop=True)
    
    
    fams_test = {}
    for ind in range(len(answer_test["Genes.PI"])):
        item = answer_test["Genes.PI"][ind]
        
        if not("." in item or "no" in item):
            fam = answer_test["Family"][ind]
    
    
            
            geneList = item.split(",")
            
            if fam in inp_test["Family.ID"].unique():
                fams_test[fam] = geneList
            #print(geneList)
    
    
    #print(inp["Genes.rank"])
    correct_test = []
    numb = 0
    for inds in range(len(inp_test["Genes.rank"])):
        gene = inp_test["Genes.rank"][inds]
        fam = inp_test["Family.ID"][inds]
        corr = False
        
        for item in fams_test[fam]:
            #print(item)
            if (item in gene) or (gene in item):
                corr = True
                numb += 1
                break
        correct_test.append(corr)
        
    inp_test["correct"] = correct_test
    inp_test["correct"]= [int(item) for item in inp_test["correct"]]
    
    key_col = inp_test[cols[[2,3,12,13,19,23,24,28,47,48,49,54,55,56,132,133]]]
    
    #remove any # from column
    key_col = key_col[[not("#" in item) for item in key_col["Genes.rank"]]]
    
    #cols2 = key_col.columns
    key_col[cols[1]] = [int(item[0]) for item in inp_test[cols[1]]]
    key_col[cols[2]] = [int("Y" in item) for item in inp_test[cols[2]]]
    key_col[cols[2]] = pd.to_numeric(key_col[cols[2]])
    
    
    
        
    key_col[cols[3]] = [Maf_rank(item) for item in inp_test[cols[3]]]
    #key_col[cols[10]] = pd.to_numeric(key_col[cols[10]])
    
    
    key_col[cols[12]] = [Cand_rank(item) for item in inp_test[cols[12]]]
    key_col[cols[13]] = [kg(item) for item in inp_test[cols[13]]]
    #key_col[cols[18]] = [item.count("Y") for item in inp[cols[18]]]
    key_col[cols[19]] = [chb_db(item) for item in inp_test[cols[19]]]
    key_col[cols[19]] = pd.to_numeric(key_col[cols[19]])
    key_col[cols[23]] = [int(not("Incomplete" in item)) for item in inp_test[cols[23]]]
    key_col[cols[28]] = [int(int(item[0])<6) for item in inp_test[cols[28]]]
    key_col[cols[47]] = pd.to_numeric(key_col[cols[47]])
    key_col[cols[48]] = pd.to_numeric(key_col[cols[48]])
    key_col[cols[49]] = pd.to_numeric(key_col[cols[49]])
    
    
    
    
    key_col[cols[54]] = pd.to_numeric(key_col[cols[54]])
    key_col[cols[55]] = [int(str(item) != "nan") for item in inp_test[cols[55]]]
    key_col[cols[56]] = [int(str(item) != "nan") for item in inp_test[cols[56]]]
    key_col[cols[132]] = pd.to_numeric(key_col[cols[132]])
    key_col[cols[133]] = pd.to_numeric(key_col[cols[133]])
    
    
    key_col["ans"] = inp_test["correct"]
    test_test = key_col#.groupby(cols[24], as_index = False).min()

    
    #key_col = key_col[["nan" in item for item in key_col["ExAC.pLI"]]]
    
    #key_col[cols[19]] = pd.to_numeric(key_col[cols[19]])
    
    
    test_test = test_test.dropna()
    ans = test_test["ans"].to_numpy()
    ins = test_test[cols[[2,3,12,23,28,47,48,49,132,133]]].to_numpy()
    
    scaled = [((2.7**(np.dot((clf.coef_), ins[n])))/(2.7**(np.dot((clf.coef_), ins[n])) +1))[0] for n in range(len(ins))] 
    scaled_table = pd.DataFrame({"Gene": test_test["Genes.rank"], "Prediction" : scaled})
    
    score = metrics.roc_auc_score(ans,scaled)
    return {"ROC AUC Score": score, "Prediction table": scaled_table, "Number correct": numb}



###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################





inp_train = pd.read_excel("C://Users/emani/Desktop/inputs/train_data.xlsx",sheet_name = "High Filters", header = 3)
answer_train = pd.read_excel("C://Users/emani/Desktop/inputs/train_data.xlsx",sheet_name = "Summary", header = 0)

input_test = pd.read_excel("C://Users/emani/Desktop/inputs/test_data_1.xlsx",sheet_name = "High Filters", header = 3)
ans_test = pd.read_excel("C://Users/emani/Desktop/inputs/test_data_1.xlsx",sheet_name = "Summary", header = 0)





result = pd.concat([inp_train, input_test], ignore_index=True, sort=False)
result_ans = pd.concat([answer_train, ans_test], ignore_index=True, sort=False)


train_result = train(inp_train,answer_train)


result_output = pd.DataFrame({"Coef": train_result["Regression object"].coef_[0], "Columns": cols[[2,3,12,23,28,47,48,49,132,133]]})
result_output.to_csv("C://Users/emani/Desktop/inputs/weights.csv")









test_result = test(input_test,ans_test, train_result["Regression object"])





