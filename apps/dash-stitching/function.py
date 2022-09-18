import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# import xlwings
# import pulp
import sys
# import cplex
# from docplex.mp.model import Model

def fun_readValuation(numberOfProduct):
    data=pd.read_excel('/Users/chenyan/Downloads/AncillaryPricing/Valuation.xlsx').values #read data
    numberOfCustomers=int(data.size/numberOfProduct)
    valuation=np.zeros((numberOfProduct,numberOfCustomers))
    for i in range(numberOfProduct):
        for j in range (numberOfCustomers):
            valuation[i,j]=data[j,i]
    return valuation;

def fun_cal_Statistics(valuation):
    averageWtp=np.zeros((valuation.shape[0]))
    probability=np.zeros((valuation.shape[0]))
    for j in range (valuation.shape[1]):
        for i in range(valuation.shape[0]):
            if valuation[i,j]>0:
                probability[i]=probability[i]+1;
                averageWtp[i]=averageWtp[i]+valuation[i,j]
    for i in range(valuation.shape[0]):
        if i==0:
            probability[i]=valuation.shape[1]
        averageWtp[i]= averageWtp[i]/probability[i]
        probability[i]=probability[i]/valuation.shape[1]
    return averageWtp,probability   

def fun_cal_MarketShare_and_Profit(price,cost,valuation):
    share=np.zeros((valuation.shape[0]))
    profit=0
    surplus=np.zeros((valuation.shape[0],valuation.shape[1]))
    purchaseOrnot=np.zeros((valuation.shape[0],valuation.shape[1]))
    nonNegativeSurplus=np.zeros((valuation.shape[0],valuation.shape[1]))
    sumAncillarySurplus=np.zeros((valuation.shape[1]))
    for j in range (valuation.shape[1]):
        for i in range(valuation.shape[0]):
            surplus[i,j]=valuation[i][j]-price[i]
            if surplus[i,j]>0:
                purchaseOrnot[i,j]=1
            nonNegativeSurplus[i][j]=purchaseOrnot[i,j]*surplus[i,j]
            if i>0:
                sumAncillarySurplus[j]=sumAncillarySurplus[j]+nonNegativeSurplus[i][j]
        if surplus[0,j] +sumAncillarySurplus[j]>0:
            purchaseOrnot[0,j]=1
        else:
            purchaseOrnot[0,j]=0 
        for i in range(valuation.shape[0]):
            if i>0:
                purchaseOrnot[i,j]= purchaseOrnot[i,j]* purchaseOrnot[0,j]
    for i in range(valuation.shape[0]):
        for j in range (valuation.shape[1]):
            share[i]=share[i]+purchaseOrnot[i,j]/valuation.shape[1];
        profit=profit+(price[i]-cost[i])*share[i]
    return share,profit

def fun_Method_random_search(numberOfExperimentsRS,valuation,wtp,cost):
    priceRS=[]
    shareRS=[]
    profitRS=[]
    for i in range (numberOfExperimentsRS):
        priceRS.append([c+ c* random.uniform(-0.5,0.5) for c in wtp])
        share,profit=fun_cal_MarketShare_and_Profit(priceRS[i],cost,valuation)
        shareRS.append(share)
        profitRS.append(profit)
    profit_rs=max(profitRS)  
    price_rs=priceRS[profitRS.index(max(profitRS))]
    share_rs=shareRS[profitRS.index(max(profitRS))]
    return  profit_rs,price_rs,share_rs 

def fun_Method_Price_Experiment(numberOfExperiments,valuation,wtp,cost):
    pricePE=[]
    sharePE=[]
    profitPE=[]
    att=0.6
    step=att/10;
    ini=-0.2;
    pri= np.arange(ini,ini+att+step,step)
    for i in range (numberOfExperiments):
        pricePE.append([c+ c* pri[random.randint(0,10)] for c in wtp])
        share,profit=fun_cal_MarketShare_and_Profit(pricePE[i],cost,valuation)
        sharePE.append(share)
        profitPE.append(profit)
    profit_ex=max(profitPE)
    price_ex=pricePE[profitPE.index(max(profitPE))]
    share_ex=sharePE[profitPE.index(max(profitPE))]
    return  pricePE,sharePE,profitPE,profit_ex,price_ex,share_ex

def fun_Demand_Identification (numberOfProduct,numberOfExperiments,price,share,outshare,vectorB,matrixA):
    price=np.array(price)
    share=np.array(share)
    outshare=np.array(outshare)
    numberOfSample=numberOfExperiments;
    # Define Model
    mdl = Model('MIP') 
    
    # Define Decision Variable
    objVar={}
    cOutShare={}
    objPos={}
    objNeg={}
    cShare={}
    objMax =mdl.continuous_var(name="objMax")
    
    for i in range(numberOfProduct):
        for j in range(numberOfSample):
            objVar[(i,j)] = mdl.continuous_var(-10000,10000,name='objvar_'+str(i)+'_'+str(j))
            objPos[(i,j)] = mdl.continuous_var(0,10000,name='objPos_'+str(i)+'_'+str(j))
            objNeg[(i,j)] = mdl.continuous_var(0,10000,name='objNeg_'+str(i)+'_'+str(j))
            cShare[(i,j)] = mdl.continuous_var(-10000,10000,name='cShare_'+str(i)+'_'+str(j))
            cOutShare[(i,j)] = mdl.continuous_var(-10000,10000,name='cOutShare_'+str(i)+'_'+str(j))
            
    #Define Objective
    mdl.minimize(objMax)
    
    #Define Constraint
    #mdl.add_constraint(numberOfProduct * numberOfSample * objMax - mdl.sum(objPos[i,j] for i in range(numberOfProduct) for j in range(numberOfSample)) - mdl.sum(objNeg[i,j] for i in range(numberOfProduct) for j in range(numberOfSample)) == 0)
    for i in range(numberOfProduct):
        mdl.add_constraint(cOutShare[i,0] == 0)
        for j in range(numberOfSample):
            mdl.add_constraint(objVar[i,j] -  cShare [i,j] + mdl.sum(matrixA[t,i]*cOutShare[t,j] for t in range(numberOfProduct)) ==price[j,i])
            mdl.add_constraint(objVar[i,j] -  objPos [i,j] + objNeg[i,j]== 0)
            for t in range(numberOfSample):
                if share[j,i]-share[t,i]>=0:
                    mdl.add_constraint(cShare[i,j] -  cShare[i,t] >= 0)
                if outshare[j,i]-outshare[t,i]>=0:
                    mdl.add_constraint(cOutShare[i,j] - cOutShare[i,t]>= 0)

    for j in range(numberOfSample):               
        mdl.add_constraint(numberOfProduct *  objMax - mdl.sum(objPos[i,j] for i in range(numberOfProduct)) - mdl.sum(objNeg[i,j] for i in range(numberOfProduct)) == 0 )

    # Solve the model
    solution = mdl.solve()
    cShareSol=np.zeros((numberOfSample,numberOfProduct))
    cOutShareSol=np.zeros((numberOfSample,numberOfProduct))
    
    for i in range(numberOfProduct):
        for j in range(numberOfSample):
            cShareSol[j,i]=solution[cShare[(i,j)]] 
            cOutShareSol[j,i]=solution[cOutShare[(i,j)]]  
            
    for i in range(numberOfProduct):
        for j in range(numberOfSample):
            for j1 in range(numberOfSample):
                if j1>j:
                    if share[j,i]>share[j1,i]:
                        ax=share[j,i]
                        share[j,i]=share[j1,i]
                        share[j1,i]=ax;
                        ax=cShareSol[j,i]
                        cShareSol[j,i]=cShareSol[j1,i]
                        cShareSol[j1,i]=ax;
                    if outshare[j,i]>outshare[j1,i]:
                        ax=outshare[j,i]
                        outshare[j,i]=outshare[j1,i]
                        outshare[j1,i]=ax;
                        ax=cOutShareSol[j,i]
                        cOutShareSol[j,i]=cOutShareSol[j1,i]
                        cOutShareSol[j1,i]=ax; 
    return share,cShareSol,outshare,cOutShareSol

def fun_plot_share(numberOfProduct,share,cShareSol):
    for i in range(numberOfProduct):
        x=share.T[i] 
        y=cShareSol.T[i]
        plt.subplot(3,3,i+1)
        plt.scatter(x, y, alpha=0.5,color='r')
        plt.title("Proudct-"+str(i+1),fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
    return plt

def fun_plot_outshare(numberOfProduct,share,cShareSol):
    for i in range(numberOfProduct):
        x=share.T[i] 
        y=cShareSol.T[i]
        plt.subplot(3,3,i+1)
        plt.scatter(x, y, alpha=0.5,color='b')
        plt.title("Proudct-"+str(i+1),fontsize=7)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
    return plt

def fun_price_optimzation(numberOfProduct,cost,numberOfExperiments,share,outshare,vectorB,matrixA,cShareSol,cOutShareSol):
    numberOfSample=numberOfExperiments;
    # Define Model
    ps = Model('MIP') 
    
    # Define Decision Variable
    xVar={}
    yVar={}
    sigmaVar={}
    sigmaOutVar={}
    zVar={}
    zOutVar={}
    lambdaVar={}
    lambdaOutVar={}

    for i in range(numberOfProduct):
        xVar[(i)]=ps.continuous_var(share[0,i],share[numberOfSample-1,i],name='x_'+str(i))
        yVar[(i)]=ps.continuous_var(outshare[0,i],outshare[numberOfSample-1,i],name='y_'+str(i))
        sigmaVar[(i)]=ps.continuous_var(-10000,10000,name='sigma_'+str(i))
        sigmaOutVar[(i)]=ps.continuous_var(-10000,10000,name='sigmaOut_'+str(i))
        for j in range(numberOfSample):
            zVar[(i,j)]=ps.binary_var(name='z_'+str(i)+'_'+str(j))
            zOutVar[(i,j)]=ps.binary_var(name='zOut_'+str(i)+'_'+str(j))
            lambdaVar[(i,j)]=ps.continuous_var(0,1,name='lambda_'+str(i)+'_'+str(j))
            lambdaOutVar[(i,j)]=ps.continuous_var(0,1,name='lambdaOut_'+str(i)+'_'+str(j))
    
    #Define Objective
    ps.maximize(ps.sum(-cost[i]*xVar[i] -sigmaVar[i]+sigmaOutVar[i] for i in range(numberOfProduct)))

    #Define Constraint
    for i in range(numberOfProduct):
        ps.add_constraint(ps.sum(matrixA[i,t]*xVar[t] for t in range(numberOfProduct)) +yVar[i]==vectorB[i])
        ps.add_constraint(lambdaVar[i,0] -zVar[i,0]<=0)
        ps.add_constraint(lambdaOutVar[i,0] -zOutVar[i,0]<=0)
        ps.add_constraint(lambdaVar[i,numberOfSample-1] -zVar[i,numberOfSample-1]<=0)
        ps.add_constraint(lambdaOutVar[i,numberOfSample-1] -zOutVar[i,numberOfSample-1]<=0)
        ps.add_constraint(ps.sum(lambdaVar[i,t] for t in range(numberOfSample))==1)
        ps.add_constraint(ps.sum(lambdaOutVar[i,t] for t in range(numberOfSample))==1)
        ps.add_constraint(ps.sum(zVar[i,t] for t in range(numberOfSample))==1)
        ps.add_constraint(ps.sum(zOutVar[i,t] for t in range(numberOfSample))==1)
        ps.add_constraint(xVar[i]-ps.sum(share[t][i]*lambdaVar[i,t] for t in range(numberOfSample))==0)
        ps.add_constraint(yVar[i]-ps.sum(outshare[t][i]*lambdaOutVar[i,t] for t in range(numberOfSample))==0)
        ps.add_constraint(sigmaVar[i]-ps.sum(cShareSol[t][i]*share[t][i]*lambdaVar[i,t] for t in range(numberOfSample))==0)
        if vectorB[i]==1:
            ps.add_constraint(sigmaOutVar[i]-ps.sum((1-outshare[t][i])*cOutShareSol[t][i]*lambdaOutVar[i,t] for t in range(numberOfSample))==0)
        else:
            ps.add_constraint(sigmaOutVar[i]+ps.sum(cOutShareSol[t][i]*outshare[t][i]*lambdaOutVar[i,t] for t in range(numberOfSample))==0)
        for t in range(1,numberOfSample):
            ps.add_constraint(lambdaVar[i,t] -zVar[i,t-1]-zVar[i,t]<=0)
            ps.add_constraint(lambdaOutVar[i,t] -zOutVar[i,t-1]-zOutVar[i,t]<=0)
            
    sol = ps.solve()
    shareMDM=np.zeros((numberOfProduct))
    priceMDM=np.zeros((numberOfProduct))
    for i in range(numberOfProduct):
        shareMDM[i]=sol[xVar[i]] 
        priceMDM[i]=-sol[sigmaVar[i]]/sol[xVar[i]]
        for j in range(numberOfProduct):
            if vectorB[j]==1:
                priceMDM[i]=priceMDM[i]+matrixA[j,i]*sol[sigmaOutVar[j]]/(1-sol[yVar[j]])
            else:    
                priceMDM[i]=priceMDM[i]-matrixA[j,i]*sol[sigmaOutVar[j]]/(sol[yVar[j]])
        if priceMDM[i]<0:
            priceMDM[i]=0
    
    share_mdm= pd.DataFrame(shareMDM)  
    price_mdm= pd.DataFrame(priceMDM)
    profit_mdm=sol.objective_value
    return sol,share_mdm,price_mdm,profit_mdm

def fun_summary_results(numberOfProduct,price_rs,share_rs,profit_rs,price_ex,share_ex,profit_ex,price_mdm,share_mdm,profit_mdm):
    summary=np.zeros((6,numberOfProduct+1))
    for i in range(numberOfProduct):
        summary[0,i]=np.array(price_rs)[i]
        summary[1,i]=np.array(share_rs)[i]
        summary[2,i]=np.array(price_ex)[i]
        summary[3,i]=np.array(share_ex)[i]
        summary[4,i]=np.array(price_mdm)[i]
        summary[5,i]=np.array(share_mdm)[i]
    summary[0,numberOfProduct]=np.array(profit_rs)
    summary[1,numberOfProduct]=np.array(profit_rs)/np.array(profit_rs)
    summary[2,numberOfProduct]=np.array(profit_ex)
    summary[3,numberOfProduct]=np.array(profit_ex)
    summary[4,numberOfProduct]=np.array(profit_rs)
    summary[5,numberOfProduct]=np.array(profit_rs)/np.array(profit_rs)

    formater="{0:.03f}".format
    result=pd.DataFrame(summary,columns=["Fare","Luggage","No AP","Refund","Miles","Standby","Meal","Boarding","Seat"]+['Total Profit'],
                    index=['Price - Random Search','Share - Random Search',
                           "Price - Experiment Best","Share - Experiment Best",
                           'Price - MDM','Share - MDM'])
    
    return summary,result

def fun_plot_totalProfit(summary,numberOfProduct):
    num_list = [summary[0,numberOfProduct],summary[2,numberOfProduct],summary[4,numberOfProduct]]
    name_list = ['RS','EB','MDM']
    plt.bar(range(len(num_list)), num_list,tick_label=name_list)
    plt.title("Total Profit under three methods")
    return plt

def fun_show_price(summary,numberOfProduct):
    for i in range(numberOfProduct):
        num_list = [summary[0,i],summary[2,i],summary[4,i]]
        name_list = ['RS','EB','MDM']
        plt.subplot(3,3,i+1)
        plt.bar(range(len(num_list)), num_list,fc='b',tick_label=name_list)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
    #plt.title("Proudct-"+str(i+1),fontsize=7)

    plt.suptitle("Price under three method")
    return plt

def fun_show_marketshare(summary,numberOfProduct):
    for i in range(numberOfProduct):
        num_list = [summary[1,i],summary[3,i],summary[5,i]]
        name_list = ['RS','EB','MDM']
        plt.subplot(3,3,i+1)
        plt.bar(range(len(num_list)), num_list,fc='r',tick_label=name_list)
        plt.ylim(0, 1)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
    #plt.title("Proudct-"+str(i+1),fontsize=7)
    
    plt.suptitle("Market Share under three methods")
    return plt

def fun_mdm_slover(numberOfProduct,numberOfExperiments,pricePE,sharePE,vectorB,matrixA,cost):
    outsharePE=fun_cal_outShare(numberOfExperiments,numberOfProduct,sharePE)
    share,cShareSol,outshare,cOutShareSol=fun_Demand_Identification (numberOfProduct,numberOfExperiments,pricePE,sharePE,outsharePE,vectorB,matrixA)
    sol,share_mdm,price_mdm,profit_mdm=fun_price_optimzation(numberOfProduct,cost,numberOfExperiments,share,outshare,vectorB,matrixA,cShareSol,cOutShareSol)
    return share_mdm,price_mdm,profit_mdm

def fun_add_user(pricePE,sharePE,profitPE,userPrice,userShare,userProfit):
    pricePE.append(userPrice)
    sharePE.append(userShare)
    profitPE.append(userProfit)
    profit_ex=max(profitPE)
    price_ex=pricePE[profitPE.index(max(profitPE))]
    share_ex=sharePE[profitPE.index(max(profitPE))]
    numberOfExperiments=np.array(profitPE).shape[0]
    return  pricePE,sharePE,profitPE,profit_ex,price_ex,share_ex,numberOfExperiments

def fun_cal_outShare(numberOfExperiments,numberOfProduct,sharePE):
    outsharePE=np.zeros((numberOfExperiments,numberOfProduct))
    for i in range(numberOfExperiments):
        for j in range (numberOfProduct):
            if j==0:
                outsharePE[i,j]=1-np.array(sharePE)[i][j]
            else:
                outsharePE[i,j]=np.array(sharePE)[i][0]-np.array(sharePE)[i][j] 
    return outsharePE

numberOfProduct=9;
vectorB=np.array([1,0,0,0,0,0,0,0,0])
matrixA=np.array([[1,0,0,0,0,0,0,0,0],[-1,1,0,0,0,0,0,0,0],[-1,0,1,0,0,0,0,0,0],[-1,0,0,1,0,0,0,0,0],[-1,0,0,0,1,0,0,0,0],[-1,0,0,0,0,1,0,0,0],[-1,0,0,0,0,0,1,0,0],[-1,0,0,0,0,0,0,1,0],[-1,0,0,0,0,0,0,0,1]])
cost=np.array([10,10,0,10,0,5,5,0,0])
wtp=np.array([206.47,40.83,24.67,21.02,20.78,13.85,11.84,9.98,5.19])

# Data
valuation=fun_readValuation(numberOfProduct)
averageWtp,probability =fun_cal_Statistics(valuation)

# Product infor table
def product_infor():
    lst0 = ["AirFare","Free Bags","No AP","Refund","Miles","Standby","Meal","Boarding","Seat"]
    lst1 = cost
    lst2 = averageWtp
    lst3 = probability
    percentile_list = pd.DataFrame(
        {'Product':lst0,
        'Cost': lst1,
        'Average WTP': lst2,
        'Probability': lst3
        })
    return percentile_list


# Method1 : Random Search
numberOfExperimentsRS = 200
profit_rs,price_rs,share_rs=fun_Method_random_search(numberOfExperimentsRS,valuation,wtp,cost)

# Method2 : Price Experiments
numberOfExperiments = 30
pricePE,sharePE,profitPE,profit_ex,price_ex,share_ex=fun_Method_Price_Experiment(numberOfExperiments,valuation,wtp,cost)

# User Input 

def cal_function(userPrice,pricePE,sharePE,profitPE):
    #here we need user input the price
    for i in range(numberOfProduct):
        userPrice[i]=wtp[i]*(1+random.uniform(-0.5,0.5))
    userShare,userProfit=fun_cal_MarketShare_and_Profit(userPrice,cost,valuation)
    pricePE,sharePE,profitPE,profit_ex,price_ex,share_ex,numberOfExperiments=fun_add_user(pricePE,sharePE,profitPE,userPrice,userShare,userProfit)
    return userShare,userProfit,profit_ex,price_ex,share_ex,numberOfExperiments

# Histogram1

wtp=np.array([206.47,40.83,24.67,21.02,20.78,13.85,11.84,9.98,5.19])
probalility=np.array([1,0.89,0.44,0.35,0.35,0.35,0.25,0.3,0.8])
def fun_generateValuation_uniform(numberOfProduct,wtp,probalility):
    valuation=np.zeros((numberOfProduct,1000))
    for i in range(numberOfProduct):
        for j in range(1000):
            if random.uniform(0,1)<probalility[i]:
                valuation[i,j]=(0.6+0.8*random.uniform(0,1))*wtp[i]
                if valuation[i,j]<0:
                    valuation[i,j]=0; 
    return valuation

valuation=fun_generateValuation_uniform(numberOfProduct,wtp,probalility)


# Histogram2 
numberOfExperiments = 30
pricePE,sharePE,profitPE,profit_ex,price_ex,share_ex=fun_Method_Price_Experiment(numberOfExperiments,valuation,wtp,cost)
def fun_show_share_and_price(k,profitPE,sharePE):
    name_list =np.array([1,2,3,4,5,6,7,8,9]);
    for i in range (2):
        if i==0:
            num_list=pricePE[k]
            plt.subplot(4,2,i+1)
            plt.grid()
            plt.bar(range(len(num_list)), num_list,fc='b',tick_label=name_list)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=7)
            plt.title("Experiment-"+str(k+1)+"-Price",fontdict={'weight': 'normal', 'size': 7})
        if i==1:
            num_list=sharePE[k]
            plt.subplot(4,2,i+1)
            plt.grid()
            plt.bar(range(len(num_list)), num_list,fc='r',tick_label=name_list)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=7)
            plt.ylim(0,1)
            plt.title("Experiment-"+str(k+1)+"-Market Share",fontdict={'weight': 'normal', 'size': 7})
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.show()
    return plt

