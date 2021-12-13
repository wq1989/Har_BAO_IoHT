clc
clear
global X_train X_test y_train y_test Thresh
Thresh=0;
load('UCI_HAR_CNN\Train_data_fc1100')
X_train=Train_data_fc1100;
X_train(1,:)=[];
y_train=X_train(:,end);
X_train(:,257:258)=[];
load('UCI_HAR_CNN\Test_data_fc1100')
X_test=Test_data_fc1100; 
X_test(1,:)=[];
y_test=X_test(:,end);
X_test(:,257:258)=[];
load('UCI_HAR_CNN\Val_data_fc1100')
X_val=Val_data_fc1100;
X_val(1,:)=[];
y_train=[y_train; X_val(:,end);];
X_val(:,257:258)=[];
X_train=[X_train; X_val];

disp('UCI Train Test data are ready ...')
%global Thresh
fobj=@(x)Acc_SVM(x);
nP=50;
MaxIt=50;
dim=size(X_train,2)
lb=-10;
ub=10;
Thresh=0; % GBO
%
% initial accuracy of all features
Mdl = fitcecoc(X_train(:,x==1),y_train); %,'Learners',t);

labels=predict(Mdl,X_test);
actual=y_test;
pred=labels;
actual_set=unique(actual);
pred_set=unique(pred);

 if length(actual_set)==length(pred_set)
    [c_matrixp,Result]= confusion_2.getMatrix(actual,pred,0);
    %pre=100*Result.Precision;
    acc=100*Result.Accuracy;
 end
 disp(['Initial Acc. = ', num2str(acc)])
%
No_runs=10;
B_AOA_res={};B_AOA_sol={};B_AOA_curve={};
jBGWO2_res={};jBGWO2_sol={};jBGWO2_curve={};
Binary_BAT_res={};Binary_BAT_sol={};Binary_BAT_curve={};
jBPSO_res={};jBPSO_sol={};jBPSO_curve={};
Binary_SCA_res={};Binary_SCA_sol={};Binary_SCA_curve={};

 
for k=1:No_runs
    
    [res,sol,curve]=B_AOA(nP,MaxIt,lb,ub,dim,fobj); %B_AOA;
    B_AOA_res{k}=res;
    B_AOA_sol{k}=sol;
    B_AOA_curve{k}=curve;
    
    [res,sol,curve]=jBGWO2(nP,MaxIt,lb,ub,dim,fobj); %B_GWO;
    jBGWO2_res{k}=res;
    jBGWO2_sol{k}=sol;
    jBGWO2_curve{k}=curve;
    
    
    [res,sol,curve]=Binary_BAT(nP,MaxIt,lb,ub,dim,fobj); %B_BAT;
    Binary_BAT_res{k}=res;
    Binary_BAT_sol{k}=sol;
    Binary_BAT_curve{k}=curve;
    
    [res,sol,curve]=jBPSO(nP,MaxIt,lb,ub,dim,fobj); %B_PSO;
    jBPSO_res{k}=res;
    jBPSO_sol{k}=sol;
    jBPSO_curve{k}=curve;
    
    [res,sol,curve]=Binary_SCA(nP,MaxIt,lb,ub,dim,fobj); %B_SCA;
    Binary_SCA_res{k}=res;
    Binary_SCA_sol{k}=sol;
    Binary_SCA_curve{k}=curve;

end

