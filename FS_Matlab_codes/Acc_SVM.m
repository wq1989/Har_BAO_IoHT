function Res=Acc_SVM(x)
%we used this function to calculate the accuracy 
%global A trn vald  Thresh
global X_train X_test y_train y_test Thresh
pre=10^10;
c_matrixp=[];
Result=[];
Res=[];
% SzW=0.5;
  x=x>Thresh;
  %x=x>rand; 
  %last=x;
  x=cat(2,x,zeros(size(x,1),1));
  x=logical(x);
% 
 if sum(x)==0
     pre=inf;
      Res.pre=pre;
    Res.Result=Result;
    Res.c_matrixp=c_matrixp;
     return;
 end
%t = templateSVM('IterationLimit',1e2);
Mdl = fitcecoc(X_train(:,x==1),y_train); %,'Learners',t);


labels=predict(Mdl,X_test(:,x==1));
actual=y_test;
pred=labels;
actual_set=unique(actual);
pred_set=unique(pred);
 if length(actual_set)==length(pred_set)
    [c_matrixp,Result]= confusion_2.getMatrix(actual,pred,0);
    %pre=Result.Precision;
    pre=Result.Accuracy;
    pre=1-pre;
 end
 Res.pre=pre;
 Res.Result=Result;
 Res.c_matrixp=c_matrixp;
 
