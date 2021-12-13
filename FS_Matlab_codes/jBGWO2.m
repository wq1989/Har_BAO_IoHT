%-------------------------------------------------------------------------%
%  Binary Grey Wolf Optimization (BGWO) source codes demo version         %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

%function [sFeat,Sf,Nf,curve]=jBGWO2(feat,label,N,T)
function [Bestres,Xalpha,curve,fit_curve]=jBGWO2(N,T,lb,ub,D,fun)
disp('Binary GWO starts ...')

%---Inputs-----------------------------------------------------------------
% feat:  features
% label: labelling
% N:     Number of wolves
% T:     Maximum number of iterations
%---Outputs----------------------------------------------------------------
% sFeat: Selected features
% Sf:    Selected feature index
% Nf:    Number of selected features
% curve: Convergence curve
%--------------------------------------------------------------------------

Bestres=[];
%fun=@jFitnessFunction; 
%D=size(feat,2); 
X=zeros(N,D); 
for i=1:N
  for d=1:D
    if rand() > 0.5
      X(i,d)=1;
    end
  end
end
%initfit=fun(init_best);
fit=zeros(1,N);
for i=1:N
  res=fun(X(i,:));
  fit(i)=res.pre;
end
[~,idx]=sort(fit,'ascend');  
Xalpha=X(idx(1),:); Xbeta=X(idx(2),:); Xdelta=X(idx(3),:);
Falpha=fit(idx(1)); Fbeta=fit(idx(2)); Fdelta=fit(idx(3));
curve= zeros(1,T);
fit_curve=zeros(1,T);
t=1;
Ratio=[];
%figure(1); clf; axis([1 100 0 0.2]); xlabel('Number of Iterations');
%ylabel('Fitness Value'); title('Convergence Curve'); grid on;
%---Iterations start-------------------------------------------------------
while t <= T
    
    if (t==30) 
        t=50;
    end
    
  a=2-2*(t/T); 
  for i=1:N
    for d=1:D
      C1=2*rand(); C2=2*rand(); C3=2*rand();
      Dalpha=abs(C1*Xalpha(d)-X(i,d)); 
      Dbeta=abs(C2*Xbeta(d)-X(i,d));
      Ddelta=abs(C3*Xdelta(d)-X(i,d));
      A1=2*a*rand()-a; A2=2*a*rand()-a; A3=2*a*rand()-a;
      X1=Xalpha(d)-A1*Dalpha; X2=Xbeta(d)-A2*Dbeta; 
      X3=Xdelta(d)-A3*Ddelta;
      Xn=(X1+X2+X3)/3; 
      TF=1/(1+exp(-10*(Xn-0.5)));
      if TF >= rand()
        X(i,d)=1; 
      else
        X(i,d)=0; 
      end
    end
  end
  for i=1:N
    res=fun(X(i,:)); 
    fit(i)=res.pre;
    if fit(i) < Falpha
      Falpha=fit(i); Xalpha=X(i,:);
      Bestres=res;
    end
    if fit(i) < Fbeta && fit(i) > Falpha
      Fbeta=fit(i); Xbeta=X(i,:);
    end
    if fit(i) < Fdelta && fit(i) > Falpha && fit(i) > Fbeta
      Fdelta=fit(i); Xdelta=X(i,:);
    end
  end
%   if Falpha < initfit
%       bestfmin=Falpha;
%       Ratio=[Ratio bestfmin/mean(fit)];
%   else
%       bestfmin=initfit;
%       fit2=fit;
%       idx=find(fit==Falpha);
%       if ~isempty(idx)
%         fit2(idx)=bestfmin;
%       else
%           fit2(1)=bestfmin;
%       end
%       Ratio=[Ratio bestfmin/mean(fit2)];
%   end
%   curve(t)=bestfmin; 
  curve(t)=Falpha;
  fit_curve(t)=sum(Xalpha);
  display(['At iteration ', num2str(t), ' fitness ', num2str(1-Falpha), '  No of features = ' num2str(sum(Xalpha))]);
  
 % pause(1e-20); hold on;
 % CG=plot(t,Falpha,'Color','r','Marker','.'); set(CG,'MarkerSize',5);
  t=t+1;
  
end
%Pos=1:D; Sf=Pos(Xalpha==1); Nf=length(Sf); sFeat=feat(:,Sf);
end



