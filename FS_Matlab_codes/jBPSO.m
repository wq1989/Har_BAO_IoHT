%-------------------------------------------------------------------------%
%  Binary Particle Swarm Optimization (BPSO) source codes demo version    %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%                                                                         %

%function [sFeat,Sf,Nf,curve]=jBPSO(feat,label,N,T,c1,c2,Wmax,Wmin,Vmax)
function [BestRes,Xgb,curve]=jBPSO(N,T,lb,ub,D,fun)

disp('Starting Binary_PSO ...');
BestRes=[];
%---Inputs-----------------------------------------------------------------
% feat:  features
% label: labelling
% N:     Number of particles
% T:     Maximum number of iterations
 %c1=1.5;%    Cognitive factor
 %c2=2;%    Social factor
 Vmax=0.5;%  Maximum velocity
 Wmax=0.9;%  Maximum bound on inertia weight
 Wmin=0.4;%  Minimum bound on inertia weight
 
 % % Constriction Coefficients
phi1=2.05;
phi2=2.05;
phi=phi1+phi2;
chi=2/(phi-2+sqrt(phi^2-4*phi));
%w=chi;          % Inertia Weight
%wdamp=1;        % Inertia Weight Damping Ratio
c1=chi*phi1;    % Personal Learning Coefficient
c2=chi*phi2;    % Global Learning Coefficient
 
 
%---Outputs----------------------------------------------------------------
% sFeat: Selected features
% Sf:    Selected feature index
% Nf:    Number of selected features
% curve: Convergence curve
%--------------------------------------------------------------------------

% Objective function
%fun=@jFitnessFunction; 
% Number of dimensions
%D=size(feat,2); 
% Initial Population: Position (X) & Velocity (V)
X=zeros(N,D); V=zeros(N,D); fit=zeros(1,N);
for i=1:N
  for d=1:D
    if rand() > 0.5
      X(i,d)=1;
    end
  end
end
% Fitness 
for i=1:N
  res=fun(X(i,:)); 
  fit(i)=res.pre;
end
% PBest & GBest
[fitG,idx]=min(fit); Xgb=X(idx,:); Xpb=X; fitP=fit; 
curve=zeros(1,T);
% Pre
% curve=inf; 
t=1; 
% figure(1); clf; axis([1 100 0 0.5]); xlabel('Number of Iterations');
% ylabel('Fitness Value'); title('Convergence Curve'); grid on;
%---Iterations start-------------------------------------------------------
while t <= T
	% Inertia weight linearly decreases from 0.9 to 0.4 Eq(6)
  w=Wmax-(Wmax-Wmin)*(t/T);
  for i=1:N
    for d=1:D
      % Two random numbers in [0,1]
      r1=rand(); r2=rand();
      % Velocity update Eq(1)
      VB=V(i,d)*w+c1*r1*(Xpb(i,d)-X(i,d))+c2*r2*(Xgb(d)-X(i,d)); 
      % Limit velocity from overflying 
      VB(VB > Vmax)=Vmax; VB(VB < -Vmax)=-Vmax; V(i,d)=VB; 
      % Sigmoid function Eq(2)
      TF=1/(1+exp(-V(i,d)));
      % Position update Eq(3)
      if TF > rand()
        X(i,d)=1;
      else
        X(i,d)=0;
      end
    end
    % Fitness
    res=fun(X(i,:));
    fit(i)=res.pre;
    % Pbest update Eq(4)
    if fit(i) < fitP(i)
      Xpb(i,:)=X(i,:); fitP(i)=fit(i);
    end
    % Gbest update Eq(5)
    if fitP(i) < fitG
      Xgb=Xpb(i,:); fitG=fitP(i);
      BestRes=res;
    end
  end
  curve(t)=fitG;  
  no_feat=sum(Xgb==1);
  disp(['Iteration ' num2str(t) ': fitness = ' num2str(1-fitG), '  No of features = ' num2str(no_feat)]);
  % Plot convergence curve
%   pause(0.000000001); hold on;
%   CG=plot(t,fitG,'Color','r','Marker','.'); set(CG,'MarkerSize',5);
  t=t+1;
  
%  if bolean_b==1 && last_iter<T
%       BestCost(last_iter:T)=fitG;
%       break;
%       t=T+1;
%  end
  
end
% Select features 
% Pos=1:D; Sf=Pos(Xgb==1); Nf=length(Sf); sFeat=feat(:,Sf); 
end



