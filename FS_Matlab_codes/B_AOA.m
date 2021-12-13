
%function [Best_FF,Best_P,Conv_curve]=AOA(N,M_Iter,LB,UB,Dim,F_obj)
function [Bestres,Best_P,Conv_curve]=B_AOA(N,M_Iter,LB,UB,Dim,F_obj)

global Thresh
disp('Binary AOA is starting ...');
%Two variables to keep the positions and the fitness value of the best-obtained solution

Bestres=[];
Best_P=zeros(1,Dim);
Best_FF=inf;
Conv_curve=zeros(1,M_Iter);
fit_curve=zeros(1,M_Iter);

%Initialize the positions of solution
%X=initialization(N,Dim,UB,LB);

X=zeros(N,Dim); 
for i=1:N
  for d=1:Dim
    if rand() > 0.5
      X(i,d)=1;
    end
  end
end

Xnew=X;
Ffun=zeros(1,size(X,1));% (fitness values)
Ffun_new=zeros(1,size(Xnew,1));% (fitness values)

MOP_Max=1;
MOP_Min=0.2;
C_Iter=1;
Alpha=5; %5;
Mu= 0.499;
Ratio=[];

for i=1:size(X,1)
    res=F_obj(X(i,:));  %Calculate the fitness values of solutions
    Ffun(1,i)=res.pre;
    if Ffun(1,i)<Best_FF
        Best_FF=Ffun(1,i);
        Best_P=X(i,:);
        Bestres=res;
    end
end 

while C_Iter<M_Iter+1  %Main loop
    
   %  if (C_Iter==30) 
    %     C_Iter=50;
    % end
    
    MOP=1-((C_Iter)^(1/Alpha)/(M_Iter)^(1/Alpha));   % Probability Ratio 
    MOA=MOP_Min+C_Iter*((MOP_Max-MOP_Min)/M_Iter); %Accelerated function
    best_to_sum=Best_FF/sum(Ffun);
    Ratio =[Ratio best_to_sum];

        %Update the Position of solutions
    for i=1:size(X,1)   % if each of the UB and LB has a just value 
        for j=1:size(X,2)
           r1=rand();
            if (size(LB,2)==1)
                if r1<MOA
                    r2=rand();
                    if r2>0.5
                        Xnew(i,j)=Best_P(1,j)/(MOP+eps)*((UB-LB)*Mu+LB);
                    else
                        Xnew(i,j)=Best_P(1,j)*MOP*((UB-LB)*Mu+LB);
                    end
                else
                    r3=rand();
                    if r3>0.5
                        Xnew(i,j)=Best_P(1,j)-MOP*((UB-LB)*Mu+LB);
                    else
                        Xnew(i,j)=Best_P(1,j)+MOP*((UB-LB)*Mu+LB);
                    end
                end               
            end
%               TF=1/(1+exp(-10*(Xnew(i,j)-0.5)));
%               %TF=abs((2/pi)*atan((pi/2)*Xnew(i,j))); 
%               if TF >= rand()
%                 Xnew(i,j)=1; 
%               else
%                 Xnew(i,j)=0; 
%               end
      end
        
%         Flag_UB=Xnew(i,:)>UB; % check if they exceed (up) the boundaries
%         Flag_LB=Xnew(i,:)<LB; % check if they exceed (down) the boundaries
%         Xnew(i,:)=(Xnew(i,:).*(~(Flag_UB+Flag_LB)))+UB.*Flag_UB+LB.*Flag_LB;
        
        for j=1:size(X,2)
              TF=1/(1+exp(-10*(Xnew(i,j)-0.5)));
              %TF=abs((2/pi)*atan((pi/2)*Xnew(i,j))); 
              if TF >= rand()
                Xnew(i,j)=1; 
              else
                Xnew(i,j)=0; 
              end  
        end

        res=F_obj(Xnew(i,:));  % calculate Fitness function 
        Ffun_new(1,i)=res.pre;
        if Ffun_new(1,i)<Ffun(1,i)
            X(i,:)=Xnew(i,:);
            Ffun(1,i)=Ffun_new(1,i);
        end
        if Ffun(1,i)<Best_FF
        Best_FF=Ffun(1,i);
        Best_P=X(i,:);
        Bestres=res;
        end
       
    end
    

    %Update the convergence curve
    Conv_curve(C_Iter)=Best_FF;
    no_feat=sum(Best_P>Thresh);
    fit_curve(C_Iter)=no_feat;
    
    %Print the best solution details after every 50 iterations
    %if mod(C_Iter,50)==0
        display(['At iteration ', num2str(C_Iter), ' the best solution fitness = ', num2str(1-Best_FF),...
            ' no of fetures= ', num2str(no_feat), ' pr=', num2str(best_to_sum)]);
    %end
     
    C_Iter=C_Iter+1;  % incremental iteration
   
end



