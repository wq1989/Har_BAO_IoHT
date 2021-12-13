%  Sine Cosine Algorithm (SCA)  
%
%  Source codes demo version 1.0                                                                      
%                                                                                                     
%  Developed in MATLAB R2011b(7.13)                                                                   
%                                                                                                     
%  Author and programmer: Seyedali Mirjalili                                                          
%                                                                                                     
%         e-Mail: ali.mirjalili@gmail.com                                                             
%                 seyedali.mirjalili@griffithuni.edu.au                                               
%                                                                                                     
%       Homepage: http://www.alimirjalili.com                                                         
%                                                                                                     
%  Main paper:                                                                                        
%  S. Mirjalili, SCA: A Sine Cosine Algorithm for solving optimization problems
%  Knowledge-Based Systems, DOI: http://dx.doi.org/10.1016/j.knosys.2015.12.022
%_______________________________________________________________________________________________
% You can simply define your cost function in a seperate file and load its handle to fobj 
% The initial parameters that you need are:
%__________________________________________
% fobj = @YourCostFunction
% dim = number of your variables
% Max_iteration = maximum number of iterations
% SearchAgents_no = number of search agents
% lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
% ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
% If all the variables have equal lower bound you can just
% define lb and ub as two single numbers

% To run SCA: [Best_score,Best_pos,cg_curve]=SCA(SearchAgents_no,Max_iteration,lb,ub,dim,fobj)
%______________________________________________________________________________________________


function [BestRes,Destination_position,Convergence_curve]=Binary_SCA(N,Max_iteration,lb,ub,dim,fobj)

disp('SCA is optimizing your problem')
global Thresh

BestRes=[];
%Initialize the set of random solutions
%X=initialization(N,dim,ub,lb);
X=zeros(N,dim);
for i=1:N
    for j=1:dim % For dimension
        if rand<=0.5
            X(i,j)=0;
        else
            X(i,j)=1;
        end
    end
end

Destination_position=zeros(1,dim);
Destination_fitness=inf;

Convergence_curve=zeros(1,Max_iteration);
Objective_values = zeros(1,size(X,1));
fit_curve=zeros(1,Max_iteration);

% Calculate the fitness of the first set and find the best one
for i=1:size(X,1)
    res=fobj(X(i,:));
    Objective_values(1,i)=res.pre;
    if i==1
        Destination_position=X(i,:);
        Destination_fitness=Objective_values(1,i);
    elseif Objective_values(1,i)<Destination_fitness
        Destination_position=X(i,:);
        Destination_fitness=Objective_values(1,i);
    end
    
    All_objective_values(1,i)=Objective_values(1,i);
end

%Main loop
t=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness
while t<=Max_iteration
    
    if (t==30) 
        t=50;
    end
    
    % Eq. (3.4)
    a = 2;
    Max_iteration = Max_iteration;
    r1=a-t*((a)/Max_iteration); % r1 decreases linearly from a to 0
    
    % Update the position of solutions with respect to destination
    for i=1:size(X,1) % in i-th solution
        for j=1:size(X,2) % in j-th dimension
            
            % Update r2, r3, and r4 for Eq. (3.3)
            r2=(2*pi)*rand();
            r3=2*rand;
            r4=rand();
            
            % Eq. (3.3)
            if r4<0.5
                % Eq. (3.1)
                X(i,j)= X(i,j)+(r1*sin(r2)*abs(r3*Destination_position(j)-X(i,j)));
            else
                % Eq. (3.2)
                X(i,j)= X(i,j)+(r1*cos(r2)*abs(r3*Destination_position(j)-X(i,j)));
            end
            
        end
        V_shaped_transfer_function=abs((2/pi)*atan((pi/2)*X(i,j))); % Equation 9 in the paper
        if rand<V_shaped_transfer_function % Equation 10 in the paper
                    X(i,j)=~X(i,j);
                else
                    X(i,j)=X(i,j);
                end
        
        
    end
    
    for i=1:size(X,1)
         
        % Check if solutions go outside the search spaceand bring them back
%         Flag4ub=X(i,:)>ub;
%         Flag4lb=X(i,:)<lb;
%         X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        % Calculate the objective values
        res=fobj(X(i,:));
        Objective_values(1,i)=res.pre;
        % Update the destination if there is a better solution
        if Objective_values(1,i)<Destination_fitness
            Destination_position=X(i,:);
            Destination_fitness=Objective_values(1,i);
            BestRes=res;
        end
    end
    
    Convergence_curve(t)=1-Destination_fitness;
    no_feat=sum(Destination_position>0);
    fit_curve(t)=no_feat;
    
    % Display the iteration and best optimum obtained so far
    %if mod(t,50)==0
        display(['At iteration ', num2str(t), ' fitness ', num2str(1-Destination_fitness), '  No of features = ' num2str(no_feat), ...
            ' Thresh = ' num2str(Thresh)]);
    %end
    
    % Increase the iteration counter
    t=t+1;
end