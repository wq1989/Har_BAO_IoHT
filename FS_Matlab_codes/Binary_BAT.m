% Modified source code :

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  BBA source codes version 1.1                                     %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili, S. M. Mirjalili, X. Yang              %
%               Binary Bat Algorithm, Neural Computing and          %
%               Application, in press,                              %
%               DOI: 10.1007/s00521-013-1525-5                      %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Main source code :
% ======================================================== % 
% Files of the Matlab programs included in the book:       %
% Xin-She Yang, Nature-Inspired Metaheuristic Algorithms,  %
% Second Edition, Luniver Press, (2010).   www.luniver.com %
% ======================================================== %    

% -------------------------------------------------------- %
% Bat-inspired algorithm for continuous optimization (demo)%
% Programmed by Xin-She Yang @Cambridge University 2010    %
% -------------------------------------------------------- %
% Usage: bat_algorithm([Particle_No Loudness Pulse_rate Variable_no Max_iteration Costfunction]);                     %

%function [best,fmin,cg_curve]=Binary_BAT(n, A, r, d, Max_iter, CostFunction)
function [BestRes,best,cg_curve]=Binary_BAT(n,Max_iter,lb,ub,d,CostFunction)
disp('Binary BAT starts ...')
% Display help
 %help bat_algorithm.m
BestRes=[];
%n is the population size, typically 10 to 25
A=0.5; %rand(n,1);  %is the loudness  (constant or decreasing)
r=0.5; %rand(n,1);  %is the pulse rate (constant or decreasing)
%d is the dimension of the search variables
%Max_iter is the maximum number of iteration

% alpha=0.5;              %constant for loudness update
% gamma=0.5;              %constant for emission rate update
% ro=0.001;                 %initial pulse emission rate

% This frequency range determines the scalings
Qmin=0;         % Frequency minimum
Qmax=2;         % Frequency maximum
% Iteration parameters
N_iter=0;       % Total number of function evaluations


% Initial arrays
Q=zeros(n,1);   % Frequency
v=zeros(n,d);   % Velocities
Sol=zeros(n,d);
cg_curve=zeros(1,Max_iter);
fit_curve=zeros(1,Max_iter);
% Initialize the population/solutions
for i=1:n,
    for j=1:d % For dimension
        if rand<=0.5
            Sol(i,j)=0;
        else
            Sol(i,j)=1;
        end
    end
end

for i=1:n
    res=CostFunction(Sol(i,:));
    Fitness(i)=res.pre;
end
% Find the current best
[fmin,I]=min(Fitness);
best=Sol(I,:);

% ======================================================  %
% Note: As this is a demo, here we did not implement the  %
% reduction of loudness and increase of emission rates.   %
% Interested readers can do some parametric studies       %
% and also implementation various changes of A and r etc  %
% ======================================================  %

% Start the iterations -- Binary Bat Algorithm
while (N_iter<Max_iter)
     
    if (N_iter==30) 
        N_iter=50;
    end  
        % Loop over all bats/solutions
        N_iter=N_iter+1;
        cg_curve(N_iter)=fmin;
        for i=1:n,
            for j=1:d
                Q(i)=Qmin+(Qmin-Qmax)*rand; % Equation 3 in the paper
                v(i,j)=v(i,j)+(Sol(i,j)-best(j))*Q(i); % Equation 1 in the paper

                V_shaped_transfer_function=abs((2/pi)*atan((pi/2)*v(i,j))); % Equation 9 in the paper
                
                if rand<V_shaped_transfer_function % Equation 10 in the paper
                    Sol(i,j)=~Sol(i,j);
                else
                    Sol(i,j)=Sol(i,j);
                end
                
                if rand>r  % Pulse rate
                      Sol(i,j)=best(j);
                end   
               
            end       
            
           res=CostFunction(Sol(i,:)); % Evaluate new solutions
           Fnew=res.pre;
     
           if (Fnew<=Fitness(i)) && (rand<A)  % If the solution improves or not too loudness
                Sol(i,:)=Sol(i,:);
                Fitness(i)=Fnew;
           end

          % Update the current best
          if Fnew<=fmin,
                best=Sol(i,:);
                fmin=Fnew;
                BestRes=res;
          end
        end
        fit_curve(N_iter)=sum(best);
        % Output/display
disp(['Number of evaluations: ',num2str(N_iter),'  fmin = ',num2str(1-fmin), ' no_features = ', num2str(sum(best))]);
     
end


