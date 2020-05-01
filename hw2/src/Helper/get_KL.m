function KL1 = get_KL(edges, X, y_pred)
% The function calculates the KL divergence between 
% X and y_pred, whose distributions are binned using the specfied edges
% The code is modified from page 174 of the textbook

f=hist(X(1,:),edges)+0.01; % generate PDFs
g=hist(y_pred,edges)+0.01;

f=f/trapz(edges,f); % normalize data
g=g/trapz(edges,g); 

% compute integrand
Int1=f.*log(f./g); 

% use if needed
Int1(isinf(Int1))=0; Int1(isnan(Int1))=0;

% KL divergence
KL1=trapz(edges,Int1); 

end 