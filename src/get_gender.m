function gender = get_gender(subnum)
% Return the gender of the subject 
% 0 - female, 1 - male
female = [5, 15, 22, 27, 28, 32, 34, 37]; 

if any(subnum==female)
    gender = 0; 
else
    gender = 1; 
end 
