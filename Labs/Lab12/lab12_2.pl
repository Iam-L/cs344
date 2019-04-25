%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Exercise  2.2 We are working with the following knowledge base:

house_elf(dobby).
witch(hermione).
witch('McGonagall').
wizard(gandalf).
witch(rita_skeeter).
magic(X) :- house_elf(X).
magic(X) :- wizard(X).
magic(X) :- witch(X).