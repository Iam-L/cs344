%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 1 - Do LPN! exercise 2.4 (i.e., the crossword problem).

%% Disable singleton warnings.
:- style_check(-singleton).

%% Knowledge base (lexicon).
word(astante,  a,s,t,a,n,t,e).
word(astoria,  a,s,t,o,r,i,a).
word(baratto,  b,a,r,a,t,t,o).
word(cobalto,  c,o,b,a,l,t,o).
word(pistola,  p,i,s,t,o,l,a).
word(statale,  s,t,a,t,a,l,e).

crossword(V1, V2, V3, H1, H2, H3) :-
                                        word(V1, V11, V12, V13, V14, V15, V16, V17),
                                        word(V2, V21, V22, V23, V24, V25, V26, V27),
                                        word(V3, V31, V32, V33, V34, V35, V36, V37),
                                        word(H1, H11, V12, H13, V22, H15, V32, H17),
                                        word(H2, H21, V14, H23, V24, H25, V34, H27),
                                        word(H3, H31, V16, H33, V26, H35, V36, H37).

% Note: Account for all shared positions to ensure the letters match in those positions.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 2 - Consider the following situation in the blocks world:
 %% FIXME - not working doing it this way.

%% Knowledge Base.
:- discontiguous isOn/2. %% Disable warning.

%% A is on B.
isOn('A', 'B').

%% B is on C.
isOn('B', 'C').

%% The table supports C.
supports('Table', 'C').

%% For any two entities, if the first entity supports the second, then the second is on the first.
isOn(E2, E1) :- supports(E1, E2).


%% For any two entities, if the first entity is on the second, then the first is “above” the second.
above(E1, E2) :- isOn(E1, E2).


%% For any three entities, if the first entity is above the second which is above the third,
%% then the first is above the third.
above(E1, E3) :-
                    above(E1, E2),
                    above(E2, E3).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Part 3 - Write a recursive implementation of Euclid’s algorithm for computing the greatest common divisor (GCD) of
%% two integers.

%% Resources:
%% URL: https://www.khanacademy.org/computing/computer-science/cryptography/modarithmetic/a/the-euclidean-algorithm

%% Knowledge Base.
gcd(0, 0, Result) :- Result = 0.
gcd(0, Y, Result) :- Result = Y.
gcd(X, 0, Result) :- Result = X.
gcd(X, X, Result) :- Result = X.
gcd(Y, Y, Result) :- Result = Y.

%% Base Case.
gcd(X, Y, Result) :-

                    X =:= Y,
                    Result is X.

%% Recursive Case.
gcd(X, Y, Result) :-

                    X > Y,
                    Math is X - Y,
                    gcd(Math, Y, Result).

gcd(X, Y, Result) :-

                    X < Y,
                    Math is Y - X,
                    gcd(X, Math, Result).

%% Note to self: Use different "intermediate" variable for arithmetic; separate from 3rd parameter variable.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
