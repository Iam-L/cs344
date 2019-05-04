%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Part 2 - Consider the following situation in the blocks world:

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