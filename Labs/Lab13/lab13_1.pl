%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 3.1 Example. (for my own reference)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Direct relationships.
child(anne,bridget).
child(bridget,caroline).
child(caroline,donna).
child(donna,emily).

%% Base Case.
descend(X,Y)  :-  child(X,Y).

%% Recursive Case.
descend(X,Y)  :- child(X,Z),
                 descend(Z,Y).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Exercise 3.2 - Russian Dolls
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Direct relationships.
directlyIn(katarina, olga).
directlyIn(olga, natasha).
directlyIn(natasha, irina).

%% Base case.
in(First, Second) :- directlyIn(First, Second).

%% Recursive case.
in(First, Second) :- directlyIn(First, Third),
                        in(Third, Second).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Exercise 4.5 - Translation from German to English (TODO - finish)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Knowledge Base.
tran(eins,one).
tran(zwei,two).
tran(drei,three).
tran(vier,four).
tran(fuenf,five).
tran(sechs,six).
tran(sieben,seven).
tran(acht,eight).
tran(neun,nine).

%% Base Case.
listtran(G, E) :- tran([], []).

%% Recursive Case.
listtran([G | TG], [E | TE]) :- tran(G, E),
                                listtran(TG, TE).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Exercise 5.3 - Add 1 to each integer in the list (TODO - finish)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Base Case.
addone([], Result).

%% Recursive Case.
addone([H | T], Result) :- Result is [H + 1, T], addone(T, Result).
