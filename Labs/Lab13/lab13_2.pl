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
%% Exercise 3.2 - Russian Dolls (modified to support containment hierarchy)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Direct relationships.
directlyIn(katarina, olga).
directlyIn(olga, natasha).
directlyIn(natasha, irina).

%% Base case.
in(First, Second, Trace) :- directlyIn(First, Second, Trace).

%% Recursive case.
in(First, Second, Trace) :- directlyIn(First, Third, Trace),
                        in(Third, Second, Trace).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

