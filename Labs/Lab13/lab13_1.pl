%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Section 3.1 Example. (for my own reference)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Knowledge Base.
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

%% Knowledge Base.
directlyIn(katarina, olga).
directlyIn(olga, natasha).
directlyIn(natasha, irina).

%% Base case.
in(First, Second) :- directlyIn(First, Second).

%% Recursive case.
in(First, Second) :-

        directlyIn(First, Third),
        in(Third, Second).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Exercise 4.5 - Translation from German to English
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
listtran([], []).

%% Recursive Case.
listtran([German | TailGerman], [English | TailEnglish]) :-

        tran(German, English),
        listtran(TailGerman, TailEnglish).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Exercise 5.3 - Add 1 to each integer in the list.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Base Case.
addone([], []).

%% Recursive Case.
addone([Head | Tail], [ResultHead | ResultTail]) :-

        ResultHead is Head + 1,
        addone(Tail, ResultTail).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
