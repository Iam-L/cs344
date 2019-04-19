%% Prolog Program - Burning the Witch (Monty Python)
%% Resource Used: http://www.cs.swan.ac.uk/~csneal/SystemSpec/Different.html

%% If the person is a witch, she is combustible.
witch(Person) :- combustible(Person).
%% If she is combustible, she must be made of wood.
combustible(Person) :- madeOfWood(Person).
%% If she is made of wood, she must float.
madeOfWood(Person) :- objectFloats(Person).
%% If she floats, then she must weigh the same as a duck.
objectFloats(Person) :- sameWeight(Person, duck).

%% She has the same weight as a duck.
sameWeight(girl, duck).

%% Therefore, she is a witch!