%% Prolog Program - Burning the Witch (Monty Python)
%% Resource Used: URL: http://www.cs.swan.ac.uk/~csneal/SystemSpec/Different.html


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Witches Burn. This one is fair enough - though the villagers suggest trying to actually burn her as way of
%% testing this.

%% Wood Burns. Hence witches are made of wood. How do you check that she is made of wood? Try building a bridge
%% out of her, one suggests - but Bedevere points out that you can also make bridges from stone.

%% Wood Floats. Bedevere gently leads them to this point, and asks them if they know anything else that floats.

%% Ducks Float. They actually have a lot of trouble thinking of something else that floats - and it is Arthur, who has
%% just arrived on the scene, who says: 'A Duck!' (stunned amazement and dramatic music.)

%% Therefore... The logic goes: that if she weighs the same as a duck, she's a witch and they can burn her. So they put
%% her on a set of scales with a duck, and of course she does weigh the same ('it's a fair cop').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% If the person is a witch, she is combustible.
witch(Person) :- combustible(Person).

%% If she is combustible, she must be made of wood.
combustible(Person) :- madeOfWood(Person).

%% If she is made of wood, she must float.
madeOfWood(Person) :- objectFloats(Person).

%% If she floats, then she must weigh the same as a duck.
objectFloats(Person) :- sameWeight(Person, duck).

%% She has the same weight as a duck (known "fact").
sameWeight(girl, duck).

%% Therefore, she is a witch!
%% And we burn her at the stake or kill her in some other terrible agonizing horrendeous way.
