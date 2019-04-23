

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Exercise 1.4 Represent the following in Prolog:
killer(butch).
married(mia, marsellus).
dead(zed).
killsPerson(marsellus):- footMassage(_Person, mia).
lovesPerson(mia):- goodDancer(_Person).
eatsItem(jules):- nutritiousItem(Item), tastyItem(Item).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Exercise  1.5 Suppose we are working with the following knowledge base:
wizard(ron).
wizard(X):-  hasBroom(X),  hasWand(X).
hasWand(harry).
quidditchPlayer(harry).
hasBroom(X):-  quidditchPlayer(X).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
