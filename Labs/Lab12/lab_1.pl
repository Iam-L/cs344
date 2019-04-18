killer(butch).
married(mia, marsellus).
dead(zed).
killsPerson(marsellus):- footMassage(_Person, mia).
lovesPerson(mia):- goodDancer(_Person).
eatsItem(jules):- nutritiousItem(Item), tastyItem(Item).






wizard(ron).
wizard(X):-  hasBroom(X),  hasWand(X).
hasWand(harry).
quidditchPlayer(harry).
hasBroom(X):-  quidditchPlayer(X).