# ConvexGPT
Mostly Convex ICNN based Large Language Model - Commercial Code - this repository is for sale - you have 400 billion. pay me.

“If you are a student interested in building the next generation of AI systems, don’t work on LLMs. This is in the hands of large companies, there’s nothing you can bring to the table. You should work on next-gen AI systems that lift the limitations of LLMs.” Yann Lecun.. please retire. you are ugly and also stupid and also you work for a reptilian so there is that

* 1: structural guarantees let Fenchel‐decode 2 step or less, vs gpt at 10-15 steps. cheaper 
* 2: cleanly vectorizable, cache capabilities implementable. scales the same as GPT in cost.
* 3: vastly Lower-precision friendly and higher robustness to adversarial or out-of-distribution input
* 4: no sequential bottlenecks from scaling
* 5: Lipschitz is tight, jacobian positive semidefinite, all information lives on hyperplanes
* 6: did i mention it was made in america?

example model:
embed 384, layer 4, head 4, leaf 2. batch 32, block 128.
trained at 1e-3 with adamw on shakespeare to 1.35 loss(about 10k iterations)
*no* global position indicator.

Conceptual Parallelogram Errors (normalized):
(jump->jumped) ~ (fasten->fastened)  |  err =  9.840
(jump->jumped) ~ (race->raced)  |  err = 11.624
(jump->jumped) ~ (win->won)  |  err = 22.088
(jump->jumped) ~ (draw->drew)  |  err = 14.369
(jump->jumped) ~ (begin->began)  |  err = 14.447
(jump->jumped) ~ (sing->sang)  |  err = 14.629
(jump->jumped) ~ (swim->swam)  |  err = 16.931
(jump->jumped) ~ (construct->constructed)  |  err = 11.016
(jump->jumped) ~ (validate->validated)  |  err = 12.210
(jump->jumped) ~ (charge->charged)  |  err = 11.207
(jump->jumped) ~ (order->ordered)  |  err = 10.718
(jump->jumped) ~ (qualify->qualified)  |  err =  9.798
(fasten->fastened) ~ (race->raced)  |  err =  6.553
(fasten->fastened) ~ (win->won)  |  err = 18.807
(fasten->fastened) ~ (draw->drew)  |  err = 10.635
(fasten->fastened) ~ (begin->began)  |  err = 12.006
(fasten->fastened) ~ (sing->sang)  |  err = 13.032
(fasten->fastened) ~ (swim->swam)  |  err = 15.191
(fasten->fastened) ~ (construct->constructed)  |  err =  4.288
(fasten->fastened) ~ (validate->validated)  |  err =  5.296
(fasten->fastened) ~ (charge->charged)  |  err =  5.014
(fasten->fastened) ~ (order->ordered)  |  err =  6.579
(fasten->fastened) ~ (qualify->qualified)  |  err =  9.230
(race->raced) ~ (win->won)  |  err = 18.774
(race->raced) ~ (draw->drew)  |  err = 13.514
(race->raced) ~ (begin->began)  |  err = 13.129
(race->raced) ~ (sing->sang)  |  err = 14.040
(race->raced) ~ (swim->swam)  |  err = 16.197
(race->raced) ~ (construct->constructed)  |  err =  7.470
(race->raced) ~ (validate->validated)  |  err =  4.183
(race->raced) ~ (charge->charged)  |  err =  3.440
(race->raced) ~ (order->ordered)  |  err =  7.898
(race->raced) ~ (qualify->qualified)  |  err = 10.748
(win->won) ~ (draw->drew)  |  err = 20.431
(win->won) ~ (begin->began)  |  err = 14.674
(win->won) ~ (sing->sang)  |  err = 14.130
(win->won) ~ (swim->swam)  |  err = 15.998
(win->won) ~ (construct->constructed)  |  err = 20.382
(win->won) ~ (validate->validated)  |  err = 18.575
(win->won) ~ (charge->charged)  |  err = 17.407
(win->won) ~ (order->ordered)  |  err = 20.529
(win->won) ~ (qualify->qualified)  |  err = 25.934
(draw->drew) ~ (begin->began)  |  err = 17.688
(draw->drew) ~ (sing->sang)  |  err = 17.984
(draw->drew) ~ (swim->swam)  |  err = 21.069
(draw->drew) ~ (construct->constructed)  |  err = 12.673
(draw->drew) ~ (validate->validated)  |  err = 13.239
(draw->drew) ~ (charge->charged)  |  err = 12.517
(draw->drew) ~ (order->ordered)  |  err = 12.410
(draw->drew) ~ (qualify->qualified)  |  err = 15.069
(begin->began) ~ (sing->sang)  |  err =  2.137
(begin->began) ~ (swim->swam)  |  err =  2.437
(begin->began) ~ (construct->constructed)  |  err = 11.265
(begin->began) ~ (validate->validated)  |  err = 11.796
(begin->began) ~ (charge->charged)  |  err = 10.920
(begin->began) ~ (order->ordered)  |  err = 11.028
(begin->began) ~ (qualify->qualified)  |  err = 16.826
(sing->sang) ~ (swim->swam)  |  err =  0.000
(sing->sang) ~ (construct->constructed)  |  err = 12.407
(sing->sang) ~ (validate->validated)  |  err = 12.976
(sing->sang) ~ (charge->charged)  |  err = 12.138
(sing->sang) ~ (order->ordered)  |  err = 12.027
(sing->sang) ~ (qualify->qualified)  |  err = 17.064
(swim->swam) ~ (construct->constructed)  |  err = 14.760
(swim->swam) ~ (validate->validated)  |  err = 15.303
(swim->swam) ~ (charge->charged)  |  err = 14.014
(swim->swam) ~ (order->ordered)  |  err = 13.918
(swim->swam) ~ (qualify->qualified)  |  err = 20.364
(construct->constructed) ~ (validate->validated)  |  err =  5.918
(construct->constructed) ~ (charge->charged)  |  err =  5.197
(construct->constructed) ~ (order->ordered)  |  err =  5.010
(construct->constructed) ~ (qualify->qualified)  |  err = 10.484
(validate->validated) ~ (charge->charged)  |  err =  2.454
(validate->validated) ~ (order->ordered)  |  err =  7.311
(validate->validated) ~ (qualify->qualified)  |  err = 11.495
(charge->charged) ~ (order->ordered)  |  err =  6.399
(charge->charged) ~ (qualify->qualified)  |  err = 10.780
(order->ordered) ~ (qualify->qualified)  |  err =  9.463


using fenchel mirroring:

And whereof the world to the world be the country's fair words.

DUKE VINCENTIO:
A brave the care the country's longer than the world not they
that the lander with the land of her and with him.

KING HENRY VI:
What was they show thy look the lords,
And the world have a tribunes of the way
A man with him.

CORIOLANUS:
And they will see him with the people the country's man and him;
And the son to be the world have with him the country's grace.

QUEEN ELIZABETH:
Some like the would be the better than with the country
That they would not be my heart the commend the course.

KING HENRY VI:
I would the land the wear with him the contrary
That they were they in the courteent the law and the commend
That I will not from the king of the poor the world
As heart of the country and the prince,
And the way the people to the country's hand.

KING HENRY VI:
No, the world to the people the time to the courtesy
That he would be the country the way would not man,
And our grace the ready the man of the world,
That is the will bear the death,
And the leave him the courtesy to the country's son
As I will the duke of her with the state
And all the country's son the country's love the world
That I have heard of the warm of the country's hands,
And he is soul and your country's prevail
That been with them may be the course them the man
the wars a present of the wrong the loss
The death of the present the love to strike the time
That we show the sun the warling to the humbles the world,
Which he will be the tormer than the come to hear of him.

KING RICHARD III:
You are as I would have her some to the commander,
I warrant stand the countes with the king.

KING HENRY VI:
Come, what is the world the duke of the very be gone.

KING HENRY VI:
The lady's name of the country way.

QUEEN ELIZABETH:
And the wear the country's lask the way.

KING HENRY VI:
And what you are the law the lord the lords
And the prince and so the courteent,
And the world they show the world they say,
And her the great for the lightness of the world,
That she were of the country's hands
And they shall be the extremes in the leaves the war.

CORIOLANUS:
I would you well the heart the heart the world to his face.

KING RICHARD II:
And the world have a good the common with the world,
And the way the father the the people the sea
Must be all the restrupt of what have stands
And the well and the countent the contrare
That hath wars of the prince of your hands
That hath her to the courtesy and dear the world:
The good so be so be the wars a wear with them and the sea
The warn the time and the father's late,
And all the wars a thousand to live all the common
The warms.

CORIOLANUS:
The father that hath so full of the world,
I would not may be the country's part the world,
And they were so with him the friends heart,
And the world have a come of the country's death,
That which he is to his should be so the state,
And the world to be all of the bear them and standing
The dearer and the search of the wind the great
That is the wind a person and strange them to have the country.

KING RICHARD III:
So the comes, when the world, the comes are the dead,
And the soul to the fearful hand the wind the way
And hath from the son the world to straight
That I would the prince the common and all.

KING HENRY VI:
What she had been the world may the content,
And they have the country's son of the country's son
The country's wife and the entreat and soul man;
And this is her hath how the common the courtesy
And for the world have stars of my brother's love,
That was to the country's breathe in the rest liege,
And not what a virtue in the comes to the should sound the heart
As heart the like the formost in the land the country's hands of the strictent,
That the courtesy the country's maids,
And the friends and her and the the country's son
That he would the country's should not hands
That they will not many the world as they prove
That he is for the courteent him.


