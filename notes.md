# evaluate performance of similarity model 

## precision at K

- generate top K recommendations for a user
- calculate proportion of recommended items that are relevant

In your case, you could manually label a few songs as being similar to each other, and then check how many of these actually appear in your top K similar songs.

## mean avg precision at K (MAP@K)

- how many of the top K recommendations are relevant
- relevance of an item decrases if it appears lower in recommendations list
- (more robust than precision at K)

## hit rate

- frequency with which your recommendation system is able to recommend a relevant item
- split dataset into **training** and **test** sets
- train model on training set and test on testing set
- hit rate would then be the proportion of test songs for which the most similar song (according to your model) is in the same class as the test song

## *obviously* manual inspection
- alg is decent rn

___
- at the moment songs seem to be identified as similar based on varying metrics (05/24/24)
- better results with a larger dataset?

**Foals - London Thunder vs. Stay Still - Alberto L. Ferro.**

- similar key/chord progression
- different instrumentation and ??energy

**The Soft Cavalry - Dive vs. Crystal Baller - 2006 Remaster - Third Eye Blind**
- latter is more sonically charged and energetic
- more distorted
- similar key and chord progression

