# Basic NLP on the Texts of Harry Potter: Topic Modeling with Latent Dirichlet Allocation


![House Ties](images/rhii-photography-767018-unsplash.jpg)
```"Hmm," said a small voice in his ear. "Difficult. Very difficult. Plenty of courage, I see. Not a bad mind either. There's talent, oh my goodness, yes - and a nice thirst to prove yourself, now that's interesting. . . . So where shall I put you?"```



Recently, I was put on a new project with a team who were unanimously shocked and disappointed that I'd never read nor seen the movies about a certain fictional wizard named Harry Potter. In order to fit in with the team, and obviously save my career from an early end, it quickly became evident that I was going to have to take a crash-course in the happenings at Hogwarts. Armed with my ebook reader and seven shiny new pdf files, I settled in to see what the fuss was all about. Meanwhile, I had started working on a side project composed of a bunch of unrelated NLP tasks. I needed a good sized set of text documents and I thought all of these shiny new pdfs would be a great source.
And somewhere around the middle of the third book,  I suddenly realized that LDA was basically just an algorithmic Sorting Hat.
LDA, or Latent Dirichlet Allocation, is a generative probabilistic model of (in NLP terms) a corpus of documents made up of words and/or phrases. The model consists of two tables; the first table is the probability of selecting a particular word in the corpus when sampling from a particular topic, and the second table is the probability of selecting a particular topic when sampling from a particular document.


---

Here's an example. Let's say I've got these three (rather non-sensical) documents:

```
document_0 = "Harry Harry wand Harry magic wand"
document_1 = "Hermione robe Hermione robe magic Hermione"
document_2 = "Malfoy spell Malfoy magic spell Malfoy"
document_3 = "Harry Harry Hermione Hermione Malfoy Malfoy"
```

Here's the term-frequency matrix for these documents:


Just from glancing at this, it seems pretty obvious that document 0 is mostly about Harry, a little bit about magic, and partly about wand. Document 1 is also a little bit about magic, but mostly about Hermione and robe. And document 2 is again partly about magic, but mostly about Malfoy and spell. Document 3 is equally about Harry, Hermione, and Malfoy. It's usually not so easy to see this because a more practical corpus would consists of thousands or ten-of-thousands of words, so let's see what the LDA algorithm chooses for topics:
Data Science with Excel!And that's roughly what we predicted just by going with term frequencies and our gut. The number of topics is a hyperparameter you'll need to choose and tune carefully, and I'll go into that later, but for this example I chose 4 topics to make my point. The upper table shows words versus topics and the lower table shows documents versus topics. Each column in the upper table and each row in the lower table must sum to 1. These tables are telling us that if we were to randomly sample a word from Topic 0, there's a 70.9% chance we'd grab "Harry". If we chose a word from Topic 3, it's near certain that we'd pick "magic". If we sampled Document 3, there's an equal chance that we would pick Topic 0, 1, or 2.
It's up to us as smart humans who can infer meaning from a bag of words to name these topics. In these examples with a very limited vocabulary, the topics quite obviously correspond to single words. If we had run LDA on, say, a couple thousand restaurant descriptions, we might find topics corresponding to cuisine type or atmosphere. It's important to note that LDA, unlike typical clustering algorithms such as Kmeans, allows a document to exist in multiple topics. So in those restaurant descriptions, we might find one restaurant placed in the "Italian", "date-night", and "cafe" topics.


---

So how is all of this like the Sorting Hat? All new students at Hogwarts go through a ceremony when they arrive on day one to determine which house they'll be in (I'm probably the only person who didn't know this up until a few weeks ago). The Sorting Hat, once placed on someone's head, understands what is in their thoughts, dreams, and experiences. This is a bit like LDA building the term-frequency matrix and understanding what words and N-grams are contained within each document.
The Sorting Hat then compares the student's attributes with the attributes of the various houses (bravery goes to Gryffindor, loyalty to Hufflepuff, wisdom to Ravenclaw, and sneaky, shifty sleazeballs go to Slytherin (ok, just a quick aside - can ANYONE explain to me why Slytherin has persisted for the thousand-year history of this school? It's like that one fraternity which finds itself in yet another ridiculously obscene scandal every damn year!)). This is where LDA creates the word-topic table and begins to associate the attributes of the topics.
Harry's placement was notably split between Gryffindor and Slytherin due to his combination of courage, intelligence, talent, and ambition, but Gryffindor just slightly edged out ahead and Harry Potter became the hero of an entire generation of young millennials instead of its villain. This is where LDA creates the document-topic table and finally determines which is the dominant topic for each document.


---

OK, so now that we know roughly what LDA does, let's look at two different implementations in Python. Check out my Github repo for all of the nitty-gritty details.
First of all, one of the best ways to determine how many topics you should model is with an elbow plot. This is the same technique often used to determine how many clusters to choose with the clustering algorithms. In this case, we'll plot the coherence score against the number of topics:
You'll generally want to pick the lowest number of topics where the coherence score begins to level off. This is why it's called an elbow plot - you pick the elbow between steep gains and shallow gains. In this case (and it's a remarkably spiky case; usually the curves are a little bit smoother than this), I'd go with somewhere around 20 topics.
The first model I used is Gensim's ldamodel. At 20 topics, Gensim had a coherence score of 0.319. This is not great; indeed the Mallet algorithm which we'll look at next almost always outperforms Gensim's. However, one really cool thing with Gensim is the pyLDAvis, an interactive chart you can run in a Jupyter notebook. It plots the clusters with two principal components and shows the proportion of words in each cluster:

Harry Potter and the Allocation of Dirichlet
The next implementation I looked at was Mallet (MAchine Learning for LanguagE Toolkit), a Java-based package put out by UMASS Amherst. The difference between Mallet and Gensim's standard LDA is that Gensim uses a Variational Bayes sampling method which is faster but less precise that Mallet's Gibbs Sampling. Fortunately for those who prefer to code in Python, Gensim has a wrapper for Mallet: Latent Dirichlet Allocation via Mallet. In order to use it, you need to download the Mallet Java package from here http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip and also install the Java Development Kit. Once everything is set up, implementing the model is pretty much the same as Gensim's standard model. Using Mallet, the coherence score for the 20-topic model increased to 0.375 (remember, Gensim's standard model output 0.319). It's a modest increase, but usually persists with a variety of data sources so although Mallet is slightly slower, I prefer it for its increase in return.
Finally, I built a Mallet model on the 192 chapters of all 7 books in the Harry Potter series. Here are the top 10 keywords the model output for each latent topic. How would you name these clusters?
