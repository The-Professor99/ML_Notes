Time Series Modeling: The Process.

We could imagine learning the components of a time series as an iterative process: first learn the trend and subtract it out from the series, then learn the seasonality from the detrended residuals and subtract the seasons out, then learn the cycles and subtract the cycles out, and finally only the unpredictable error remains.
Add together all the components we learned and we get the complete model. This is essentially what linear regression would do if you trained it on a complete set of features modeling trend, seasons, and cycles.

In previous lessons, we used a single algorithm (linear regression) to learn all the components at once. But it's also possible to use one algorithm for some of the components and another algorithm for the rest. This way we can always choose the best algorithm for each component. To do this, we use one algorithm to fit the original series and then the second algorithm to fit the residual series.

We'll usually want to use different feature sets (X_train_1 and X_train_2 above) depending on what we want each model to learn. If we use the first model to learn the trend, we generally wouldn't need a trend feature for the second model, for example.

Successfully combining models, though, requires that we dig a bit deeper into how these algorithms operate

For inspiration, check out Kaggle's previous forecasting competitions. Studying winning competition solutions is a great way to upgrade your skills.

Here are some great resources you might like to consult for more on time series and forecasting. They all played a part in shaping this course:

Learnings from Kaggle's forecasting competitions, an article by Casper Solheim Bojer and Jens Peder Meldgaard.
Forecasting: Principles and Practice, a book by Rob J Hyndmann and George Athanasopoulos.
Practical Time Series Forecasting with R, a book by Galit Shmueli and Kenneth C. Lichtendahl Jr.
Time Series Analysis and Its Applications, a book by Robert H. Shumway and David S. Stoffer.
Machine learning strategies for time series forecasting, an article by Gianluca Bontempi, Souhaib Ben Taieb, and Yann-Aël Le Borgne.
On the use of cross-validation for time series predictor evaluation, an article by Christoph Bergmeir and José M. Benítez.


### Prediction of Football Match Results with Machine Learning
The unpredictability of sport is widely known. Inorder to make predictions of football matches before they take place, it is necessary to provide the prediction models with data that is available before the start of each match. Because of this, we consider the averages of the available data such as the average of goals before a given game. 

In this study, for a given match, the average goals for the home team are calculated based on the previous home matches played by that team during that season. For away teams, averages are also calculated based on previous games played as visitiors during the season in which the game is played.
Then, in the total variables available, it was verified which ones were most related to each other, which variables could more clearly predict the goal attribute and which could be excluded. To verify the relationship vetween variables, a correlation matrix was made between all numerical variables.


Note that performance of teams and players change seasons by seasons.
The test set should include all the games of a season, because throughout the season the team performance varies. At the beginning of the season, the teams may not be at their normal level and at the end of the season, they may be worn out or have already reached their goals, leading some teams to have a performance below normal. These factors can lead to unexpected results, so for a classification model to be considered credible, it must be tested over an entire season.

Variables to create:
Average goals of a team(before a game):
- Total
- Home
- Away
Average goals conceeded by a team(before a game):
- Total
- Home
- Away

Add number of odd and even goals scored in prev matches(before a game)
Add number of odd and even goals when officiated by referee


factor in X match outcomes into algorithm
Typically for example, given that games start with an even number, an even number of goals is seen as being more likely by bookies.As more than six goals are super rare in a game, the outcome is slightly skewed to “Even Goals” as 0-0 and 6 total goals both count toward even.[https://parimatch.co.tz/blog/en/goals-odd-even/#:~:text=Summary%20on%20Odd%2FEven%20Goals&text=With%20roughly%20a%2050%2F50,the%20profit%20margin%20is%20slim.]
As experienced punters know, the draw can skew your betting predictions, especially with winner bets (1X2). As we said before, draws happen 25% of the time in soccer, so it’s important to factor this in when betting. Evens/odds bets don’t take into account extra time or penalties, meaning a draw is always possible. Moreover, a draw is always even. In this sense, betting on an even number of total goals can be a plus to protect your stake a little mor
decimal odds are used.
many experienced gamblers will tell you that this kind of betting is simply guesswork
decimal odds are used.

### investopedia
According to a study published in the Journal of Gambling Studies, the more hands a player wins, the less money they are likely to collect, especially with respect to novice players. That is because multiple wins are likely to yield small stakes, for which you need to play more, and the more you play, the more likely you will eventually bear the brunt of occasional and substantial losses.

Behavioral economics comes into play here. A player continues playing the lottery, either in hopes of a big gain that would eventually offset the losses or the winning streak compels the player to keep playing. In both cases, it is not rational or statistical reasoning but the emotional high of a win that motivates them to play further.


### Attention Mechanism
One weakness of the encoder-decoder architecture is that the final hidden state of the encoder creates an _information bottleneck_ as it has to represent the meaning of the whole input sequence since this is all the decoder has access to when generating the output. This is especially challenging for long sequences where information at the start of the sequence might be lost in the process of compressing everything into a single, fixed representation. One way out of this bottleneck is by allowing the decoder to have access to all of the encoder's hidden states. This mechanism is called _attention_.
The main idea behind attention is that instead of producing a single hidden state for the input sequence, the encoder outputs a hidden state at each step that the decoder can access. However, using all the states at the same time would create a huge input for the decoder, so some mechanism is needed to prioritize which states to use. This is where attention comes in: it lets te decoder assign a different amount of weight, or "attention" to each of the encoder states at every decoding timestep. By focusing on which input tokens are most relevant at each timestep, these attention-based models are able to learn non trivial alignments between the words in the generated translation and those in the source sentence.

### Transfer Learning in NLP
Transfer learning, architecturally, involves splitting a model into a body and a head, where the head is a task specific network. During training, the weights of the body learn broad features of the source domain, and these weights are used to initialize a new model for the new task.

PyTorch and TensorFlow also offer hubs of their own and are
worth checking out if a particular model or dataset is not available
on the Hugging Face Hub.

Having a good dataset and powerful model is worthless if one can't reliably measure the performance. Unfortunately, classic NLP metrics come with many different implementations that can vary slightly and lead to deceptive results. The huggingface Dataset API  provides scripts for many metrics, and thus helps make experiments more reproducible and the results more trustworthy.

### Dealing with Imbalanced Datasets.
Some ways to deal with imbalanced datasets include:
- Randomly oversample the minority class.
- Randomly undersample the majority class.
- Gather more labeled data from the underrepresented classes.
Checkout [imbalanced-learn library](https://imbalanced-learn.org/stable/user_guide.html) and make sure not to apply sampling methods before creating your train/test splits, or you'll get plenty of leakage between them.

### Tokenization
When tokenizing, you can treat the whole dataset as a single batch if you want to ensure that the input tensors and attention masks have the same shape globally. Data collators can be used to dynamically pad the tensors in each batch hence you may not need to treat the whole dataset as a single batch as inputs are padded globally.
