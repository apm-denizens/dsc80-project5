# League of League of Legends Role Prediction from Post-Game Data

## Framing the Problem

-   Clearly state your prediction problem and type (classification or regression). If you are building a classifier, make sure to state whether you are performing binary classification or multiclass classification.

This is a multiclass classification problem. The role/'position' being predicted has 5 possible values ('top', 'jng', 'mid', 'bot', 'sup').

-   Report the response variable (i.e. the variable you are predicting) and why you chose it, the metric you are using to evaluate your model and why you chose it over other suitable metrics (e.g. accuracy vs. F1-score).

The response variable, as you might be able to guess, is 'position'. The reason we chose it, as you might be able to guess, is because this is the only variable that contains role information. If we're trying to predict role information, there really isn't much other choice.

The metric we have chosen to evaluate the model with is simply accuracy. Since we are not dealing with high-stakes scenarios (like medical diagnosis) where false-negatives could lead to deaths, we don't really need to consider precision & recall. For our case, false-negatives and false-positives have an equal cost. We want to maximize true-positives & true-negatives, and since we have an exactly even class distribution, we have chosen accuracy.

-   Note: Make sure to justify what information you would know at the “time of prediction” and to only train your model using those features. For instance, if we wanted to predict your final exam grade, we couldn’t use your Project 5 grade, because Project 5 is only due after the final exam!

Since we're dealing with post-game data, our "time of prediction" would be when the game has already ended. Subsequently, other than the 'position' column (what we're trying to predict), we have access to all of the other columns at prediction time.

## Baseline Model

-   Describe your model and state the features in your model, including how many are quantitative, ordinal, and nominal, and how you performed any necessary encodings.

For our baseline model, we decided to keep it fairly bare-bones. The features we chose for it were, 'champion' (the name of the champion) and 'kill-death ratio' (an engineered feature for kills/deaths).

For 'champion' (nominal variable), most champions are usually designed for a specific role/position, so there is a ton of information that can be gained for using this column to predict position. We decided to one-hot encode this column, as it is a nominal data column.

The features of 'kills' and 'deaths' (both quantitative discrete variables) both potentially provide information into what role a player has. Roles such as supports, usually will have far less kills than more damage-oriented roles like a midlaner. Furthermore, since supports end up receiving less resources, they will end up being easier to kill than other members on their team. Thus, they would be expected to have a higher amount of deaths as well.

We chose to engineer a new feature 'kill-death ratio' (a quantitative continuous variable) because depending on how long a game goes on, kills and deaths could both increase dramatically. A high-kill hero, in a short game, could have less kills than a passive support in a long game. This fact could very well make it harder for a model to pick up on kill death differences between roles. Accounting for this through a ratio, makes it so that we have a feature that is more consistent and resilient to the effects of game duration.

-   Report the performance of your model and whether or not you believe your current model is “good” and why.

For our baseline model, we achieved these accuracies:

TRAIN: 0.941124497991967
VAL: 0.931004016064257

In this case, it doesn't seem as if there is too much overfitting going on, which is nice. The training accuracy is not significantly higher than the validation accuracy. Considering that we only used two features, this performance is better than expected. (We initially thought that the model would get maybe an 80% accuracy. No scientific reason for why, pretty much just a feeling.)

(ASK COSTIN)
For the human benchmark, since neither of us are avid league players, we're not too sure on how perfectly we can tell position from post-game data. Positions such as mid and bot can be difficult to tell apart. They both will have higher kd's and plenty of resources. It's not surprising to see mids have more resrouces than bots.

-   Tip: Make sure to hit all of the points above: many projects in the past have lost points for not doing so.

## Final Model

-   State the features you added and why they are good for the data and prediction task. Note that you can’t simply state “these features improved my accuracy”, since you’d need to choose these features and fit a model before noticing that – instead, talk about why you believe these features improved your model’s performance from the perspective of the data generating process.

-   Describe the modeling algorithm you chose, the hyperparameters that ended up performing the best, and the method you used to select hyperparameters and your overall model. Describe how your Final Model’s performance is an improvement over your Baseline Model’s performance.
