# Good Film Hunting - Movie Tracking and Recommendation Tool

## Project Overview
Over the past two decades, the modern consumer has become digital forward; discovery of new products is driven by recommendation systems packaged on apps and platforms over both mobile and desktop apps. We rely on companies like Google and Amazon to recommend goods and services that they predict we will purchase, based on a combination of implicit and explicit data that we provide them. GoodReads recommends books, TikTok recommends short form content, Netflix recommends movies and TV shows, and a number of restaurant recommendation apps have also emerged. 

Although Netflix succeeds as presenting itself as the new-age television, with it's massive library of content that consumers can engage with for hours, there has been some question as to the accuracy of its recommendation engine for users. Netflix's recommendation system largely relies on item-based filtering and implicit user data, as users rarely actively input review information on content they consume. There exists, therefore, an opportunity for a tool or application for users to track and rate content that they have watched, and then get recommended relevent content based on their personal tastes.

For this project, I sought to begin building a recommendation system based on explicit data and collaborative filtering that allows users to track content that they have seen, and be presented with personalized suggestions on movies to watch, based on criteria like genre.

## Application Inspiration
The inspiration for this application came from a different application that I recently discovered called [Beli](https://beliapp.com/). Beli is a restaurant tracking and recommendation app that allows users to select restaurants they have been to, rate them on a 1-3 level scale, and then asks users to rank it to other restaurants they have been to. The platform then begins to build a user-specific normalized distribution of restraurant ratings on a 1-10 scale, and each time a new restaurant is added to the list, ratings adjust based on where the new restaurant lies in the ranking distribution. Over time, the recommendation system intimately learns a users preferences, and then uses a combination of both collaborative and item-based filtering to customize a user-specific rating for any restaurant in any city (in the Beli database).  

While at home with my parents, I noticed that they spent a lot of time searching for new content across the various platforms that they subscribe to, but due to an oversaturation of content options and little clarity on how to search, quit the process before selecting something to watch. An application that learns a user's viewing habits and preferences, and then is able to recommend content and a platform based on some minor inputs, would significantly improve the consumer viewing experience. 

## Data Understanding
To conduct this project, I used the famous Movie Lens datasets provided by the [Group Lens](https://grouplens.org/datasets/movielens/) Research Group of The University of Minnesota. This lab has collected movie reviews and user demographics data for more than 20 years, and has cleaned and packaged various sizes of datasets for public use. The two datasets I used were the 1M and 25M datasets. 

Movie Lens 1M Dataset:
- 1 million movie ratings 
- 6,000 users
- 4,000 movies
- movie genres
- user age
- user occupation 
- gender
- zipcode

Movie Lens 25M Dataset:
- 25 million movie ratings 
- 600,000 users 
- 60,000 movies
- movie genres 
- tags 

These two datasets require different approaches for modeling. The 1M dataset is small enough to conduct analysis with SVD via the Surprise library within a Jupyter notebook. The 25M dataset requires distributed computing, so I used Spark via Databricks to run the ALS methodology for analysis. 

## Data Analysis
### SVD
To build a baseline model, I used the 1M dataset and the Surprise library to train and test the movie review dataset. The success metric used for this model was RMSE, which represented the margin of error of the predicted values for movies that a user had not seen. The baseline results were promising, consisting of an RMSE of .8715. After running a gridsearch for various levels of regularization (from .02 to .1) and number of factors (latent values ranging from 20 to 100), it was determined that factors=70 and regularization=.03 yielded optimal results. This brought RMSE down to .8688. A third model was run on a dataframe that merged the initial users, movies, and ratings dataframe with movie genres (one hot encoded), year release, and user demographic information. This type of model incorporated both collaborative and item-based filtering as more information was provided for movies (genre counts, release year). This yielded an RMSE of .8672. 

Once the model was chosen, I built three functions that could begin to serve as the core for a movie recommendation application. 

#### Get Similar Movies
This function takes in the movie title, dataframe, and model, and leverages cosine similarity to return the top 5 movies most similar to the chosen movie. For example, if a user recently watched Toy Story, and decided that he wanted to see something similar to it, he could use this function to look up similar movies. This feature allows users to engage with the app without fully investing the time and energy of providing information. If the results are satisfactory, odds are he will move on to the next feature seen in function 2.

![Function_1]("Images\function_1.jpg")
![Function_1]("Images\function_1b.jpg")

#### Rate Movies
This function presents movies to a user in order of descending platform popularity, and asks the user to rank it on a 1-5 scale. If the user has not seen the movie, he may skip it. This process will continue until the user has ranked 10 movies. These rankings are then stored in a unique list and a new user ID is generated. In some future state, new user lists can be appended to the model to improve it.

![Function_2]("Images\function_2.jpg")

#### Get Top Movie Recommendations
This function takes in user ID, a specific genre, and number of films for output, and returns the top films of that specific genre for that specific user. Once a user has engaged with the platform and gone through the 'Rate Movies' process, this function ultimately provides the biggest value for said user.

![Function_3]("Images\function_3.jpg")

### ALS
The optimal model was run on Spark using Databricks using the 25M dataset. Following the same general steps as with SVD (except using pySpark instead), I was able to split the data, and then tune hyperparameters to yield an RMSE of .7851. Much like with SVD, I also attempted to merge a full dataset that included movie information like genre and release year, but the dataset was so massive that I could only run a baseline model as the gridsearch ran for more than 24 hours with no output. The RMSE of the full baseline model was .8078. Considering that the first ALS model improved by nearly .04 (from .82 to .7851), it is fair to assume that a similar improvement would result from fine tuning the hyper parameters of the full model. 

## Conclusion and Next Steps
Through the above analysis, it is clear that with more data, the recommendation improves. As the dataset increases in size, it becomes imperative to shift away from SVD and instead use ALS, as it is better suited for distributed computing. The ALS model can be improved by building a wide and deep model that uses both collaborative and item-based filtering. The next steps to improve the model are to complete the run of the full dataframe with hyperparameters tuned. An additional model with more meta-data like box office information, movie cast and crew, and movie review data can be incorporated to build a more comprehensive model with a lower RMSE. 

In some future state, I hope to use this comprehensive data to train a neural network and see if recommendation error improves. The first steps in deploying this tool is through a web app; seeing if I can direct traffic to it and generate engagement. If there is sufficient engagement, I can begin thinking of incorporating social components to the tool so users can share and compare with their peers through some type of mobile application. 


