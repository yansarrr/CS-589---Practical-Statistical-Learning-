from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)


genre_list = ["Action", "Adventure", "Animation",
               "Children's", "Comedy", "Crime",
               "Documentary", "Drama", "Fantasy",
               "Film-Noir", "Horror", "Musical",
               "Mystery", "Romance", "Sci-Fi",
               "Thriller", "War", "Western"]


@app.route('/getmsg/', methods=['GET'])
def respond():
    # Retrieve the name from the url parameter /getmsg/?name=
    name = request.args.get("name", None)

    # For debugging
    print(f"Received: {name}")

    response = {}

    # Check if the user sent a name at all
    if not name:
        response["ERROR"] = "No name found. Please send a name."
    # Check if the user entered a number
    elif str(name).isdigit():
        response["ERROR"] = "The name can't be numeric. Please send a string."
    else:
        response["MESSAGE"] = f"Welcome {name} to our awesome API!"

    # Return the response in json format
    return jsonify(response)


@app.route('/post/', methods=['POST'])
def post_something():
    param = request.form.get('name')
    print(param)
    # You can add the test cases you made in the previous function, but in our case here you are just testing the POST functionality
    if param:
        return jsonify({
            "Message": f"Welcome {name} to our awesome API!",
            # Add this option to distinct the POST request
            "METHOD": "POST"
        })
    else:
        return jsonify({
            "ERROR": "No name found. Please send a name."
        })


@app.route('/system1/genre', methods=['GET'])
def system1():
    genre = request.args.get('genre')
    return jsonify(popularity_dict[genre])


def load_data():
    movies = pd.read_csv('~/Downloads/ml-1m/movies.dat', sep='::', names=['MovieID', 'Title', 'Genre', 'Year'],
                            encoding='latin-1')
    ratings = pd.read_csv('~/Downloads/ml-1m/ratings.dat', sep='::',
                             names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    return movies, ratings


def preprocess(movies, ratings, year_filter=None):
    # assign year
    movies['Year'] = movies['Title'].str[-5:-1]
    movies['Year'] = movies['Year'].str.extract('(\d+)', expand=False)
    movies = movies[movies['Year'].notna()]
    movies['Year'] = movies['Year'].astype(int)

    # remove year from title
    movies['Title'] = movies['Title'].str[:-6]

    # remove movies where there are less than 30 total reviews
    ratings_grouped = ratings.groupby('MovieID', as_index=False).size()
    movies = movies.join(ratings_grouped.set_index('MovieID'), on='MovieID')
    movies = movies.rename(columns={'size': 'RatingCount'})
    size1 = len(movies)
    movies = movies[movies['RatingCount'] >= 30]
    size2 = len(movies)
    print('\n\nRemoved', size1-size2, 'movies because there aren\'t enough reviews\n\n')

    # remove movies from before a certain year
    if year_filter:
        movies = movies[movies['Year'] >= year_filter]

    # assign average ratings
    ratings_avg = ratings.groupby('MovieID', as_index=False)['Rating'].mean()
    movies = movies.join(ratings_avg.set_index('MovieID'), on='MovieID')
    movies = movies.rename(columns={'Rating': 'AvgRating'})

    return movies


popularity_dict = {}


# populates the popularity dict
def get_top_by_genre(movies, category, top_n):
    for genre in genre_list:
        genre_movies = movies[movies['Genre'].str.contains(genre)]
        genre_top = genre_movies.sort_values(by=[category], ascending=[False]).head(top_n)
        popularity_dict[genre] = genre_top.to_dict('records')


if __name__ == '__main__':
    # Get the top 5 movies per genre by popularity (quantity of reviews)
    movies_df, ratings_df = load_data()
    cleaned_movies_df = preprocess(movies_df, ratings_df)
    get_top_by_genre(cleaned_movies_df, 'RatingCount', top_n=5)

    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
