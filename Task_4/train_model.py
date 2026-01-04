import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ------------------------------
# Load dataset
# ------------------------------
csv_path = os.path.join(BASE_DIR, "ratings.csv")
ratings = pd.read_csv(csv_path )
ratings = ratings[['userId', 'movieId', 'rating']]

# ------------------------------
# Create full user-item matrix
# ------------------------------
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0).astype(float)

# ------------------------------
# Train-test split by masking some ratings (randomly)
# ------------------------------
test_ratio = 0.2
train_matrix = user_item_matrix.copy().astype(float)
test_matrix = pd.DataFrame(0.0, index=user_item_matrix.index, columns=user_item_matrix.columns)

np.random.seed(42)
for user in user_item_matrix.index:
    rated_movies = user_item_matrix.loc[user][user_item_matrix.loc[user] > 0].index
    test_size = max(1, int(len(rated_movies) * test_ratio))
    test_movies = np.random.choice(rated_movies, size=test_size, replace=False)
    
    train_matrix.loc[user, test_movies] = 0.0  # remove from train
    test_matrix.loc[user, test_movies] = user_item_matrix.loc[user, test_movies]  # put in test

# ------------------------------
# Train SVD on train_matrix
# ------------------------------
svd = TruncatedSVD(n_components=50, random_state=42)
latent_matrix = svd.fit_transform(train_matrix)
reconstructed_matrix = np.dot(latent_matrix, svd.components_)

# ------------------------------
# Evaluation on test_matrix
# ------------------------------
actual = test_matrix.values
predicted = reconstructed_matrix

mask = actual > 0  # only compare for actual test ratings
rmse = np.sqrt(mean_squared_error(actual[mask], predicted[mask]))
mae = mean_absolute_error(actual[mask], predicted[mask])

print("RMSE:", rmse)
print("MAE:", mae)
