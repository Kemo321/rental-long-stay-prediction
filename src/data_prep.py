import pandas as pd
import os

def main() -> None:
    """Prepare a processed dataset for long-stay rental prediction.

    Pipeline steps:
    1. Read raw CSV files.
    2. Build aggregate features from calendar and reviews.
    3. Clean and enrich listings/users/sessions data.
    4. Merge sources into one modeling table.
    5. Perform final feature engineering and missing-value handling.
    6. Save the processed dataset to disk.
    """
    os.makedirs('data/processed', exist_ok=True)

    print("1. Loading 5 files from data/raw/ ... (this may take several seconds)")
    listings: pd.DataFrame = pd.read_csv('data/raw/listings.csv')
    sessions: pd.DataFrame = pd.read_csv('data/raw/sessions.csv')
    users: pd.DataFrame = pd.read_csv('data/raw/users.csv')
    reviews: pd.DataFrame = pd.read_csv('data/raw/reviews.csv')
    calendar: pd.DataFrame = pd.read_csv('data/raw/calendar.csv')

    print("2. Extracting features from calendar and reviews (aggregations)...")
    
    # -- CALENDAR: Compute yearly availability rate for each listing --
    calendar['is_available'] = (calendar['available'] == 't').astype(int)
    availability: pd.DataFrame = calendar.groupby('listing_id')['is_available'].mean().reset_index()
    availability.rename(columns={'is_available': 'availability_rate'}, inplace=True)
    
    # -- REVIEWS: Compute mean review length (safe for missing comments) --
    reviews['comment_length'] = reviews['comments'].fillna('').str.len()
    avg_review_len: pd.DataFrame = reviews.groupby('listing_id')['comment_length'].mean().reset_index()
    
    # -- USERS: Flag whether the user provided city information --
    users['has_city_info'] = users['city'].notna().astype(int)

    print("3. Cleaning data (Listings and Sessions) and generating new features...")
    
    # -- LISTINGS: Price, amenities, and superhost status --
    # Convert price from currency string (e.g., "$123.00") to numeric.
    listings['price_num'] = listings['price'].astype(str).str.replace(r'[\$,]', '', regex=True)
    listings['price_num'] = pd.to_numeric(listings['price_num'], errors='coerce')

    # Extract specific amenity indicators from free-text amenities field.
    listings['has_kitchen'] = listings['amenities'].str.contains('Kitchen', case=False, na=False).astype(int)
    listings['has_washer'] = listings['amenities'].str.contains('Washer', case=False, na=False).astype(int)
    listings['has_workspace'] = listings['amenities'].str.contains('workspace', case=False, na=False).astype(int)
    
    # Approximate the total number of amenities using comma count.
    listings['amenities_count'] = listings['amenities'].str.count(',') + 1
    
    # Map host superhost flag: 't' -> 1, otherwise 0.
    listings['is_superhost'] = (listings['host_is_superhost'] == 't').astype(int)

    # -- SESSIONS: Stay length and booking lead time --
    bookings: pd.DataFrame = sessions[sessions['action'] == 'book_listing'].copy()
    
    # Parse date columns used for time-based features.
    bookings['session_time'] = pd.to_datetime(bookings['timestamp'], errors='coerce').dt.normalize()
    bookings['booking_date'] = pd.to_datetime(bookings['booking_date'], errors='coerce')
    bookings['booking_duration'] = pd.to_datetime(bookings['booking_duration'], errors='coerce')
    
    # Stay length in days.
    bookings['stay_length_days'] = (bookings['booking_duration'] - bookings['booking_date']).dt.days
    
    # Lead time in days (how far in advance the booking was made).
    bookings['lead_time_days'] = (bookings['booking_date'] - bookings['session_time']).dt.days
    # Clip negative values that can appear due to timezone inconsistencies.
    bookings['lead_time_days'] = bookings['lead_time_days'].clip(lower=0)

    # Remove rows with invalid dates or missing key identifiers.
    valid_bookings: pd.DataFrame = bookings.dropna(subset=['stay_length_days', 'listing_id', 'user_id']).copy()

    print("4. Joining all sources into one base table...")
    # Merge all available sources into a single modeling DataFrame.
    df: pd.DataFrame = valid_bookings.merge(listings, left_on='listing_id', right_on='id', how='inner')
    df = df.merge(users, left_on='user_id', right_on='id', how='left', suffixes=('', '_user'))
    df = df.merge(availability, on='listing_id', how='left')
    df = df.merge(avg_review_len, on='listing_id', how='left')

    print("5. Final feature engineering...")
    
    # Target variable: stays of 7+ days are labeled as long stays.
    df['is_long_stay'] = (df['stay_length_days'] >= 7).astype(int)

    # Core numeric features used by the model.
    numeric_features: list[str] = [
        'accommodates', 'bathrooms', 'bedrooms', 'beds', 
        'price_num', 'number_of_reviews', 'review_scores_rating',
        'availability_rate', 'comment_length', 'has_city_info',
        'has_kitchen', 'has_washer', 'has_workspace', 'amenities_count',
        'is_superhost', 'lead_time_days', 'reviews_per_month', 'calculated_host_listings_count'
    ]
    # Categorical features to encode.
    categorical_features: list[str] = ['room_type']

    # Missing-value handling (imputation).
    df['review_scores_rating'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())
    df['comment_length'] = df['comment_length'].fillna(0)
    # If unknown, assume neutral availability (50%).
    df['availability_rate'] = df['availability_rate'].fillna(0.5)
    
    # Fill remaining numeric nulls with 0 (including no-review listings).
    df[numeric_features] = df[numeric_features].fillna(0)

    # One-hot encode categorical variables.
    df_encoded: pd.DataFrame = pd.get_dummies(df[categorical_features], drop_first=True)

    # Build final training table (features + target).
    processed_df: pd.DataFrame = pd.concat([df[numeric_features], df_encoded, df[['is_long_stay']]], axis=1)

    print("6. Saving processed data...")
    output_path: str = 'data/processed/processed_data.csv'
    processed_df.to_csv(output_path, index=False)
    print(f"SUCCESS! Processed dataset with engineered features saved to: {output_path}")

if __name__ == "__main__":
    main()