#importng libraries
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import config
#from keras.models import Sequential
#from keras.layers import Dense


#ML Model
#func to train model
def train_model(features, labels):
    model = LinearRegression()
    model.fit(features, labels)
    return model


#func to predict the model
def predict_frequencies(model, features):
    predictions = model.predict(features)
    return predictions



#setting up authentification
#importing my details for data
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=config.CLIENT_ID,
                                               client_secret=config.CLIENT_SECRET,
                                               redirect_uri=config.REDIRECT_URI,
                                               scope="user-library-read playlist-read-private"))





#now analyzing playlists and finding their frequesncies for cymatics::

#function to get audio features from a track(s)
def extract_audio_features(track_id):
    # Get audio analysis for the track
    audio_analysis = sp.audio_analysis(track_id)
    
    #extract relevant features (example: pitches and durations)
    pitches = [segment['pitches'] for segment in audio_analysis['segments']]
    durations = [segment['duration'] for segment in audio_analysis['segments']]
    
    #calculate mean pitch values for the entire track
    mean_pitches = np.mean(pitches, axis=0)
    
    #calculate total duration of the track
    total_duration = sum(durations)
    return mean_pitches, total_duration


#function to get playlist(s)
def get_user_playlists():
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        print(f"Playlist Name: {playlist['name']}, Playlist ID: {playlist['id']}")


#finding audio features from a track
def get_track_features(playlist_id):
    playlist_tracks = sp.playlist_tracks(playlist_id)
    all_features = []

    for track in playlist_tracks['items']:
        track_id = track['track']['id']
        if track_id:
            mean_pitches, total_duration = extract_audio_features(track_id)
            all_features.append((mean_pitches, total_duration))



        #finding audio data
        #finding mean of pitches of all tracks
        if all_features:
            features = np.array([item[0] for item in all_features])
            labels = np.array([item[1] for item in all_features])

            print("Features shape:", features.shape)
            print("Labels shape:", labels.shape)

            model = train_model(features, labels)
            print("Model trained successfully!")

            predictions = predict_frequencies(model, features)
            print("Predictions:", predictions)

            return predictions
        else:
            print("No audio data found in the playlist.")


#implements the preprocessing into machine learning 
#using keras for ML model
#def train_model(features, labels):
#    model = Sequential([
#        Dense(64, activation='relu', input_shape=(features.shape[1],)),
#        Dense(1) 
#    ])
#    model.compile(optimizer='adam', loss='mean_squared_error')
#    model.fit(features, labels, epochs=10, batch_size=32)  
#    return model


#analysing tracks in playlist
        #calling playlist_id in get_track_features func
def analyze_playlist(playlist_id):
    get_track_features(playlist_id)


#calling all functions in right order using the "main" func
if __name__ == "__main__":
    #user input for specific playlist
    get_user_playlists()
    playlist_id = input("Enter the ID of the playlist you want to analyze: ")

    #analysing playlist
    analyze_playlist(playlist_id)
    #playlist_features = get_track_features(playlist_id)