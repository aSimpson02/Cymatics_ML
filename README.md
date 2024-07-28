# Spotify Playlist Audio Analysis Tool For Cymatics 

This project is a Python application that utilises Spotify's API and certain data analysis libraries/modules to extract and analyse features from tracks' audios in the user's selected playlist. The extracted features are trained using a machine learning model to predict audio characteristics.

## Features
* Spotify Integration: Connects to Spotify's API to access user music data, including playlists and track information.
* Audio Feature-Extraction: Uses librosa library to extract and process audio features from tracks.
* Machine Learning: Trains a linear regression model for predicting audio characteristics based on extracted features.
* Interactive Analysis: Allows users to select a playlist for analysis and view predictions.

## Usage
1. Authenticate: This project will require you to log in to your Spotify account for authorisation.
2. Select Playlist: The application will list your playlists. Enter the ID of the playlist you want to analyse in your terminal.
3. Analyse: Then audio features will b eextracted, trained in a machine learning model, and its frequency shown.


## code structure
The config.py file contains the information needed to acquirecthe spotify API, it will need your own credentials inside it. 

The main.py file is the main code which takes your details and begins analysingand predicting your chosen playlist. 
Below is an overview of the key functions and processes in the main.py file:
* Importing relevant libraries
* Authentification setup
* Machine learning model
* Audio feature extractino
* Playlist retrieval and analysis
* the main execution
