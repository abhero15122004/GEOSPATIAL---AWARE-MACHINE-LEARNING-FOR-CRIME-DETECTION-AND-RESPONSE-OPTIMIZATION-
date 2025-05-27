# GEOSPATIAL---AWARE-MACHINE-LEARNING-FOR-CRIME-DETECTION-AND-RESPONSE-OPTIMIZATION-
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from geopy.distance import geodesic
import networkx as nx
from IPython.display import display
import folium
import osmnx as ox
import geopandas as gpd
from shapely import wkt
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

# Load the crime dataset
crime_data = pd.read_csv('/content/AA.csv')

# Load the advanced geospatial dataset
geo_data = pd.read_csv('/content/drive/MyDrive/jammu_kashmir_geospatial_data.csv')

# Preprocessing
label_encoder = LabelEncoder()

# Label encoding for categorical variables
categorical_cols = ['day_of_week', 'suspected_activity', 'risk_level',
                    'weather_conditions', 'nearest_military_camp', 'part_of_day']
for col in categorical_cols:
    crime_data[col] = label_encoder.fit_transform(crime_data[col])

# Convert time to numerical (splitting into hours and minutes)
crime_data['hour'] = crime_data['time'].apply(lambda x: int(x.split(':')[0]) % 24)
crime_data['minute'] = crime_data['time'].apply(lambda x: int(x.split(':')[1]) % 60)
crime_data = crime_data.drop(columns=['time'])  # Drop the original 'time' column

# Define features (X) and target (y)
X = crime_data.drop(['crime_detected'], axis=1)  # Features
y = crime_data['crime_detected']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Load Geospatial Data and Create Road Network Graph
geo_data['geometry'] = geo_data['geometry'].apply(wkt.loads)
geo_gdf = gpd.GeoDataFrame(geo_data, geometry='geometry')

# Separate different feature types
roads_gdf = geo_gdf[geo_gdf['highway'].notnull()]
mountains_gdf = geo_gdf[geo_gdf['natural'] == 'mountain']
rivers_gdf = geo_gdf[geo_gdf['natural'] == 'river']
lakes_gdf = geo_gdf[geo_gdf['natural'] == 'water']

# Build the Road Network Graph
place_name = "Jammu and Kashmir, India"
G = ox.graph_from_place(place_name, network_type='drive')

if 'simplified' not in G.graph or not G.graph['simplified']:
    G = ox.simplify_graph(G)
else:
    print("The graph is already simplified. Skipping simplification step.")

# Function to buffer obstacles and remove nodes within the buffer
def remove_obstacles(G, obstacles_gdf, buffer_distance=0.01):
    if obstacles_gdf.empty or obstacles_gdf['geometry'].isnull().all():
        print("No obstacles found. Skipping obstacle removal.")
        return G

    obstacles_union = obstacles_gdf.unary_union
    obstacles_buffered = obstacles_union.buffer(buffer_distance)

    nodes_to_remove = []
    for node, data in G.nodes(data=True):
        point = gpd.points_from_xy([data['x']], [data['y']])[0]
        if obstacles_buffered.contains(point):
            nodes_to_remove.append(node)

    print(f"Removing {len(nodes_to_remove)} nodes within obstacles...")
    G.remove_nodes_from(nodes_to_remove)
    return G

# Remove mountains and rivers as obstacles
G_safe = remove_obstacles(G, mountains_gdf, buffer_distance=0.01)
G_safe = remove_obstacles(G_safe, rivers_gdf, buffer_distance=0.005)

# Define Functions to Find Nearest Nodes
def get_nearest_node(G, latitude, longitude):
    return ox.distance.nearest_nodes(G, longitude, latitude)

# Military Coding System for Crime Types based on AA dataset
military_code_dict = {
    'border breach': 'Alpha',
    'smuggling': 'Beta',
    'unauthorized surveillance': 'Gamma',
    'infiltration': 'Delta'
}

# Equipment Allocation System based on AA dataset
equipment_dict = {
    'border breach': ['Rifle', 'Night Vision Goggles', 'Body Armor', 'GPS'],
    'smuggling': ['Rifle', 'Drone Surveillance Kit', 'Handcuffs'],
    'unauthorized surveillance': ['Camera Jammer', 'Rifle', 'Binoculars'],
    'infiltration': ['Rifle', 'Thermal Imaging Camera', 'Grenades', 'Body Armor'],
}

# Function to Assign Military Code - Ensure that the crime type exists
def get_military_code(suspected_activity):
    return military_code_dict.get(suspected_activity, "Unknown")

# Plot Safe Paths on Map for all military camps
def plot_safe_paths(crime_location, military_camps, G, shortest_distance, shortest_path_coords):
    crime_map = folium.Map(location=crime_location, zoom_start=12)

    # Mark the crime location (Ending Point)
    folium.Marker(
        location=crime_location,
        popup="Crime Location (Ending Point)",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(crime_map)

    # Define a color list for paths (unique color for each camp)
    path_colors = ['gray', 'green', 'orange', 'purple', 'pink', 'yellow']

    # Iterate through each military camp and plot paths
    for idx, camp in enumerate(military_camps):
        camp_location = (camp['lat'], camp['lon'])
        camp_node = get_nearest_node(G, camp_location[0], camp_location[1])
        crime_node = get_nearest_node(G, crime_location[0], crime_location[1])

        try:
            path = nx.shortest_path(G, source=crime_node, target=camp_node, weight='length')
            path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]

            # Plot each path with a different color from the list
            folium.PolyLine(
                locations=path_coords,
                color=path_colors[idx % len(path_colors)],  # Cycle through colors
                weight=3,
                opacity=0.6
            ).add_to(crime_map)

            # Label the military camp (Starting Point)
            folium.Marker(
                location=camp_location,
                popup=f"{camp['name']} (Starting Point)",
                icon=folium.Icon(color='blue', icon='flag')
            ).add_to(crime_map)

        except nx.NetworkXNoPath:
            print(f"No path found for camp at ({camp['lat']}, {camp['lon']})")

    # Highlight the shortest path in blue
    folium.PolyLine(
        locations=shortest_path_coords,
        color='blue',
        weight=5,
        opacity=0.8
    ).add_to(crime_map)

    display(crime_map)

    print(f"Shortest safe distance: {shortest_distance:.2f} km")

# Function to Allocate Equipment based on Suspected Activity from AA dataset
def allocate_equipment(suspected_activity):
    if suspected_activity in equipment_dict:
        equipment = equipment_dict[suspected_activity]
        print(f"Equipment allocated for {suspected_activity}: {', '.join(equipment)}")
    else:
        equipment = ['Standard Equipment']
        print(f"No specific equipment found for {suspected_activity}. Allocating: {', '.join(equipment)}")
    return equipment

# Function to send email notification
def send_email(subject, body, to_email):
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_password = os.environ.get("SENDER_PASSWORD")

    if not sender_email or not sender_password:
        print("Error: Gmail credentials not set. Please set environment variables.")
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Add emails to military camps
military_camps = [
    {'name': 'Srinagar Army Base', 'lat': 34.0837, 'lon': 74.7973, 'email': 'abhero151228@gmail.com'},
    {'name': 'Tezpur Army Camp', 'lat': 34.6508, 'lon': 74.9267, 'email': 'sumanthponugupati@gmail.com'},
    {'name': 'Jammu Army Base', 'lat': 32.7266, 'lon': 74.8570, 'email': 'jammu_base@gmail.com'},
    {'name': 'Jaisalmer Army Camp', 'lat': 32.9157, 'lon': 74.8783, 'email': 'jaisalmer_camp@gmail.com'},
    {'name': 'Arunachal Military Base', 'lat': 34.2179, 'lon': 75.7278, 'email': 'arunachal_base@gmail.com'},
    {'name': 'Gangtok Military Base', 'lat': 33.0010, 'lon': 74.3172, 'email': 'reddy.abhinay1512@gmail.com'},
]

# Initialize an empty list to store results
crime_summary = []

# Function to store the results in a table
def store_crime_summary(suspected_activity, latitude, longitude, camp_name, distance, military_code, equipment):
    crime_summary.append({
        'Crime Type': suspected_activity,
        'Location (Lat, Lon)': f"({latitude}, {longitude})",
        'Nearest Military Camp': camp_name,
        'Distance to Camp (km)': round(distance, 2),
        'Military Code': military_code,
        'Allocated Equipment': ', '.join(equipment)
    })

# Modify the monitor_and_alert function to store results
def monitor_and_alert_with_summary(X_test, y_test, G, military_camps):
    for i, row in X_test.iterrows():
        prediction = model.predict(row.values.reshape(1, -1))[0]

        if prediction == 1:  # Crime detected
            latitude = row['latitude']
            longitude = row['longitude']

            suspected_activity = "smuggling"

            print(f"\nSuspicious activity detected: {suspected_activity} at location ({latitude}, {longitude})")

            # Get military code for "smuggling"
            military_code = get_military_code(suspected_activity)
            print(f"Military code for {suspected_activity}: {military_code}")

            crime_location = (latitude, longitude)

            # Calculate geodesic distance to all military camps
            shortest_distance = float('inf')
            shortest_camp = None
            shortest_path_coords = None

            for camp in military_camps:
                camp_location = (camp['lat'], camp['lon'])
                distance = geodesic(crime_location, camp_location).km

                # Find the shortest path
                camp_node = get_nearest_node(G, camp_location[0], camp_location[1])
                crime_node = get_nearest_node(G, crime_location[0], crime_location[1])

                try:
                    path = nx.shortest_path(G, source=crime_node, target=camp_node, weight='length')
                    path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]

                    if distance < shortest_distance:
                        shortest_distance = distance
                        shortest_camp = camp
                        shortest_path_coords = path_coords

                except nx.NetworkXNoPath:
                    print(f"No path found for camp at ({camp['lat']}, {camp['lon']})")

            print(f"Geodesic distance to nearest military camp: {shortest_distance:.2f} km")

            # Allocate equipment for "smuggling"
            equipment = allocate_equipment(suspected_activity)

            # Store the result in the crime summary table
            if shortest_camp:
                store_crime_summary(
                    suspected_activity,
                    latitude,
                    longitude,
                    shortest_camp['name'],
                    shortest_distance,
                    military_code,
                    equipment
                )

            # Plot safe paths from all camps and highlight the shortest path
            plot_safe_paths(crime_location, military_camps, G, shortest_distance, shortest_path_coords)

            # Send email notification to the nearest military camp
            if shortest_camp:
                subject = f"Suspicious Activity Alert: {suspected_activity}"
                body = (f"A suspicious activity ({suspected_activity}) has been detected at coordinates "
                        f"({latitude}, {longitude}).\n\n"
                        f"Military Code: {military_code}\n"
                        f"Nearest Military Camp: {shortest_camp['name']}\n"
                        f"Distance: {shortest_distance:.2f} km\n"
                        f"Please take necessary actions.")
                send_email(subject, body, shortest_camp['email'])

# Set environment variables for email credentials
os.environ['SENDER_EMAIL'] = 'Reddy.abhinay1512@gmail.com'
os.environ['SENDER_PASSWORD'] = 'wpwqmqhrswkihptv'

# Function to display the crime summary in a tabular format
def display_crime_summary():
    if crime_summary:
        df = pd.DataFrame(crime_summary)
        display(df)
    else:
        print("No crimes detected.")

# Start monitoring and collecting data for the final summary
monitor_and_alert_with_summary(X_test, y_test, G_safe, military_camps)

# Display the crime summary table
display_crime_summary()

summary_df = pd.DataFrame(crime_summary)
summary_df.to_csv('/content/crime_summary.csv', index=False)
print("Crime summary saved to 'crime_summary.csv'.")
