import sqlite3
import numpy as np
from collections import Counter
import cv2

def view_database():
    # Connect to the database
    conn = sqlite3.connect('customer_visits.db')
    c = conn.cursor()

    # Query to get the customer_id, emotion, timestamp, and image from face_images
    c.execute("SELECT customer_id, emotion, timestamp, image FROM face_images")
    face_image_rows = c.fetchall()

    # Query to get the customer_id, dominant_emotion, and timestamp from visit_emotions
    c.execute("SELECT customer_id, timestamp, dominant_emotion FROM visit_emotions")
    visit_emotion_rows = c.fetchall()

    # Check if the database is empty
    if not face_image_rows and not visit_emotion_rows:
        print("The database is empty.")
        conn.close()
        return

    # Dictionary to track emotions and frequency for each customer
    customer_data = {}

    # Process face_images data
    for row in face_image_rows:
        customer_id = row[0]
        emotion = row[1]
        timestamp = row[2]
        image_blob = row[3]

        if customer_id not in customer_data:
            customer_data[customer_id] = {
                'emotions': [],
                'timestamps': [],
                'image_blob': image_blob,
                'dominant_emotions': []
            }

        customer_data[customer_id]['emotions'].append(emotion)
        customer_data[customer_id]['timestamps'].append(timestamp)
    
    # Process visit_emotions data
    for row in visit_emotion_rows:
        customer_id = row[0]
        visit_timestamp = row[1]
        dominant_emotion = row[2]

        if customer_id in customer_data:
            customer_data[customer_id]['dominant_emotions'].append({
                'timestamp': visit_timestamp,
                'dominant_emotion': dominant_emotion
            })

    # Display data for each customer
    for customer_id, data in customer_data.items():
        # Calculate the dominant emotion overall
        emotion_counts = Counter(data['emotions'])
        overall_dominant_emotion = emotion_counts.most_common(1)[0][0]

        # Get the frequency of visits
        frequency = len(data['timestamps'])

        # Decode and display a sample image (just showing the first image)
        img_array = np.frombuffer(data['image_blob'], np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Show image
        cv2.imshow(f"Customer {customer_id}", image)
        cv2.waitKey(1000)  # Display the image for 1 second (adjust as needed)
        cv2.destroyAllWindows()

        # Print customer info
        print(f"Customer ID: {customer_id}")
        print(f"Overall Dominant Emotion: {overall_dominant_emotion}")
        print(f"Frequency of Visits: {frequency}")
        print(f"Visit Timestamps: {data['timestamps']}")

        # Print dominant emotions for each visit
        print("Dominant Emotions per Visit:")
        for visit in data['dominant_emotions']:
            print(f"  Timestamp: {visit['timestamp']}, Dominant Emotion: {visit['dominant_emotion']}")
        print("\n")

    # Close the connection
    conn.close()

# Call the function to view the data
view_database()
