# TODO

"""
Nice wrapper code around the actual logic
    1. Add photos of friends
    2. Label photos of friends
    3. Get photos from webcam (optional)
    4. Train the model and save it
    5. Ask for a testing photo
    6. Detect all faces from the photo.
    7. Draw bounding box around it 
    8. Label the name with % confidence.
    9. If no match then say no match
"""

# The similarity network will judge similarity based on embeddings and will help in creating more
# training data.
# first make random pairs for training.

"""
Take 2 inputs into the model, then pass it through the model. Now calculate the similarity score 
using triplet loss. Make the anchor go different processing to get different embeddings
"""