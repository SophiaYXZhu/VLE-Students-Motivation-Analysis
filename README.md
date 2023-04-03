# VLE Motivation Analysis
Virtual Learning Environment (VLE) is an Internet-based platform providing digital courses. This model can evaluate students' performance and determine effective practices in VLE via a webapp.

# Setting up the Website
Run ```flask run``` to create local server.

# Training the Model
The model is based on semi-supervised learning embeds three intrinsic indexes and two extrinsic indexes as vectors using a simple 3-layered FNN, evaluating students' performance in VLE and providing graphics via the webapp. We adopt with a cosine similarity loss function designed to find the optimized balance between the indexes.

# Mechanics
See the uploaded paper for reference.
