================================================================================
                    VISUALIZING ART - USER GUIDE
================================================================================

DESCRIPTION
-----------

This project analyzes artwork from the Metropolitan Museum of Art (MET) to 
discover visual and semantic connections between art pieces. The system uses 
CLIP (Contrastive Language-Image Pre-Training) neural networks to extract 
visual features from artwork images and combines them with metadata features 
(artist, medium, department, country, period, culture, dates, etc.) to create 
comprehensive embeddings.

The system generates similarity scores between artworks based on both visual 
appearance and metadata characteristics. These similarities are visualized 
through an interactive web interface that allows users to explore connections 
between artworks, filter by various criteria, and discover hidden relationships 
in the MET collection.

The project processes thousands of artwork images and their metadata, generates 
high-dimensional embeddings, and creates similarity matrices that power the 
interactive visualizations.

See the live website here (with large dataset):

https://storage.googleapis.com/visualizing-art/index.html

For instructions on running the files/server locally please see below--this will
use a reduced dataset that conforms to submission standards.


INSTALLATION
------------

1. Prerequisites:
   - Python 3.7 or higher
   - pip (Python package installer)

2. Install Python dependencies:
   cd python # If you're not already in the python folder
   pip install -r requirements.txt

   This will install all required packages including:
   - numpy, pandas, scikit-learn (data processing)
   - open-clip-torch, torch, torchvision (CLIP model and deep learning)
   - Pillow (image processing)
   - requests, tqdm, openpyxl, xlrd (utilities)

   Note: The pipeline script can also automatically check and install missing 
   dependencies when you run it.


EXECUTION
---------

1. Run the complete pipeline:
   cd python # If you're not already in the python folder
   python run_pipeline.py

   This script will:
   - Check and install missing dependencies automatically
   - Download artwork images and metadata from the MET API (if not already present)
   - Generate CLIP embeddings from images
   - Generate thumbnails (smaller images from the normal images)
   - Process metadata features (categorical, numerical, text)
   - Combine visual and metadata features
   - Generate similarity matrices
   - Start a local web server on port 5500

2. Access the web interface:
   Once you see "Serving HTTP on :: port 5500 (http://[::]:5500/) ...".  Open your web browser and navigate to:
   http://localhost:5500

3. Explore the visualization:
   - Art Connections: Interactive visualization of artwork similarities
   - Art Filters: Filter and explore artworks by various criteria
   - About: Detailed information about the project

4. Stop the server:
   Press Ctrl+C in the terminal where the server is running

Note: The first run will take longer as it downloads images and generates 
embeddings. Subsequent runs will be faster as it reuses existing data
