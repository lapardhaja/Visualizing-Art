# 🎨 Visualizing Art

Welcome to the **Visualizing Art** project!

This interactive web-based visualization platform allows users to **explore relationships between thousands of artworks** from museum collections. Each piece of art is represented as a node in a similarity network, where connections are based on shared metadata, visual features, or artist characteristics.

The project combines **data visualization (D3.js)** with **local data processing (Python)** to provide an offline-capable experience using locally stored images and metadata. Users can view high-level relationships on the graph and click into individual artworks for detailed metadata and full-resolution images.

---

## 📁 Project Structure

Below is the current folder and file layout of the project, reflecting the most recent updates.

<details><summary><b>CLICK HERE</b></summary>
<p>

```bash
Visualizing_Art/
├── data/                       # CSVs, Excel files, and combined summary datasets
│   ├── met_paintings.csv
│   ├── MetObjects.xlsx
│   ├── summary_combined.csv
│   ├── summary_combined_reduced.csv
│   ├── summary_cross_artist.csv
│   ├── summary_visual.csv
│   ├── summary_visual_cross_artist.csv
│   └── with_images_final.csv
│
├── embeddings/                 # Embedding vectors and similarity JSON/NPY outputs
│   ├── all_similarities.json
│   ├── ...
│   └── statistics_visual_cross_artist.json
│
├── images/                     # Artwork images (e.g., 466.jpg, 470.jpg, ...)
│
├── metadata/                   # Matching JSON metadata files (e.g., 466.json, 470.json, ...)
│
├── python/                     # Python scripts for data preprocessing and embeddings
│   ├── add_images_urls_better.py
│   ├── add_links_to_objects.py
│   ├── combine_features.py
│   ├── download_met_images.py
│   ├── generate_all_similarities.py
│   ├── generate_embeddings.py
│   ├── process_metadata.py
│   ├── requirements.txt
│   └── run_pipeline.py
│
├── utils/                      # Utility scripts
│   └── gcs_image_upload.py     # Script for uploading images to Google Cloud
│
├── web/
│   ├── lib/                    # External JavaScript libraries (D3, topojson, etc.)
│   │   ├── d3.v5.min.js
│   │   ├── d3-dsv.min.js
│   │   ├── d3-geo-projection.v2.min.js
│   │   ├── d3-legend.min.js
│   │   ├── d3-tip.min.js
│   │   └── topojson.v2.min.js
│   │
│   ├── pages/                  # Website HTML pages
│   │   ├── about.html          # Project overview and team/methodology description
│   │   └── art.html            # Main visualization page with D3 similarity graph
│   │
│   ├── scripts/                # JavaScript logic and visualization handling
│   │   ├── main.js             # Global site navigation and layout scripts
│   │   └── art.js              # Helper JS for detailed artwork view interactions
│   │
│   └── styles/                 # CSS for consistent styling
│   │   └── main.css            # Shared site-wide styles
│
├── index.html                  # Splash/home page for the Visualizing Art website
│
└── README.md
```

</p>
</details>

**Convention:** image and JSON share the same numeric ID.  
Example: `images/466.jpg` ↔ `metadata/466.json`.

## ▶️ Run it locally

How to run a server locally and then visit our website.  Run the follwoing command inside of the `.../Visualizing_Art` folder.

**Python:**
```bash
python -m http.server 5000
```

Next, open up our webiste at:

```html
http://localhost:5000/
```