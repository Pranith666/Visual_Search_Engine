# Visual_Search_Engine
Architected and implemented a dual-pipeline visual search engine, enabling users to search via local
image uploads or Google Images. The local pipeline utilized ResNet-50 for feature extraction, storing and indexing embeddings
in FAISS for rapid retrieval of visually similar products with associated image and description. The Google Search pipeline
integrated Gemini to generate descriptive queries from Google Images, leveraging the Google Search Engine API to fetch and
present relevant images and metadata.
