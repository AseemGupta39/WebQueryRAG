# WebQueryRAG

WebQueryRAG is a project that allows users to ask questions and get answers based on data scraped from specified websites. It uses Retrieval-Augmented Generation (RAG) to enhance the information available to the language model by retrieving relevant text from a vector database (Chroma).

## Features

- Scrapes data from specified URLs
- Splits and deduplicates the scraped content
- Stores the processed data in a vector database
- Uses RAG to retrieve relevant information and generate answers

## Requirements

- Python 3.10 any version is mandatory to install the requirements

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/AseemGupta39/WebQueryRAG.git
   cd WebQueryRAG
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:

   ```sh
   python main.py
   ```

2. Enter the URLs (comma-separated) when prompted:

   ```
   Enter , separated urls: https://example.com,https://anotherexample.com
   ```

3. Ask questions based on the scraped data:

   ```
   Enter question: What is the main topic of the first website?
   ```

4. To exit the program, type `exit`:
   ```
   Enter question: exit
   ```

## Future Updates

- Clean web interface with deployment.
- Modularize the code.
- Fixing issues from scraping data from websites.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Contact

For any questions or inquiries, please contact [Aseem Gupta](https://github.com/AseemGupta39).

