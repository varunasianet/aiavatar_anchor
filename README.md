

# SEO Analysis Tool

A tool for analyzing websites and generating comprehensive SEO insights using AI and natural language processing.

### Features

* Web Scraping: Extracts content from websites including meta tags, heading tags, and full content
* Content Analysis: Analyzes website content using advanced NLP techniques
* Keyword Extraction: Identifies key terms and phrases using multiple methods (KeyBERT, YAKE)
* Visual Analytics: Generates visual representations of keyword relationships and clusters
* AI-Powered SEO Analysis: Uses OpenAI's GPT-3 to provide detailed SEO recommendations
* Full Site Support: Can analyze both single pages and entire websites
* Interactive Interface: Built with Gradio for easy usage

### Prerequisites

```python
# Core dependencies
requests
beautifulsoup4
nltk
gensim
pandas
matplotlib
seaborn
gradio
openai
google-search-results
pytrends
scikit-learn
numpy
```

### Installation

1. Clone the repository
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY='your-api-key'
```

### Usage

Run the application:
```bash
python seo_analyzer.py
```

The tool provides a web interface where you can:
1. Enter a URL to analyze
2. Choose between single page or full site analysis
3. View detailed analysis including:
   - Meta tags
   - Heading structure
   - Top keywords
   - Keyword clusters
   - Visual representations
   - AI-generated SEO recommendations

### Key Components

* `scrape_article()`: Extracts content from web pages
* `extract_keywords()`: Identifies key terms using multiple algorithms
* `analyze_website()`: Main analysis function
* `visualize_clusters_plot()`: Creates visual keyword clusters
* `analyse_SEO()`: Generates AI-powered SEO recommendations

### Output

The tool generates several types of analysis:
1. Meta Tags Analysis
2. Heading Tags Structure
3. Top 10 Keywords
4. Keyword Cluster Table
5. Cluster Visualization Plot
6. Keyword Frequency Plot
7. Comprehensive SEO Analysis

### Best Practices

* Use full site analysis for comprehensive insights
* Consider rate limiting for large sites
* Review both automated and AI-generated insights
* Export results for documentation

