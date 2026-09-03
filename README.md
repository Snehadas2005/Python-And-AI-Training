# Employee Sentiment Analysis and Flight Risk Prediction

## Project Overview
This project performs an end-to-end exploratory data analysis and sentiment classification on employee internal communications using Python. The primary objective is to evaluate employee engagement trends, compute monthly sentiment rankings, detect flight risk indicators, and model monthly sentiment trajectories using statistical machine learning techniques.

## Key Features & Deliverables
* Sentiment Labeling: Analyzes message body content and classifies sentiment as Positive, Negative, or Neutral using Natural Language Processing via TextBlob.
* Exploratory Data Analysis (EDA): Checks data integrity, visualizes sentiment distribution, and plots long-term monthly sentiment trends.
* Monthly Scoring & Ranking: Calculates aggregate sentiment scores per employee on a monthly basis (+1 for Positive, -1 for Negative, 0 for Neutral) and ranks employees each month.
* Flight Risk Detection: Flags high-risk employees who send 4 or more negative messages within any single month.
* Predictive Modeling: Fits a Linear Regression model using scikit-learn to model and evaluate average monthly sentiment trends over time.

## Repository File Structure
* Final_LLM_Assessment.ipynb: Main Jupyter Notebook containing the full pipeline execution, comments, visualizations, and modeling steps.
* processed_sentiment_data.csv: Cleaned dataset augmented with sentiment polarity categories and numerical sentiment scores.
* monthly_employee_rankings.csv: Monthly aggregated table showing scores, total message counts, rank, and flight risk flags per employee.
* test.csv: Raw input dataset containing employee communications.

## Prerequisites & Dependencies
The project requires Python 3.8+ and the following libraries:
* pandas
* numpy
* matplotlib
* seaborn
* textblob
* scikit-learn

Install all required packages via pip:
pip install pandas numpy matplotlib seaborn textblob scikit-learn

## Setup and Usage Instructions
1. Clone the repository:
   git clone https://github.com/Snehadas2005/Python-And-AI-Training.git
   cd Python-And-AI-Training

2. Launch Jupyter Notebook or Google Colab:
   jupyter notebook Final_LLM_Assessment.ipynb

3. Run all cells sequentially to execute data loading, sentiment scoring, visualization, and regression modeling.

## Methodology & Findings
* Sentiment Distribution: Messages skew towards positive and neutral categories overall, with specific clusters of negative sentiment occurring in peak project periods.
* Flight Risk Candidates: A total of 9 flight risk instances were identified across the dataset where employees crossed the threshold of 4 or more negative messages in a single month.
* Model Evaluation: The Linear Regression model evaluates average monthly sentiment trends using temporal indices and message volume as predictors.
