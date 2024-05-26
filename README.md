## End to end Machine Learning Project

# ML Project: Student Score Prediction

## Project Overview

This project is an end-to-end data science application aimed at predicting students' scores based on various demographic and educational features. The project leverages machine learning techniques and follows industry best practices, including structured folder organization, custom loggers, exception handling, and a CI/CD pipeline.

## Objectives

- Develop a machine learning model to predict student scores in math, reading, and writing.
- Implement a robust data processing and model training pipeline.
- Ensure code quality and maintainability through structured project organization, custom logging, and exception handling.
- Automate the pipeline using continuous integration and continuous deployment (CI/CD) practices.

## Features

The dataset includes the following features:

- **Gender**: Sex of the students (Male/Female)
- **Race/Ethnicity**: Ethnicity of the students (Group A, B, C, D, E)
- **Parental Level of Education**: The highest level of education attained by the parents (bachelor's degree, some college, master's degree, associate's degree, high school)
- **Lunch**: Type of lunch before the test (standard or free/reduced)
- **Test Preparation Course**: Completion status of the test preparation course (complete or not complete)
- **Math Score**: Score in math
- **Reading Score**: Score in reading
- **Writing Score**: Score in writing

## Key Components

1. **Data Ingestion**: Loading and preprocessing raw data.
2. **Data Transformation**: Feature engineering and data transformation.
3. **Model Training**: Training machine learning models to predict student scores.
4. **Logging**: Custom logging to track the pipeline execution.
5. **Exception Handling**: Custom exceptions to handle errors gracefully.
6. **CI/CD Pipeline**: Automated pipeline for continuous integration and deployment.

## Technologies Used

- **Python**: Core programming language.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning library.
- **Logging**: Python's logging module for custom logging.
- **GitHub Actions**: CI/CD pipeline implementation.

## Installation and Usage

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/MLProject.git
   cd MLProject
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate # On Windows use venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the data ingestion script:

   ```
   python src/components/data_ingestion.py
   ```

5. Run the data transformation script:

   ```
   python src/components/data_transformation.py
   ```

6. Train the model:

   ```
   python src/components/model_trainer.py
   ```

## CI/CD Pipeline

The project includes a CI/CD pipeline set up using GitHub Actions. This ensures that any changes to the codebase are automatically tested and deployed, maintaining the integrity and functionality of the project.

## Conclusion

This project demonstrates the complete lifecycle of a data science project, from data ingestion and preprocessing to model training and deployment. By following industry-standard practices and leveraging a CI/CD pipeline, the project ensures reliability, maintainability, and scalability. This setup can serve as a solid foundation for more complex machine learning applications in real-world scenarios.
