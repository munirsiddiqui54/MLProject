# ML Project: Student Score Prediction

## Project Overview

This project is an end-to-end data science application aimed at predicting students' scores based on various demographic and educational features. The project leverages machine learning techniques and follows industry best practices, including structured folder organization, custom loggers and custom exception handling.

[Visit Here](https://mlproject-cevz.onrender.com)

## Objectives

- Develop an end to end machine learning model to predict student scores based on various features
- Implement a robust data processing and model training pipeline.
- Ensure code quality and maintainability through structured project organization, custom logging, and exception handling.

## Features

The dataset includes the following features:

- **Gender**: Sex of the students (Male/Female)
- **Race/Ethnicity**: Ethnicity of the students (Group A, B, C, D, E)
- **Parental Level of Education**: The highest level of education attained by the parents (bachelor's degree, some college, master's degree, associate's degree, high school)
- **Lunch**: Type of lunch before the test (standard or free/reduced)
- **Test Preparation Course**: Completion status of the test preparation course (complete or not complete)
- **Reading Score**: Score in reading
- **Writing Score**: Score in writing

## Key Components

1. **Data Ingestion**: Loading and preprocessing raw data.
2. **Data Transformation**: Feature engineering and data transformation.
3. **Model Training**: Training machine learning models to predict student scores.
4. **Logging**: Custom logging to track the pipeline execution.
5. **Exception Handling**: Custom exceptions to handle errors gracefully.

## Installation and Usage

1. Clone the repository:

   ```
   git clone https://github.com/munirsiddiqui54/MLProject.git
   cd MLProject
   ```

2. Create and activate a virtual environment:
   ```
   conda create -n venv python=3.7
   conda activate venv
   ```
3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

4. Train the model:

   ```
   python src/components/model_trainer.py
   ```

## Conclusion

This project demonstrates the complete lifecycle of a data science project, from data ingestion and preprocessing to model training and deployment. By following industry-standard practices, the project ensures reliability, maintainability, and scalability. This setup can serve as a solid foundation for more complex machine learning applications in real-world scenarios.
