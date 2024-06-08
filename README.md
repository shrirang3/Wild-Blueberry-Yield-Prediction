🧾Description: The dataset used for predictive modelling was generated by the Wild Blueberry Pollination Simulation Model, which is an open-source, spatially-explicit computer simulation program, that enables exploration of how various factors, including plant spatial arrangement, outcrossing and self-pollination, bee species compositions and weather conditions, in isolation and combination, affect pollination efficiency and yield of the wild blueberry agro-ecosystem. The simulation model has been validated by the field observation and experimental data collected in Maine USA and Canadian Maritimes during the last 30 years and now is a useful tool for hypothesis testing and theory development for wild blueberry pollination researches. This simulated data provides researchers who have actual data collected from field observation and those who wants to experiment the potential of machine learning algorithms response to real data and computer simulation modelling generated data as input for crop yield prediction models.

The below mentioned pipeline has been implemented in this project
Data Visualization (Data format is csv)
Feature Engineering(Considers only relevant features)
Modelling(Creating the model and training)
Saving the model (using joblib)
Prediction on data (accuracy calculation, ROC, confusion matrix)
Explainable AI (Using shap library to get insights of model prediction)
Front-End using Streamlit
Deployment on Amazon Web Services(AWS)