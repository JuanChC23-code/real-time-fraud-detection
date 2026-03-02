# Real-Time Streaming Fraud Detection System

## Overview
This project implements an event-driven fraud detection architecture using Kafka and machine learning models for real-time inference.

## Architecture

Transaction Event  
→ Kafka (`fraud.transactions.v1`)  
→ Inference Consumer (XGBoost model)  
→ Kafka (`fraud.predictions.v1`)  
→ Monitoring & KPI computation  

## Components

- Event Generator (dataset replay)
- Kafka Topics
- Stream Inference Consumer
- Model Registry
- CI/CD Pipeline
- Monitoring (Prometheus ready)

## Model

Dataset: ULB Credit Card Fraud Detection  
Models:
- Logistic Regression
- XGBoost (production model)

Evaluation Metrics:
- ROC-AUC
- PR-AUC (primary metric)

## Run Locally

Start Kafka (Docker):

```bash
docker-compose up -d
