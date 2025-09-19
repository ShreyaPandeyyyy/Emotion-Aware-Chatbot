# 🎭 Emotion-Aware Chatbot  

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/streamlit-deployed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, **text-based emotion classifier** that predicts six emotions from user input and visualizes probabilities in real time.  

🔗 **Live Demo:** [Click here to try the app](https://emotion-aware-chatbot-inogm4xnwu27aqvgqmpt8a.streamlit.app/)  

---

## 📑 Table of Contents
- [Features](#-features)
- [Preview](#-preview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Demo](#-demo)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features
- Predicts **six emotions**: *joy, sadness, anger, fear, surprise, neutral*  
- Displays **probability bar chart** for transparent, interpretable output  
- Clean, responsive **Streamlit UI**  
- Fast **scikit-learn pipeline** (vectorizer + classifier)  
- Ships with ready-to-use trained models:  
  - `models/vectorizer.pkl`  
  - `models/classifier.pkl`  

---

## 🖼️ Preview
![App Screenshot](assets/screenshot.png)  
*(Add your own screenshot in an `assets/` folder and update this link.)*

---

## 🛠️ Tech Stack
- **Python 3.9+**  
- **scikit-learn** – model training & classification  
- **Streamlit** – interactive web interface  
- **pandas, numpy, matplotlib** – preprocessing & visualization  

---

## 📂 Project Structure
