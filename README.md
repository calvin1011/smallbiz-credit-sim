# Small Business Credit Score Simulator

An AI-powered web app that simulates how small business decisions affect their credit score. Built using HP AI Studio and MLflow, this project showcases real-time credit scoring powered by a Random Forest regression model trained on synthetic data.

---

## Features
- **Interactive web UI** for entering financial metrics
- **ML model** trained on synthetic credit data (scikit-learn)
- **MLflow integration** for model tracking and versioning
- **Swagger-ready deployment** for seamless API access

---

## How to Run
1. Set up the environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r ai_model/requirements.txt
```

2. Train and log the model:
```bash
python ai_model/model.py
```

3. Launch in HP AI Studio to access Swagger and UI via `demo/`.

---

## Model Details
- **Algorithm**: RandomForestRegressor
- **Features**: Payment history, credit utilization, revenue, time in business, debt, industry risk
- **Target**: Simulated credit score

---

## Example Use Case
A small business enters their financial metrics and receives a simulated credit score, along with guidance on which factors may be impacting it.

---

## Future Improvements
- Integrate GPT agent for financial advice
- Use real-world anonymized credit datasets
- Add secure user authentication
- Enable document upload with OCR + NLP for financial data extraction

---

## 🛠 Tech Stack
- Python, scikit-learn, pandas
- MLflow (HP AI Studio)
- HTML/CSS for frontend demo

