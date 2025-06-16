# 📊 CSE Analyzer

A full-stack analytical application built with **Streamlit**, designed to analyze and visualize stock market data for companies listed on the **Colombo Stock Exchange (CSE)**. The system supports both **cloud-based data ingestion from Azure Cosmos DB** and **fallback to local storage** for offline access.

---

## 🚀 Features

* 🔁 **ETL Pipeline** for automated data extraction from Cosmos DB or local files.
* 📉 **Forecasting models** (LSTM, ARIMA, etc.) for stock price prediction.
* 📊 **Value at Risk (VaR)** analysis using Historical, Parametric, and Monte Carlo methods.
* ⚙️ **Custom model forecast comparison** and testing for forecasting under dates.
* 🧠 **Model evaluation and visualization**.
* 🌐 **Streamlit-based Web UI** (`Home.py`) for interacting with the system.
* 💾 Local file support for offline/backup analysis.
* 🧹 Utility scripts for CSV conversion, object cloning, and comparison.

---

## 🗂️ Directory Structure

```
CSE-Analyzer/
│
├── Deprecated/                   # Old or legacy scripts
│   ├── CosmosObjects.py
│   └── ...
│
├── pages/                        # Streamlit page routing directory
│
├── ETL.py                        # Main ETL logic wrapper
├── ETL_PIPELINE.py              # Detailed ETL flow for Cosmos or local
├── Home.py                       # Streamlit app entrypoint
├── VAR.py                        # Value at Risk calculation methods
├── model.py                      # Unified model framework
├── model for one esset.py        # Model test for individual assets
├── Forcast_model.py              # Forecasting model logic
├── Forcast_price.py              # Model to forecast and display price
├── forcate comparisson.py        # Compare multiple forecasting models
├── Localdata2csv.py              # Convert local DB to CSV
├── local2spcsv.py                # Special CSV formatter
├── clonefilesto local.py         # Utility to copy cloud files to local
├── DownloadFiles2Local.py        # Cosmos -> Local downloader
├── background.jpg                # UI background asset
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore settings
```

---




## 🔐 Environment Configuration (`.env`)

Before running the application, you must create a `.env` file in the root directory to securely store your **Azure Cosmos DB** connection details.

### 🔧 Required Variables

```env
COSMOS_ENDPOINT="https://<your-cosmos-account>.documents.azure.com:443/"
COSMOS_KEY="<your-primary-or-secondary-key>"
COSMOS_DATABASE_NAME="db_conversation_history"
CONTAINER_NAME="csetest"
```

### 📌 Description of Each Variable:

| Variable               | Description                                                             |
| ---------------------- | ----------------------------------------------------------------------- |
| `COSMOS_ENDPOINT`      | The URL endpoint of your Azure Cosmos DB account.                       |
| `COSMOS_KEY`           | The primary or secondary key used to authenticate with Cosmos DB.       |
| `COSMOS_DATABASE_NAME` | The name of the Cosmos DB database (e.g., `db_conversation_history`).   |
| `CONTAINER_NAME`       | The name of the container (collection) used to store and retrieve data. |

### ⚠️ Security Tips:

* **Do not commit** your `.env` file to version control (already covered by `.gitignore`).
* Rotate keys periodically for better security.
* Use **Azure Key Vault** in production deployments for enhanced credential management.

---

## 🧪 Models and Analytics

### ✅ Forecasting

* Uses historical stock data to train and test models.
* Includes support for:

  * LSTM
    
* Generates CSVs and plots for trends.

### ✅ Value at Risk (VaR)

Implemented methods:

* **Historical Simulation**
* **Parametric (Variance-Covariance)**
* **Monte Carlo Simulation**

Each method produces graphical and numerical summaries for risk comparison.

---

## 🌐 Streamlit UI (`Home.py`)

* Enter company symbol (e.g., `APPL.N000`) to load data.
* Access detailed pages under the `pages/` directory for:

  * Forecasting
  * Risk Analysis (VaR)
  * Forecast Evaluation
  * Trend Analysis
  * Portfolio Simulations (future)

---

## 🔌 Data Ingestion

### 1. From Cosmos DB

* ETL scripts use Cosmos APIs to fetch latest stock/market data.
* Scripts: `CosmosObjects.py`, `DownloadFiles2Local.py`, `ETL_PIPELINE.py`

### 2. From Local Storage

* Fallback for offline environments.
* Scripts: `Localdata2csv.py`, `local2spcsv.py`, `clonefilesto local.py`

---


## 🔍 Sample Screenshot – Cosmos DB Integration

This screenshot shows a sample structure of data stored in **Azure Cosmos DB**, which the ETL pipeline extracts and processes.

[![Screenshot of Cosmos DB](https://github.com/sachithbrowns/CSE-ANALYSER/blob/main/Screenshot%20of%20Exampls/ss%20of%20Cosmos%20DB.png)](https://github.com/sachithbrowns/CSE-ANALYSER/blob/main/Screenshot%20of%20Exampls/ss%20of%20Cosmos%20DB.png)

> 📌 *Click the image to view it in full size.*

---




## 📦 Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/sachithbrowns/CSE-ANALYSER.git
cd cse-analyzer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run Home.py
```

---

## 📌 Future Plans

* ✅ Add real-time market feeds
* ✅ Automate daily ETL with cron/Azure Functions
* ✅ Deploy on Azure App Service with Key Vault integration
* ✅ Add ESG/Carbon footprint analysis (GreenLink integration)
* ✅ Integrate chatbot assistant

---

## 👨‍💻 Author

Yamannage Sachith Nimesh

Specializing in Data Science & Risk Management

📫 [LinkedIn](https://www.linkedin.com/in/sachith-y-29a336175)  | 🌱 \[IEEE Member: 98467655]

---
