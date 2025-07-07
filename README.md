# 🤖 AI Customer Intelligence Agent

A powerful, AI-driven customer analytics platform that transforms customer data into actionable business insights through natural language interaction.

## ✨ Features

- **🧠 AI-Powered Analysis**: Ask questions in natural language and get comprehensive insights
- **📊 Advanced Analytics**: Customer segmentation, churn prediction, and behavioral analysis
- **🎯 Risk Assessment**: Identify high-risk customers and predict churn patterns
- **📈 Predictive Insights**: Revenue forecasting and trend analysis
- **🚀 Strategic Recommendations**: AI-generated action plans and business strategies
- **💻 Modern Web Interface**: Beautiful, responsive Streamlit dashboard
- **📱 Interactive Visualizations**: Dynamic charts and graphs powered by Plotly

## 🏗️ Architecture

```
ai-customer-intelligence/
├── src/
│   ├── agents/           # AI agent core engine
│   ├── analysis/         # Customer analytics modules
│   ├── models/           # ML models and algorithms
│   ├── utils/            # Data processing utilities
│   └── web/              # Streamlit web application
├── data/
│   ├── raw/              # Raw customer data
│   ├── processed/        # Cleaned and processed data
│   └── exports/          # Analysis results and reports
├── config/               # Configuration files
├── tests/                # Unit tests
├── docs/                 # Documentation
└── static/               # Web assets
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

#### Option 1: Automated Setup (Recommended)

**Linux/Mac:**
```bash
chmod +x setup_dev.sh
./setup_dev.sh
```

**Windows:**
```cmd
setup_dev.bat
```

#### Option 2: Manual Setup

1. **Clone and setup environment:**
   ```bash
   git clone https://github.com/yourusername/ai-customer-intelligence.git
   cd ai-customer-intelligence
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:8501`

## 🎯 Usage

### 1. **Data Loading**
- Upload your customer CSV/Excel file, or
- Use the built-in sample dataset for demonstration

### 2. **AI Chat Interface**
Ask natural language questions like:
- "Why are customers churning?"
- "What customer segments do we have?"
- "Which customers are at highest risk?"
- "What actions should we take to improve retention?"

### 3. **Advanced Analytics**
- **Churn Analysis**: Identify patterns and risk factors
- **Customer Segmentation**: Discover customer groups and behaviors
- **Predictive Insights**: Forecast trends and revenue impact
- **Strategic Recommendations**: Get AI-generated action plans

### 4. **Export Results**
- Download high-risk customer lists
- Generate comprehensive action plan reports
- Export analysis results as CSV/Excel

## 📊 Data Format

The platform works with customer datasets containing columns like:

| Column | Description | Required |
|--------|-------------|----------|
| `customer_id` | Unique customer identifier | ✅ Yes |
| `age` | Customer age | 🔶 Recommended |
| `gender` | Customer gender (M/F or 0/1) | 🔶 Recommended |
| `total_spent` | Total customer spending | 🔶 Recommended |
| `monthly_visits` | Average monthly visits/usage | 🔶 Recommended |
| `satisfaction_score` | Customer satisfaction (1-5) | 🔶 Recommended |
| `churn` | Churn status (0/1) | 🔶 Recommended |

### Sample Data
The system automatically generates realistic sample data if no file is provided.

## 🔧 Configuration

Key configuration options in `.env`:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Application Settings
APP_NAME=AI Customer Intelligence Agent
DEBUG=True

# Data Processing
MAX_RECORDS=100000
CACHE_TTL=3600

# UI Preferences
THEME=dark
LAYOUT=wide
```

## 🧪 Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

### Adding New Features

1. **New Analysis Module:**
   - Add to `src/analysis/`
   - Import in `customer_analyzer.py`
   - Add query routing in `customer_agent.py`

2. **New UI Component:**
   - Add to `src/web/streamlit_app.py`
   - Create new render method
   - Add to navigation

## 📈 Performance

- **Recommended**: Up to 100K customer records
- **Memory Usage**: ~2-5 MB per 10K records
- **Response Time**: <5 seconds for most queries
- **API Costs**: ~$0.10-0.50 per complex analysis

## 🔒 Security & Privacy

- **API Keys**: Stored securely in environment variables
- **Data Processing**: All data processing happens locally
- **No Data Storage**: Customer data is not permanently stored
- **Privacy First**: No customer data is sent to external services except OpenAI for insights

## 🚢 Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment

#### Streamlit Cloud
1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Add environment variables
4. Deploy

#### Docker
```bash
docker build -t ai-customer-intelligence .
docker run -p 8501:8501 ai-customer-intelligence
```

#### Heroku/AWS/GCP
See detailed deployment guides in `docs/deployment/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-customer-intelligence/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-customer-intelligence/discussions)

## 🎯 Roadmap

- [ ] Advanced ML models (XGBoost, Neural Networks)
- [ ] Real-time data integration
- [ ] Multi-language support
- [ ] Mobile app
- [ ] API endpoints
- [ ] Advanced visualization dashboards
- [ ] Integration with CRM systems

---

**Built with ❤️ for data-driven customer success**
