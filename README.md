# LangGraph +OpenAI AI Customer Intelligence Agent

A comprehensive customer analytics platform built with Streamlit that provides automated data generation, advanced analytics, and AI-powered insights for e-commerce businesses.

## Features

### Data Generation & Visualization
- Automatic generation of realistic e-commerce customer datasets
- Interactive dashboards with customer metrics and KPIs
- Advanced analytics including distribution analysis, churn analysis, customer segmentation, and correlation analysis
- Real-time data refresh capability

### AI-Powered Analysis
- **Direct OpenAI Integration**: Fast, comprehensive analysis using GPT-4o
- **LangGraph Workflow**: Multi-step analysis pipeline with planning, execution, validation, and synthesis
- Natural language query interface for customer insights
- Automated business intelligence reporting

### Analytics Capabilities
- Customer churn prediction and analysis
- Behavioral segmentation and pattern identification
- Statistical correlation analysis
- Business recommendations and strategic insights

## Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-customer-intelligence
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your OpenAI API key:
   - Add your API key to Streamlit secrets, or
   - Set the `OPENAI_API_KEY` environment variable, or
   - Update the hardcoded key in the configuration (for testing only)

4. Run the application:
```bash
streamlit run app.py
```

5. Access the application at `http://localhost:8501`

## Usage

### Data Overview Tab
- View automatically generated customer data
- Monitor key business metrics (churn rate, average spending, satisfaction scores)
- Generate fresh datasets using the "New Data" button

### Analytics Tab
- **Distribution Analysis**: Customer demographics and spending patterns
- **Churn Analysis**: Factors contributing to customer churn
- **Customer Segments**: Value-based customer segmentation
- **Correlation Analysis**: Statistical relationships between variables

### AI Chat Tab
- Configure analysis method (Direct OpenAI or LangChain Agent)
- Ask natural language questions about customer data
- Get detailed analysis with business recommendations
- View step-by-step workflow execution for LangChain analysis

### Sample Questions
- "Why are customers churning?"
- "What customer segments exist?"
- "Which customers are at risk?"
- "What drives customer value?"
- "Show satisfaction insights"
- "Analyze spending patterns"

## File Structure

```
├── app.py                 # Main Streamlit application
├── ai_analyzer.py         # AI analysis engine with OpenAI and LangGraph integration
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Configuration

### OpenAI API Key
The application requires an OpenAI API key for AI analysis features. Configure it using one of these methods:

1. **Streamlit Secrets** (Recommended for production):
   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY = "sk-..."
   ```

2. **Environment Variable**:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

3. **Direct Configuration** (Development only):
   Update the hardcoded key in `ai_analyzer.py` and `app.py`

### Analysis Methods

**Direct OpenAI Analysis**
- Fast response times
- Clean, formatted output
- Professional consulting-grade insights
- Recommended for most use cases

**LangChain Agent Analysis**
- Multi-step workflow with detailed tracking
- Advanced statistical analysis capabilities
- Step-by-step execution monitoring
- Comprehensive business intelligence reports

## Dependencies

### Core Framework
- streamlit>=1.28.0
- pandas>=2.0.0
- numpy>=1.24.0

### AI/ML Libraries
- openai>=1.0.0
- langchain>=0.1.0
- langchain-openai>=0.0.5
- langchain-experimental>=0.0.50
- langgraph>=0.0.40
- scikit-learn>=1.3.0

### Visualization
- plotly>=5.17.0
- matplotlib>=3.7.0
- seaborn>=0.12.0

### Statistical Analysis
- scipy>=1.10.0

See `requirements.txt` for complete dependency list.

## Development

### Running in Development Mode
```bash
streamlit run app.py --server.port 8501 --server.address localhost
```

### Testing AI Connection
```python
import ai_analyzer
result = ai_analyzer.test_ai_connection('')
print(result)
```

## Technical Architecture

### Data Flow
1. **Data Generation**: Automatic creation of realistic e-commerce datasets
2. **Data Processing**: Statistical analysis and feature engineering
3. **AI Analysis**: Natural language processing of business questions
4. **Insight Generation**: Automated business intelligence reporting
5. **Visualization**: Interactive charts and dashboards

### AI Integration
- **OpenAI GPT-4o**: Primary language model for analysis
- **LangGraph**: Multi-agent workflow orchestration
- **Pandas Agent**: Automated data exploration and analysis
- **Statistical Computing**: Advanced analytics with scipy and scikit-learn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues, questions, or contributions, please open an issue on GitHub. 
