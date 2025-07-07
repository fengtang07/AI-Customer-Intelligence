#!/bin/bash

# AI Customer Intelligence Agent - Development Setup Script

echo "🚀 Setting up AI Customer Intelligence Agent"
echo "============================================"

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 Python version: $python_version"

if [[ "$python_version" < "3.8" ]]; then
    echo "❌ Python 3.8+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,exports}
mkdir -p logs
mkdir -p static/{css,js,images}

# Copy sample .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔑 Creating .env file..."
    cp .env .env
    echo "⚠️  Please edit .env file and add your OpenAI API key"
fi

# Run initial setup
echo "🎯 Running initial configuration check..."
python -c "from config.config import Config; Config.validate(); print('✅ Configuration valid')" 2>/dev/null || echo "⚠️  Please configure your .env file"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Run the application: streamlit run app.py"
echo "3. Open your browser to http://localhost:8501"
echo ""
echo "💡 For development:"
echo "   source venv/bin/activate  # Activate virtual environment"
echo "   streamlit run app.py      # Run the application"
