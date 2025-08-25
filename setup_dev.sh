# Development setup script for MCP RAG Server
# Run this after installing requirements.txt

echo "Setting up MCP RAG Server development environment..."

# Create necessary directories
mkdir -p data/{raw,processed,embeddings}
mkdir -p logs
mkdir -p models/cache

# Set up environment file
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
else
    echo ".env file already exists"
fi

# Download a sample document for testing
if [ ! -f data/raw/sample_document.txt ]; then
    echo "Creating sample document for testing..."
    cat > data/raw/sample_document.txt << 'EOF'
# Machine Learning Fundamentals

## Chapter 1: Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.

### 1.1 Types of Machine Learning

There are three main types:
1. Supervised Learning
2. Unsupervised Learning  
3. Reinforcement Learning

### 1.2 Mathematical Foundations

The basic linear regression equation is:
$$y = \beta_0 + \beta_1 x + \epsilon$$

For multiple variables:
$$y = X\beta + \epsilon$$

Where $X$ is the design matrix and $\beta$ is the parameter vector.

## Chapter 2: Neural Networks

Neural networks are inspired by biological neural networks.

### 2.1 Perceptron

The perceptron activation is:
$$y = f(\sum_{i=1}^n w_i x_i + b)$$

### 2.2 Backpropagation

The gradient computation uses:
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial w}$$

## Bibliography

1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
EOF
    echo "Sample document created at data/raw/sample_document.txt"
fi

# Test basic functionality
echo "Testing basic functionality..."
python3 main.py process-document data/raw/sample_document.txt
if [ $? -eq 0 ]; then
    echo "✅ Document processing test passed"
else
    echo "❌ Document processing test failed"
fi

echo "Setup complete! You can now:"
echo "1. Process documents: python main.py process-and-embed data/raw/sample_document.txt"
echo "2. Start server: python main.py start-server"
echo "3. Run example: python example_usage.py"
echo "4. Run tests: python -m pytest tests/ -v"
