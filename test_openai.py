# test_openai.py - Test OpenAI integration separately
"""
Test script for OpenAI integration
Run this in your IDE to debug the OpenAI connection
"""

import pandas as pd
import numpy as np


def test_openai_connection():
    """Test OpenAI API connection and functionality"""

    # Get API key
    api_key = input("Enter your OpenAI API key: ").strip()

    if not api_key:
        print("❌ No API key provided")
        return False

    print(f"🔑 API Key length: {len(api_key)} characters")
    print(f"🔑 API Key starts with: {api_key[:10]}...")

    # Create sample customer data
    print("\n📊 Creating sample customer data...")
    np.random.seed(42)
    n_customers = 100

    sample_data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(n_customers)],
        'age': np.random.randint(18, 70, n_customers),
        'gender': np.random.choice(['M', 'F'], n_customers),
        'total_spent': np.random.lognormal(6, 1, n_customers),
        'monthly_visits': np.random.poisson(8, n_customers),
        'satisfaction_score': np.random.normal(3.8, 0.8, n_customers),
        'churn': np.random.choice([0, 1], n_customers, p=[0.75, 0.25])
    }

    # Clean up the data
    sample_data['satisfaction_score'] = np.clip(sample_data['satisfaction_score'], 1, 5)
    sample_data['total_spent'] = np.round(sample_data['total_spent'], 2)

    df = pd.DataFrame(sample_data)
    print(f"✅ Created dataset with {len(df)} customers")
    print(f"   Churn rate: {df['churn'].mean() * 100:.1f}%")
    print(f"   Avg satisfaction: {df['satisfaction_score'].mean():.2f}")

    # Test OpenAI integration
    print("\n🧠 Testing OpenAI integration...")

    # Method 1: Try new OpenAI v1.x
    try:
        from openai import OpenAI
        print("✅ OpenAI v1.x library detected")

        # Create client
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client created successfully")

        # Prepare data summary
        data_summary = f"""
        Dataset: {len(df)} customers
        Churn rate: {df['churn'].mean() * 100:.1f}%
        Average age: {df['age'].mean():.1f}
        Average spending: ${df['total_spent'].mean():.2f}
        Average satisfaction: {df['satisfaction_score'].mean():.2f}/5.0
        """

        question = "What are the key insights about customer churn in this data?"

        prompt = f"""
        You are a customer analytics expert. Analyze this data and provide insights:

        {data_summary}

        Question: {question}

        Provide a brief analysis with 2-3 key insights and recommendations.
        """

        print("🚀 Sending request to OpenAI...")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a customer analytics expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )

        print("✅ OpenAI request successful!")
        print("\n" + "=" * 50)
        print("🤖 AI RESPONSE:")
        print("=" * 50)
        print(response.choices[0].message.content)
        print("=" * 50)

        return True

    except ImportError:
        print("❌ OpenAI v1.x not available, trying alternative...")
        return test_with_requests(api_key, df)

    except Exception as e:
        print(f"❌ OpenAI v1.x error: {str(e)}")
        print("🔄 Trying alternative method...")
        return test_with_requests(api_key, df)


def test_with_requests(api_key, df):
    """Test OpenAI using direct HTTP requests"""
    try:
        import requests
        import json

        print("🌐 Testing with direct HTTP requests...")

        # Prepare data
        data_summary = f"""
        Dataset: {len(df)} customers
        Churn rate: {df['churn'].mean() * 100:.1f}%
        Average age: {df['age'].mean():.1f}
        Average spending: ${df['total_spent'].mean():.2f}
        """

        # API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a customer analytics expert."},
                {"role": "user", "content": f"Analyze this customer data and provide insights: {data_summary}"}
            ],
            "max_tokens": 200,
            "temperature": 0.1
        }

        print("🚀 Sending HTTP request to OpenAI...")

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )

        print(f"📡 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("✅ HTTP request successful!")
            print("\n" + "=" * 50)
            print("🤖 AI RESPONSE:")
            print("=" * 50)
            print(result['choices'][0]['message']['content'])
            print("=" * 50)
            return True
        else:
            print(f"❌ HTTP Error {response.status_code}:")
            print(response.text)
            return False

    except Exception as e:
        print(f"❌ HTTP request error: {str(e)}")
        return False


def test_api_key_validity(api_key):
    """Quick test to validate API key format"""
    print("\n🔍 Validating API key format...")

    if not api_key:
        print("❌ Empty API key")
        return False

    if not api_key.startswith('sk-'):
        print("❌ API key should start with 'sk-'")
        return False

    if len(api_key) < 40:
        print("❌ API key seems too short")
        return False

    print("✅ API key format looks valid")
    return True


def main():
    """Main test function"""
    print("🧪 OpenAI Integration Test")
    print("=" * 30)

    # Get and validate API key
    api_key = input("Enter your OpenAI API key: ").strip()

    if not test_api_key_validity(api_key):
        print("\n❌ Invalid API key format. Please check and try again.")
        return

    # Test connection
    success = test_openai_connection()

    if success:
        print("\n🎉 SUCCESS! OpenAI integration is working correctly.")
        print("💡 You can now use this code in your Streamlit app.")
    else:
        print("\n❌ FAILED! Please check:")
        print("   1. API key is correct")
        print("   2. You have OpenAI credits available")
        print("   3. Internet connection is working")
        print("   4. Try: pip install openai --upgrade")


if __name__ == "__main__":
    main()