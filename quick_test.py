# quick_test.py - Simple test to verify OpenAI connection
"""
Quick test script to verify OpenAI and LangChain are working properly
Run this before the main app to debug any issues
"""


def test_openai_basic():
    """Test basic OpenAI connection"""
    try:
        from openai import OpenAI

        # Get API key
        api_key = input("Enter your OpenAI API key: ").strip()

        if not api_key:
            print("❌ No API key provided")
            return False

        # Test basic OpenAI connection
        print("🧪 Testing OpenAI connection...")

        client = OpenAI(
            api_key=api_key,
            timeout=30.0,
            max_retries=2
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'Hello from OpenAI!'"}
            ],
            max_tokens=50,
            temperature=0
        )

        result = response.choices[0].message.content
        print(f"✅ OpenAI Success: {result}")
        return True

    except Exception as e:
        print(f"❌ OpenAI Error: {str(e)}")
        return False


def test_langchain_basic():
    """Test basic LangChain setup"""
    try:
        from langchain_openai import OpenAI

        # Get API key
        api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()

        if not api_key:
            print("⏭️ Skipping LangChain test (no API key)")
            return True

        print("🧪 Testing LangChain...")

        # Test LangChain OpenAI
        llm = OpenAI(
            temperature=0,
            openai_api_key=api_key,
            model_name="gpt-3.5-turbo-instruct"
        )

        result = llm.invoke("Say 'Hello from LangChain!'")
        print(f"✅ LangChain Success: {result}")
        return True

    except Exception as e:
        print(f"❌ LangChain Error: {str(e)}")
        return False


def test_imports():
    """Test all required imports"""
    print("🔧 Testing imports...")

    packages = {
        'openai': 'OpenAI API',
        'langchain': 'LangChain Core',
        'langchain_openai': 'LangChain OpenAI',
        'langchain_experimental': 'LangChain Experimental',
        'pandas': 'Pandas',
        'streamlit': 'Streamlit'
    }

    success_count = 0

    for package, description in packages.items():
        try:
            __import__(package)
            print(f"✅ {description}: Available")
            success_count += 1
        except ImportError:
            print(f"❌ {description}: Missing")

    print(f"\n📊 Import Results: {success_count}/{len(packages)} packages available")
    return success_count == len(packages)


def main():
    """Run all tests"""
    print("🚀 Quick Test Suite for AI Customer Intelligence Agent")
    print("=" * 60)

    # Test 1: Imports
    imports_ok = test_imports()

    if not imports_ok:
        print("\n❌ Some packages are missing. Install them with:")
        print("pip install openai langchain langchain-openai langchain-experimental pandas streamlit")
        return

    print("\n" + "=" * 60)

    # Test 2: OpenAI
    openai_ok = test_openai_basic()

    print("\n" + "=" * 60)

    # Test 3: LangChain
    langchain_ok = test_langchain_basic()

    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   Imports: {'✅' if imports_ok else '❌'}")
    print(f"   OpenAI: {'✅' if openai_ok else '❌'}")
    print(f"   LangChain: {'✅' if langchain_ok else '❌'}")

    if imports_ok and openai_ok:
        print("\n🎉 You're ready to run the main app!")
        print("Run: streamlit run app.py")
    else:
        print("\n🔧 Fix the issues above before running the main app")


if __name__ == "__main__":
    main()