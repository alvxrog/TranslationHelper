# Usage
To use this script:
1. Install Python 3.9
2. Install the required libraries
```
pip install googletrans==3.1.0a0 deepl python-dotenv
```
3. Create a file named ".env", in the same directory as the script, and add a line with:
```
DEEPL_API_KEY=your_api_key_here
```
Where your_api_key is your DeepL API Key. You can get yours at ...

# Appendix: Getting your DeepL API Key 
1. Go to the [DeepL landing page](https://www.deepl.com/es/pro-api)
2. Create an account and subscribe to their free (or paid) tier
3. Go to your [account API keys](https://www.deepl.com/es/your-account/keys) section and grab your API key by copying it
4. Paste it on the .env file following the described format 