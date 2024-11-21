# Installation
To use this script:
1. Install Python >=3.9
2. Install the required libraries
```
pip install googletrans==3.1.0a0 deepl python-dotenv
```
3. Install the PyTorch GPU compatible releases
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
If using Anaconda
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
4. Create a file named ".env", in the same directory as the script, and add a line with:
```
DEEPL_API_KEY=your_api_key_here
```
Where your_api_key is your DeepL API Key. We have an appendix explaining how to get yours if you dont have one yet.

# Usage
```
usage: main.py [-h] file_path {en,es}
```
`file_path (required)`: path to the input text file, which should be formatted to have one sentence per line, and no empty/whitespaced lines

`language (required) {en, es}`: destination language key to translate the input file sentences to. `en` will translate them to english, `es` to spanish 

To use the script, just open a terminal on the folder the script is located and create a file named `input.txt` (or any name), fill it with the desired sentences, and on the terminal type:
```
python main.py input.txt es
```
You should get a message when the translation is done with the translated outputs file name

# Appendix: Getting your DeepL API Key 
1. Go to the [DeepL landing page](https://www.deepl.com/es/pro-api)
2. Create an account and subscribe to their free (or paid) tier
3. Go to your [account API keys](https://www.deepl.com/es/your-account/keys) section and grab your API key by copying it
4. Paste it on the .env file following the described format 