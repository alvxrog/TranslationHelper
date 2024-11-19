import os
from googletrans import Translator, LANGCODES, LANGUAGES
import deepl
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Initialize translators
google_translator = Translator()
deepl_translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))

def translate_file(input_path:str, language:str):
    # Get the directory and filename
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    output_path = os.path.join(directory, f"translated_{filename}")
    
    with open(input_path, 'r', encoding='utf-8') as input_file, open(output_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            # Strip whitespace and skip empty lines
            original_text = line.strip()
            if not original_text:
                continue
            
            # Translate using Google Translate
            google_translation = google_translator.translate(original_text, dest=language.lower()).text
            
            # Translate using DeepL
            language_deepl: deepl.Language
            # deepl language string preprocessing
            if language.lower() == 'en':
                # hardcoded EN_GB translation
                language_deepl = deepl.Language.ENGLISH_AMERICAN
            elif language.lower() == 'es':
                language_deepl = deepl.Language.SPANISH

            deepl_translation = deepl_translator.translate_text(original_text, target_lang=language_deepl).text
            
            # Write to output file
            output_file.write(f"{original_text}\n")
            output_file.write(f"Translation 1. {google_translation}\n")
            output_file.write(f"Translation 2. {deepl_translation}\n\n")
    
    print(f"Translations saved to {output_path}")

def file_path_validation(path):
  if not os.path.isfile(path):
    raise argparse.ArgumentTypeError(f"File '{path}' doesn't exist")
  return path

def create_parser():
    parser = argparse.ArgumentParser(description="Translates all sentences from an input file using Google translate and DeppL. Outputs a new file with the specified translations")
    
    parser.add_argument("file_path", 
                        type=file_path_validation,
                        help="Path to the input text file")
    
    parser.add_argument("language", 
                        type=str, 
                        choices=["en", "es"], 
                        help="Language code: 'en' for english, 'es' for spanish")
    
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    print(f"Input file path: {args.file_path}")
    print(f"Language selected: {args.language}")

    translate_file(args.file_path, args.language)

if __name__ == '__main__':
    main()