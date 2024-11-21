import os
from googletrans import Translator, LANGCODES, LANGUAGES
import deepl
from dotenv import load_dotenv
import argparse
from transformers import BertModel, BertTokenizer
import torch

# sentence max length
MAX_LEN = 512

# "spill" possible variations for spill the beans idiom
spill_variations = {"spill", "spills", "spilled", "spilling", "spilt"}
spill_the_beans_idioms = {f"{spill} the beans" for spill in spill_variations}

# Load environment variables
load_dotenv()

# Try to use GPU if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CUDA GPU Warning
if device == 'cpu':
    print(f"{'\nWARNING: Inference could run faster on a GPU. If you have one in your system but are still seeing this message, see the documentation to install a torch implementation that supports CUDA processing\n\n' if device != 0 else ''}")

# Initialize translators
google_translator = Translator()
deepl_translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))

# https://huggingface.co/abdallahashrafx/Bert_idiom_classifier#how-to-get-started-with-the-model
# Load the BERT model and tokenizer
bert_model = BertModel.from_pretrained('abdallahashrafx/Bert_idiom_classifier')
tokenizer = BertTokenizer.from_pretrained('abdallahashrafx/Bert_idiom_classifier')

# Define the IdiomClassifier class
class IdiomClassifier(torch.nn.Module):
    def __init__(self):
        super(IdiomClassifier, self).__init__()
        self.bert = bert_model
        self.drop = torch.nn.Dropout(p=0.4)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)
# Instantiate the model and move it to the GPU
model = IdiomClassifier().to(device)

def classify_spill_the_beans_usage(sentence: str) -> str:
    """
    Classify the usage of "spill the beans" in each sentence as idiomatic or literal.

    Args:
        sentences (str): Sentence in English or Spanish.

    Returns:
        result: The sentence classification
              Possible classifications are 'idiomatic', 'literal' or 'phrase not present'.
    """
    result:str

    # Check if "spill the beans" is present in the (translated) sentence
    if any(idiom in sentence.lower() for idiom in spill_the_beans_idioms):
        encoded_sentence = tokenizer.encode_plus(
        sentence,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
        )

        input_ids = encoded_sentence['input_ids'].to(device)
        attention_mask = encoded_sentence['attention_mask'].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(output)
        prediction = (probs > 0.5).int()

        prob_value = probs.item()

        class_names = ["Literal","Idiom"]
        result = f"{class_names[prediction.item()]}. Confidence: {prob_value}"
    else:
        result = "phrase not present"

    return result

def translate_file(input_path:str, language:str):
    """
    Translate the input file line by line using both Google Translate and DeepL.
    Classify the usage of "spill the beans" in each sentence and shows the expected idiomatic or literal meaning.

    Args:
        input_path (str): Path to the input text file.
        language (str): Target language code ('en' or 'es').

    Returns:
        None
    """
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
            
            # Zero-shot classifier for spill the beans
            classification_result = ""
            classification_result_google = ""
            classification_result_deepl = ""
            if(language.lower() == 'es'):
                classification_result = classify_spill_the_beans_usage(line)
            elif(language.lower() == 'en'):
                classification_result_google = classify_spill_the_beans_usage(google_translation)
                classification_result_deepl = classify_spill_the_beans_usage(deepl_translation)

            # Write to output file
            output_file.write(f"{original_text}\n")
            if(classification_result):
                output_file.write(f"'spill the beans' classification: {classification_result}\n")

            output_file.write(f"Translation 1. {google_translation}\n")
            if(classification_result_google):
                output_file.write(f"'spill the beans' classification: {classification_result_google}\n")

            output_file.write(f"Translation 2. {deepl_translation}\n\n")
            if(classification_result_deepl):
                output_file.write(f"'spill the beans' classification: {classification_result_deepl}\n")

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
    
    print(f"\nInput file path: {args.file_path}")
    print(f"Language selected: {args.language}")

    translate_file(args.file_path, args.language)

if __name__ == '__main__':
    main()