from fileinput import filename
import os
import shutil
from transformers import pipeline
import PyPDF2
from PIL import Image

print("Starting AI document Origanizer")
print("Creating folders...")

os.makedirs("input_files", exist_ok=True)
os.makedirs("sorted_files/documents", exist_ok=True)
os.makedirs("sorted_files/images", exist_ok=True)

print("Folders created.")

print("Loading AI model...")
classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
print("AI model loaded.")

def read_pdf_text(pdf_path): # This function opens a PDF and gets the text out of it

    try:
        with open(pdf_path, 'rb') as file: #read binary
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + " "
                return text[:500]  # Limit to first 500 characters
    except Exception as error:
        print(f"Couldnt read the pdf: {error}")
        return "" # Return empty text
    
def read_text_file(file_path): # This function opens a .txt file and reads it
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            return text[:500]  # Limit to first 500 characters
    except Exception as error:
        print(f"Couldnt read the text file: {error}")
        return "" # Return empty text

def classify_document(text): # Ask the AI: what kind of document is this?
    if not text:
        return "unknown"
    try:
        result = classifier(text)
        classification = result[0]['label']
        confidence = result[0]['score']
        print(f" AI Classification: {classification} (Confidence: {confidence:.2f})")
        return classification
    
    except Exception as error:
        print(f" AI Couldnt classify the document: {error}")
        return "unknown"
def is_image_file(filename): #check img is img by ext
    extension = filename.lower().split('.')[-1]
    image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
    return extension in image_extensions



def process_single_file(file_path): #Take one file and decide where it should go
    filename = os.path.basename(file_path)
    print(f"Processing file: {filename}")


    if is_image_file(filename):
        destination = os.path.join("sorted_files/images", filename)

        try: 
            shutil.copy2(file_path, destination)
            print(f"File moved to: {destination}")
        except Exception as error:
            print(f"Couldnt move the image file: {error}")
        return
    text = ""

    if filename.endswith(".pdf"):
        text = read_pdf_text(file_path)
    elif filename.endswith(".txt"):
        text = read_text_file(file_path)
    else:
        print(f"AI doesnt know how to read: {filename}")
        return
    
    category = classify_document(text)
    destination = os.path.join("sorted_files/documents", filename)


    try:
        shutil.copy2(file_path, destination)
        print(f"Moved {filename} to documents folder (category: {category})")
    except Exception as error:
        print(f"Couldnt move the document file: {error}")



def main():
    print("looking for files to sort...")

    if not os.path.exists("input_files"):
        print("Input folder does not exist. Please create an 'input_files' folder and add files to it.")
        return
    
    files_to_process = []

    for filename in os.listdir("input_files"):
        full_path = os.path.join("input_files", filename)
        
        if os.path.isfile(full_path):
            files_to_process.append(full_path)

    if not files_to_process:
        print("No files found to process.")
        return
    print(f"Found {len(files_to_process)} files to process.")
    for file_path in files_to_process:
        process_single_file(file_path)
        print()

    #start count
    documents_count = 0
    images_count = 0
    if os.path.exists("sorted_files/documents"):
        documents_count = len(os.listdir("sorted_files/documents"))
    if os.path.exists("sorted_files/images"):
        images_count = len(os.listdir("sorted_files/images"))

    print("done!")
    print("File organization complete.")
    print(f"Total documents sorted: {documents_count}")
    print(f"Total images sorted: {images_count}")
    print("check the sorted_files  folder to see results")
   
if __name__ == "__main__":
    main()   