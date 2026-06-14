"""
Arabic Text Preprocessing Utilities
Handles: Franco-Arabic conversion, normalization, stopwords, stemming
"""

import re
import string
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Initialize stemmer and stopwords
stemmer = ISRIStemmer()
arabic_stops = set(stopwords.words('arabic'))

# Custom stopwords
custom_stops = {'هذا', 'هذه', 'ذلك', 'تلك', 'كان', 'كانت', 'يكون', 'تكون'}
arabic_stops.update(custom_stops)


def is_english_word(word):
    """Check if a word is English or Franco-Arabic"""
    english_patterns = [
        r'^[a-zA-Z]+$',
        r'[aeiouAEIOU]',
        r'\b(the|and|is|was|are|were|have|has|will|can|this|that|with|for|from)\b'
    ]
    
    franco_patterns = [
        r'[0-9]',
        r'^[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]+$',
        r'(7|5|3|6|9|2)'
    ]
    
    for pattern in english_patterns:
        if re.search(pattern, word, re.IGNORECASE):
            return True
    
    for pattern in franco_patterns:
        if re.search(pattern, word):
            return False
    
    consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', word))
    vowels = len(re.findall(r'[aeiouAEIOU]', word))
    
    return vowels > 0 and consonants / max(len(word), 1) < 0.8


def convert_franco_to_arabic(text):
    """Convert Franco-Arabic to Arabic script"""
    conversion_dict = {
        'thaـ': 'ثا', 'tha': 'ثا',
        'sh': 'ش', 'th': 'ث', 'gh': 'غ', 'kh': 'خ',
        'a': 'ا', 'b': 'ب', 't': 'ت', 'g': 'ج', 'd': 'د',
        'r': 'ر', 'z': 'ز', 's': 'س', 'f': 'ف', 'q': 'ق',
        'k': 'ك', 'l': 'ل', 'm': 'م', 'n': 'ن', 'h': 'ه',
        'w': 'و', 'y': 'ي',
        '7': 'ح', '5': 'خ', '3': 'ع', '9': 'ص', '6': 'ط', '2': 'ء'
    }
    
    for franco, arabic in conversion_dict.items():
        text = text.replace(franco, arabic)
    
    return text


def handle_mixed_language_text(text):
    """Handle Franco-Arabic and mixed language text"""
    words = text.split()
    processed_words = []
    
    for word in words:
        if re.search(r'[\u0600-\u06FF]', word):
            processed_words.append(word)
        elif re.search(r'[0-9]', word) or re.search(r'^[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{3,}$', word.lower()):
            converted = convert_franco_to_arabic(word.lower())
            processed_words.append(converted)
        else:
            continue
    
    return ' '.join(processed_words)


def normalize_arabic_text(text):
    """Enhanced Arabic text normalization"""
    
    # Handle mixed language
    text = handle_mixed_language_text(text)
    
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize letters
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    text = re.sub("ڤ", "ف", text)
    text = re.sub("چ", "ج", text)
    text = re.sub("پ", "ب", text)
    text = re.sub("ڜ", "ش", text)
    text = re.sub("ڪ", "ك", text)
    text = re.sub("ڧ", "ق", text)
    text = re.sub("ٱ", "ا", text)
    
    # Remove tatweel/kashida
    text = re.sub(r'\u0640', '', text)
    
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)
    
    # Convert emojis to text
    text = emoji.demojize(text, language='ar')
    
    # Convert numbers to words
    number_map = {
        '٠': 'صفر', '١': 'واحد', '٢': 'اثنان', '٣': 'ثلاثة', '٤': 'أربعة',
        '٥': 'خمسة', '٦': 'ستة', '٧': 'سبعة', '٨': 'ثمانية', '٩': 'تسعة',
        '0': 'صفر', '1': 'واحد', '2': 'اثنان', '3': 'ثلاثة', '4': 'أربعة',
        '5': 'خمسة', '6': 'ستة', '7': 'سبعة', '8': 'ثمانية', '9': 'تسعة'
    }
    for num, name in number_map.items():
        text = text.replace(num, name)
    
    # Remove punctuation
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ'''
    english_punctuations = string.punctuation
    all_punctuations = arabic_punctuations + english_punctuations
    text = re.sub(f"[{re.escape(all_punctuations)}]", " ", text)
    
    # Remove repeated characters (3+ times -> 2 times)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Keep only Arabic characters and spaces
    text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s]', '', text)
    
    return text


def remove_stopwords_and_stem(tokens):
    """Remove stopwords and apply stemming"""
    if not tokens:
        return []
    
    filtered_tokens = [token for token in tokens if token not in arabic_stops and len(token) > 2]
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    return stemmed_tokens


def preprocess_text(text):
    """Complete preprocessing pipeline"""
    if not isinstance(text, str):
        return ""
    
    # Normalize Arabic text
    text = normalize_arabic_text(text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and stem
    processed_tokens = remove_stopwords_and_stem(tokens)
    
    return ' '.join(processed_tokens)


def preprocess_for_inference(text):
    """Preprocess single text for prediction"""
    return preprocess_text(text)
