# app.py

import os
import re
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from google.cloud import vision
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from PIL import Image
import io
import openai
import googlemaps
import requests
from dotenv import load_dotenv
import pandas as pd
from rapidfuzz import fuzz, process

# Load environment variables from .env file
load_dotenv()

# INITIAL SETUP
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gcp_credentials.json'
app = Flask(__name__)
CORS(app)
vision_client = vision.ImageAnnotatorClient()

# Configuration from environment variables
BGN_TO_EUR_RATE = float(os.getenv('BGN_TO_EUR_RATE', 1.95583))
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')

# OpenAI setup for version >= 1.0.0
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Google Maps setup
gmaps = None
if GOOGLE_MAPS_API_KEY:
    gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
else:
    print("Warning: GOOGLE_MAPS_API_KEY not found. Geolocation-based store detection will be disabled.")

# Google Sheets setup
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = os.getenv('SPREADSHEET_ID', '1z1lclOhKeNEKbSGr3CULc2dMhER3btruLC510CN24bM')
RANGE_NAME = 'Sheet1!A:J'

# Initialize Google Sheets client
credentials = Credentials.from_service_account_file('gcp_credentials.json', scopes=SCOPES)
sheets_service = build('sheets', 'v4', credentials=credentials)

MIN_RELATIVE_HEIGHT_THRESHOLD = 0.03

STORE_OPTIONS = [
    'Kaufland', 'Lidl', 'Metro', 'Billa', 'Fantastico', 'T-Market', 'Пикадили', 'CBA', 'Други'
]

# Default coordinates for major store chains in Sofia (approximate central locations)
STORE_DEFAULT_COORDINATES = {
    'Kaufland': {'lat': 42.6864, 'lng': 23.3238},     # Kaufland Mall of Sofia
    'Lidl': {'lat': 42.6506, 'lng': 23.3806},        # Lidl Mladost
    'Metro': {'lat': 42.6234, 'lng': 23.3844},       # Metro Ringmall
    'Billa': {'lat': 42.6977, 'lng': 23.3219},       # Billa Center
    'Fantastico': {'lat': 42.6584, 'lng': 23.3486},  # Fantastico Vitosha
    'T-Market': {'lat': 42.6977, 'lng': 23.3219},    # T-Market general Sofia
    'Пикадили': {'lat': 42.6977, 'lng': 23.3219},    # Pikadilly general Sofia
    'CBA': {'lat': 42.6977, 'lng': 23.3219},         # CBA general Sofia
    'Други': {'lat': 42.6977, 'lng': 23.3219}        # Other stores - Sofia center
}

# Global variables for product dictionary
PRODUCT_DICTIONARY = []
PRODUCT_DICTIONARY_NORMALIZED = []

# Define column indices for clarity
TIMESTAMP_COL, PRODUCT_NAME_COL, BGN_PRICE_COL = 0, 1, 3

# Phrases to exclude from product names (add more as needed)
EXCLUDED_PHRASES = [
    'специална цена', 'special price', 'промоция', 'promo', 'намаление', 'отстъпка',
    'discount', 'sale', 'акция', 'валиден до', 'valid until', 'expires', 'изтича',
    'fresh', 'свеж', 'daily', 'дневен', 'ново', 'new', 'топ качество', 'top quality',
    'препоръчваме', 'recommend', 'популярен', 'popular', 'hit', 'хит', 'organic',
    'био', 'eco', 'еко', 'natural', 'натурален', 'premium', 'премиум', 'luxury',
    'луксозен', 'imported', 'внос', 'местен', 'local', 'regional', 'регионален', 
    'порция свежест', 'грил майстор', 'стара планина', 'xxl', 'xl', 'l', 'размер',
    'size', 'опаковка', 'package', 'pack', 'бройка', 'брой', 'count', 'cantidad', 'планина'
]

# --- START: PRODUCT DICTIONARY FUNCTIONS ---

def load_product_dictionary():
    """Load and preprocess the Bulgarian product dictionary from CSV."""
    global PRODUCT_DICTIONARY, PRODUCT_DICTIONARY_NORMALIZED
    
    try:
        # Load the CSV file
        df = pd.read_csv('final_product_name.csv')
        
        # Extract product names and clean them
        product_names = df['product_name'].dropna().tolist()
        
        # Remove duplicates and empty entries
        product_names = list(set([name.strip() for name in product_names if name.strip()]))
        
        # Store original names
        PRODUCT_DICTIONARY = product_names
        
        # Create normalized versions for better matching
        PRODUCT_DICTIONARY_NORMALIZED = [normalize_text_for_matching(name) for name in product_names]
        
        print(f"Loaded {len(PRODUCT_DICTIONARY)} products from dictionary")
        
    except Exception as e:
        print(f"Error loading product dictionary: {e}")
        PRODUCT_DICTIONARY = []
        PRODUCT_DICTIONARY_NORMALIZED = []

def normalize_text_for_matching(text):
    """Normalize text for better fuzzy matching (handles common OCR errors)."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Extreme OCR misrecognition patterns (specific cases)
    extreme_replacements = {
        'kloomema': 'кюфтета',
        'klyfteta': 'кюфтета',
        'kyfteta': 'кюфтета',
        'klqfteta': 'кюфтета',
        'ke6an4eta': 'кебапчета',
        'kebapчeta': 'кебапчета',
        'kebap4eta': 'кебапчета',
    }
    
    for wrong, correct in extreme_replacements.items():
        if wrong in text:
            text = text.replace(wrong, correct)
    
    # Common Latin->Cyrillic OCR error corrections
    cyrillic_replacements = {
        'a': 'а', 'e': 'е', 'p': 'р', 'o': 'о', 'c': 'с', 
        'x': 'х', 'y': 'у', 't': 'т', 'n': 'н', 'm': 'м', 'k': 'к',
        'h': 'н', 'b': 'в', 'u': 'у', 'i': 'и', 'j': 'ј', 'l': 'л',
        'q': 'ю', 'w': 'ш', 'f': 'ф', 'g': 'г', 'd': 'д', 'z': 'з',
        '6': 'б', '4': 'ч', '3': 'з', '9': 'д'
    }
    
    for latin, cyrillic in cyrillic_replacements.items():
        text = text.replace(latin, cyrillic)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def fuzzy_match_product(ocr_text, threshold=80, limit=3):
    """
    Find the best fuzzy match for OCR text in the product dictionary.
    Returns tuple: (best_match, confidence_score, all_matches)
    """
    if not PRODUCT_DICTIONARY or not ocr_text:
        return None, 0, []
    
    # Normalize the OCR text
    normalized_ocr = normalize_text_for_matching(ocr_text)
    
    # First pass: Try different matching strategies
    strategies = [
        ('exact', fuzz.ratio),
        ('partial', fuzz.partial_ratio), 
        ('token_sort', fuzz.token_sort_ratio),
        ('token_set', fuzz.token_set_ratio)
    ]
    
    best_matches = []
    
    for strategy_name, scorer in strategies:
        # Match against normalized dictionary
        matches = process.extract(
            normalized_ocr, 
            PRODUCT_DICTIONARY_NORMALIZED, 
            scorer=scorer, 
            limit=limit
        )
        
        for match, score, _ in matches:
            if score >= threshold:
                # Find the original product name
                try:
                    original_idx = PRODUCT_DICTIONARY_NORMALIZED.index(match)
                    original_name = PRODUCT_DICTIONARY[original_idx]
                    best_matches.append((original_name, score, strategy_name))
                except ValueError:
                    continue
    
    # Second pass: If no good matches, try with original OCR text at lower threshold
    if not best_matches and threshold > 50:
        print(f"No matches found for normalized '{normalized_ocr}', trying original '{ocr_text}' with lower threshold")
        for strategy_name, scorer in strategies:
            matches = process.extract(
                ocr_text, 
                PRODUCT_DICTIONARY, 
                scorer=scorer, 
                limit=limit
            )
            
            for match, score, _ in matches:
                if score >= (threshold - 20):  # Lower threshold for second pass
                    best_matches.append((match, score, f"{strategy_name}_original"))
    
    if not best_matches:
        return None, 0, []
    
    # Sort by score and return best match
    best_matches.sort(key=lambda x: x[1], reverse=True)
    best_match = best_matches[0]
    
    print(f"Best match for '{ocr_text}' -> '{best_match[0]}' (score: {best_match[1]}, strategy: {best_match[2]})")
    
    return best_match[0], best_match[1], best_matches[:limit]

def enhance_product_candidates_with_dictionary(candidates):
    """Enhance product candidates with dictionary matches."""
    enhanced_candidates = []
    
    for candidate in candidates:
        # Get original candidate info
        enhanced_candidate = candidate.copy()
        
        # Try fuzzy matching with lower threshold for extreme OCR errors
        match, confidence, all_matches = fuzzy_match_product(candidate["text"], threshold=60)
        
        if match and confidence > 80:
            # High confidence dictionary match - boost this candidate
            enhanced_candidate["dictionary_match"] = match
            enhanced_candidate["dictionary_confidence"] = confidence
            enhanced_candidate["dictionary_all_matches"] = all_matches
            
            # Boost prominence score for dictionary matches
            enhanced_candidate["weighted_area"] *= 1.5
            enhanced_candidate["has_dictionary_match"] = True
            
            print(f"Dictionary match: '{candidate['text']}' -> '{match}' (confidence: {confidence})")
        else:
            enhanced_candidate["dictionary_match"] = None
            enhanced_candidate["dictionary_confidence"] = 0
            enhanced_candidate["has_dictionary_match"] = False
        
        enhanced_candidates.append(enhanced_candidate)
    
    return enhanced_candidates

# --- END: PRODUCT DICTIONARY FUNCTIONS ---

# --- START: SPATIAL ANALYSIS FUNCTIONS ---

def get_block_height(vertices):
    """Calculates the height of a bounding box."""
    return max(v.y for v in vertices) - min(v.y for v in vertices)

def get_block_width(vertices):
    """Calculates the width of a bounding box."""
    return max(v.x for v in vertices) - min(v.x for v in vertices)

def get_block_center(vertices):
    """Gets the center point of a bounding box."""
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    return {
        'x': sum(x_coords) / len(x_coords),
        'y': sum(y_coords) / len(y_coords)
    }

def is_price_like(text):
    """Checks if a string resembles a price or currency-related noise."""
    text = text.strip().lower()
    # Matches numbers like 12,99, 1.45, or text with currency symbols
    if re.search(r'^\d+([.,]\d{1,2})?$', text):
        return True
    if re.search(r'(лв|bgn|eur|€|st|ct|ст)', text):
        return True
    # Weight/quantity indicators
    if re.search(r'(кг|kg|гр|gr|g|мл|ml|л|l|бр|br|шт|pc)$', text):
        return True
    # Barcode patterns
    if re.search(r'^\d{8,13}$', text):
        return True
    return False

def is_store_info(text):
    """Detects store-related information."""
    text = text.strip().lower()
    store_keywords = ['kaufland', 'lidl', 'metro', 'billa', 'fantastico', 't-market', 'пикадили', 'cba']
    for keyword in store_keywords:
        if keyword in text:
            return True
    return False

def contains_excluded_phrase(text):
    """Check if text contains any excluded phrases."""
    text_lower = text.lower().strip()
    for phrase in EXCLUDED_PHRASES:
        if phrase.lower() in text_lower:
            return True
    return False

def remove_excluded_phrases_from_text(text):
    """Remove excluded phrases from text while preserving structure."""
    if not text:
        return text
    
    # Split into lines to preserve structure
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        should_exclude = False
        
        # Check if entire line contains excluded phrases
        for phrase in EXCLUDED_PHRASES:
            if phrase.lower() in line_lower:
                should_exclude = True
                break
        
        if not should_exclude:
            filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def get_text_style_weight(annotation):
    """
    Analyze text style properties to determine importance weight.
    Uses bounding box height as proxy for font size/emphasis.
    """
    weight = 1.0
    
    # Get bounding box height as proxy for font size
    if hasattr(annotation, 'bounding_poly') and annotation.bounding_poly.vertices:
        vertices = annotation.bounding_poly.vertices
        height = get_block_height(vertices)
        
        # Larger text gets higher weight (height-based emphasis detection)
        if height > 50:  # Very large text (likely headers/product names)
            weight *= 2.0
        elif height > 35:  # Large text (likely important)
            weight *= 1.6
        elif height > 25:  # Medium-large text
            weight *= 1.3
        elif height > 15:  # Standard text
            weight *= 1.0
        else:  # Very small text (likely fine print)
            weight *= 0.7
    
    # Check if we have access to text style information
    if hasattr(annotation, 'property') and annotation.property:
        # Google Vision API sometimes provides text style info
        if hasattr(annotation.property, 'detected_break'):
            break_type = annotation.property.detected_break.type_
            # Text with strong breaks (paragraphs) might be more important
            if break_type in ['LINE_BREAK', 'EOL_SURE_SPACE']:
                weight *= 1.1
        
        # Check for font properties if available
        if hasattr(annotation.property, 'detected_text') and annotation.property.detected_text:
            # This is rarely available but worth checking
            detected_text = annotation.property.detected_text
            if hasattr(detected_text, 'is_bold') and detected_text.is_bold:
                weight *= 1.8
            if hasattr(detected_text, 'font_size') and detected_text.font_size:
                # Larger font sizes get higher weight
                if detected_text.font_size > 14:
                    weight *= 1.4
                elif detected_text.font_size > 12:
                    weight *= 1.2
    
    return weight

def is_date_like(text):
    """Detects date patterns."""
    text = text.strip()
    if re.search(r'\d{1,2}[./]\d{1,2}[./]\d{2,4}', text):
        return True
    return False

def analyze_text_blocks_spatial(annotations):
    """
    Analyzes text blocks with spatial relationships to identify product name candidates.
    """
    if not annotations or len(annotations) < 2:
        return []

    blocks = []
    for annotation in annotations[1:]:  # Skip the first full-text annotation
        text = annotation.description.strip()
        if len(text) > 1:
            vertices = annotation.bounding_poly.vertices
            
            # Get style weight for this text block
            style_weight = get_text_style_weight(annotation)
            
            block_info = {
                "text": text,
                "height": get_block_height(vertices),
                "width": get_block_width(vertices),
                "center": get_block_center(vertices),
                "area": get_block_height(vertices) * get_block_width(vertices),
                "vertices": vertices,
                "style_weight": style_weight,
                "is_price_like": is_price_like(text),
                "is_store_info": is_store_info(text),
                "is_date_like": is_date_like(text),
                "has_excluded_phrase": contains_excluded_phrase(text)
            }
            blocks.append(block_info)
    
    # Filter out obvious non-product-name blocks
    product_candidates = []
    for block in blocks:
        # Skip blocks with excluded phrases, prices, store info, dates
        if not (block["is_price_like"] or block["is_store_info"] or 
                block["is_date_like"] or block["has_excluded_phrase"]):
            # Apply style weight to the area score
            block["weighted_area"] = block["area"] * block["style_weight"]
            product_candidates.append(block)
    
    # Sort by weighted area (style-adjusted visual prominence)
    product_candidates.sort(key=lambda x: x["weighted_area"], reverse=True)
    
    return product_candidates

def find_connected_text_blocks(candidates, max_distance=50):
    """
    Groups text blocks that are spatially close to each other (likely multi-line product names).
    """
    if not candidates:
        return []
    
    groups = []
    used_indices = set()
    
    for i, block in enumerate(candidates):
        if i in used_indices:
            continue
            
        # Start a new group with this block
        group = [block]
        used_indices.add(i)
        
        # Find nearby blocks
        for j, other_block in enumerate(candidates):
            if j in used_indices or j == i:
                continue
            
            # Calculate distance between block centers
            dx = abs(block["center"]["x"] - other_block["center"]["x"])
            dy = abs(block["center"]["y"] - other_block["center"]["y"])
            
            # If blocks are close vertically (multi-line) or horizontally (continuation)
            if dy < max_distance or (dx < max_distance and dy < block["height"]):
                group.append(other_block)
                used_indices.add(j)
        
        if group:
            groups.append(group)
    
    return groups

def extract_product_name_advanced(annotations):
    """
    Enhanced product name extraction with spatial analysis and LLM verification.
    Returns tuple: (product_name, bounding_box_vertices)
    """
    if not annotations or len(annotations) < 2:
        return "Неизвестен продукт", None

    # Get all text lines for context, filtering out excluded phrases
    full_text = annotations[0].description
    filtered_full_text = remove_excluded_phrases_from_text(full_text)
    
    # Analyze spatial relationships
    product_candidates = analyze_text_blocks_spatial(annotations)
    
    if not product_candidates:
        return "Неизвестен продукт", None
    
    # Enhance candidates with dictionary matching
    product_candidates = enhance_product_candidates_with_dictionary(product_candidates)
    
    # Group spatially connected blocks
    text_groups = find_connected_text_blocks(product_candidates)
    
    # Prepare candidates for LLM analysis
    llm_candidates = []
    
    # Add individual blocks (sorted by prominence)
    for block in product_candidates[:8]:  # Top 8 most prominent blocks
        candidate_info = {
            "text": block["text"],
            "type": "individual",
            "prominence_score": block["weighted_area"],  # Use weighted area
            "style_weight": block["style_weight"],
            "vertices": block["vertices"]
        }
        
        # Add dictionary match information if available
        if block.get("has_dictionary_match", False):
            candidate_info["dictionary_match"] = block["dictionary_match"]
            candidate_info["dictionary_confidence"] = block["dictionary_confidence"]
        
        llm_candidates.append(candidate_info)
    
    # Add grouped blocks (potential multi-line names)
    for group in text_groups:
        if len(group) > 1:  # Only multi-block groups
            # Sort blocks in group by vertical position (top to bottom)
            group.sort(key=lambda x: x["center"]["y"])
            combined_text = " ".join([block["text"] for block in group])
            total_weighted_area = sum([block["weighted_area"] for block in group])
            avg_style_weight = sum([block["style_weight"] for block in group]) / len(group)
            
            # Create combined bounding box for grouped text
            all_vertices = []
            for block in group:
                all_vertices.extend(block["vertices"])
            
            # Try dictionary matching on the combined text
            group_match, group_confidence, group_all_matches = fuzzy_match_product(combined_text, threshold=60)
            
            group_candidate = {
                "text": combined_text,
                "type": "grouped",
                "prominence_score": total_weighted_area,
                "component_count": len(group),
                "style_weight": avg_style_weight,
                "vertices": all_vertices,
                "group_blocks": group
            }
            
            # Add dictionary match information if available
            if group_match and group_confidence > 75:
                group_candidate["dictionary_match"] = group_match
                group_candidate["dictionary_confidence"] = group_confidence
                group_candidate["has_dictionary_match"] = True
                # Boost prominence for dictionary matches
                group_candidate["prominence_score"] *= 1.8  # Higher boost for grouped matches
                print(f"Dictionary match for group: '{combined_text}' -> '{group_match}' (confidence: {group_confidence})")
            else:
                group_candidate["dictionary_match"] = None
                group_candidate["dictionary_confidence"] = 0
                group_candidate["has_dictionary_match"] = False
            
            llm_candidates.append(group_candidate)
    
    # Sort all candidates by prominence
    llm_candidates.sort(key=lambda x: x["prominence_score"], reverse=True)
    
    # Prepare enhanced prompt for LLM
    # Create list of excluded phrases for LLM prompt
    excluded_phrases_text = ', '.join(EXCLUDED_PHRASES)
    
    prompt = f"""
    You are analyzing a Bulgarian price label to extract the product name. Below are text candidates extracted from the label, including both individual text blocks and spatially-grouped multi-line text combinations. The final product name MUST be written ONLY in Bulgarian Cyrillic script. If the OCR text contains Latin characters that look like Bulgarian words, convert them to proper Bulgarian Cyrillic.

    FULL LABEL TEXT (for context - promotional phrases already filtered):
    {filtered_full_text}

    CANDIDATE TEXTS (ordered by visual prominence):
    {json.dumps([{
        "text": c["text"], 
        "type": c["type"],
        "prominence_score": c["prominence_score"],
        "dictionary_match": c.get("dictionary_match", None),
        "dictionary_confidence": c.get("dictionary_confidence", 0)
    } for c in llm_candidates[:10]], ensure_ascii=False, indent=2)}

    ANALYSIS INSTRUCTIONS:
    1. Identify which candidate represents the complete product name
    2. CRITICAL: Strongly prefer "grouped" type candidates over "individual" ones - they represent complete multi-word product names
    3. Multi-line product names are common - look for candidates that combine related text blocks
    4. Ignore: prices, weights (кг, гр, мл), store names, dates, barcodes, promotional text
    5. CRITICAL: Completely avoid and reject ANY text containing these promotional phrases: {excluded_phrases_text}
    6. IMPORTANT: If a candidate has a dictionary_match with high dictionary_confidence (>75), strongly prefer it as it matches a real Bulgarian product
    7. The product name should be meaningful and coherent in Bulgarian
    8. Prefer candidates that appear to be emphasized/bold (higher style weight and prominence_score)
    9. If multiple candidates seem valid, choose the most complete/descriptive one (grouped > individual)
    10. Clean up the final name (proper capitalization, remove extra spaces)
    11. If choosing a dictionary match, use the dictionary_match text, not the original OCR text

    EXAMPLES:
    - "ЛУКАНКА ОРЕХИТЕ ТРАДИЦИОННА" → "Луканка Орехите Традиционна"
    - "МЛЕЧЕН ШОКОЛАД MILKA" → "Млечен Шоколад Milka"
    - "ХЛЯБ ЧЕРЕН ПЪЛНОЗЪРНЕСТ" → "Хляб Черен Пълнозърнест"

    Respond with a JSON object containing:
    {{
        "product_name": "extracted product name",
        "confidence": "high/medium/low",
        "reasoning": "brief explanation of your choice"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are an expert at analyzing Bulgarian product labels and extracting product names from OCR text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        result_json = json.loads(response.choices[0].message.content)
        product_name = result_json.get("product_name", "").strip()
        confidence = result_json.get("confidence", "low")
        reasoning = result_json.get("reasoning", "")
        
        print(f"LLM Product Extraction - Name: {product_name}, Confidence: {confidence}, Reasoning: {reasoning}")
        
        # Find the matching candidate to get bounding box
        selected_candidate = None
        if product_name and product_name.lower() not in ["неизвестен продукт", "unknown", ""]:
            # Try to find the candidate that matches the selected product name
            for candidate in llm_candidates:
                if candidate["text"].lower() == product_name.lower():
                    selected_candidate = candidate
                    break
        
        # Fallback if LLM returns empty or invalid result
        if not product_name or product_name.lower() in ["неизвестен продукт", "unknown", ""]:
            # Try to use dictionary matches first
            dictionary_candidates = [c for c in llm_candidates if c.get("dictionary_match")]
            if dictionary_candidates:
                best_dict_candidate = max(dictionary_candidates, key=lambda x: x.get("dictionary_confidence", 0))
                product_name = best_dict_candidate["dictionary_match"]
                selected_candidate = best_dict_candidate
                print(f"Using dictionary fallback: {product_name}")
            elif llm_candidates:
                product_name = llm_candidates[0]["text"]
                selected_candidate = llm_candidates[0]
            else:
                product_name = "Неизвестен продукт"
                selected_candidate = None
        
        # If still no match, use the first candidate
        if not selected_candidate and llm_candidates:
            selected_candidate = llm_candidates[0]
        
        # Extract bounding box vertices
        bounding_box = None
        if selected_candidate and "vertices" in selected_candidate:
            vertices = selected_candidate["vertices"]
            if selected_candidate["type"] == "grouped":
                # For grouped text, create a bounding box that encompasses all blocks
                x_coords = [v.x for v in vertices]
                y_coords = [v.y for v in vertices]
                bounding_box = [
                    {'x': min(x_coords), 'y': min(y_coords)},
                    {'x': max(x_coords), 'y': min(y_coords)},
                    {'x': max(x_coords), 'y': max(y_coords)},
                    {'x': min(x_coords), 'y': max(y_coords)}
                ]
            else:
                # For individual text blocks
                bounding_box = [{'x': v.x, 'y': v.y} for v in vertices]
        
        return product_name, bounding_box

    except Exception as e:
        print(f"LLM extraction error: {e}")
        
        # Enhanced fallback strategy
        if llm_candidates:
            best_candidate = None
            best_score = 0
            
            for candidate in llm_candidates:
                text = candidate["text"]
                score = candidate["prominence_score"]
                
                # Boost score for longer, more descriptive text
                if len(text) > 10:
                    score *= 1.5
                
                # Boost score for grouped (multi-line) candidates
                if candidate["type"] == "grouped":
                    score *= 1.3
                
                # Boost score for text with higher style weight (bold/emphasized)
                style_weight = candidate.get("style_weight", 1.0)
                if style_weight > 1.2:
                    score *= 1.2
                
                # Penalize very short text
                if len(text) < 4:
                    score *= 0.5
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            if best_candidate:
                # Extract bounding box for fallback candidate
                bounding_box = None
                if "vertices" in best_candidate:
                    vertices = best_candidate["vertices"]
                    if best_candidate["type"] == "grouped":
                        # For grouped text, create a bounding box that encompasses all blocks
                        x_coords = [v.x for v in vertices]
                        y_coords = [v.y for v in vertices]
                        bounding_box = [
                            {'x': min(x_coords), 'y': min(y_coords)},
                            {'x': max(x_coords), 'y': min(y_coords)},
                            {'x': max(x_coords), 'y': max(y_coords)},
                            {'x': min(x_coords), 'y': max(y_coords)}
                        ]
                    else:
                        # For individual text blocks
                        bounding_box = [{'x': v.x, 'y': v.y} for v in vertices]
                
                return best_candidate["text"], bounding_box
            else:
                return "Неизвестен продукт", None
        
        return "Неизвестен продукт", None

# --- END: SPATIAL ANALYSIS FUNCTIONS ---

# --- START: UTILITY FUNCTIONS ---

def find_last_price_in_sheets(product_name):
    """Find the last recorded price for a given product."""
    try:
        result = sheets_service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
        values = result.get('values', [])
        if not values or len(values) <= 1: 
            return None
        
        for row in reversed(values[1:]):
            if len(row) > BGN_PRICE_COL and len(row) > PRODUCT_NAME_COL:
                if row[PRODUCT_NAME_COL].strip().lower() == product_name.lower():
                    return {
                        "last_price_bgn": float(row[BGN_PRICE_COL]), 
                        "last_scan_date": row[TIMESTAMP_COL].split(' ')[0]
                    }
        return None
    except Exception as e:
        print(f"Error finding historical price: {e}")
        return None

def calculate_area(vertices):
    """Calculate the area of a bounding box."""
    x_coords = [v.x for v in vertices]
    y_coords = [v.y for v in vertices]
    return (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))

def parse_price_string(s):
    """Parse a price string into a float."""
    if not s: 
        return None
    try: 
        return float(s.replace(',', '.'))
    except (ValueError, TypeError): 
        return None

def get_fallback_coordinates(store_name, latitude=None, longitude=None):
    """
    Get fallback coordinates when GPS is not available.
    Priority order:
    1. Use provided GPS coordinates if valid
    2. Use store-specific default coordinates
    3. Use Sofia city center as ultimate fallback
    """
    # If we have valid GPS coordinates, use them
    if latitude is not None and longitude is not None:
        try:
            lat = float(latitude)
            lng = float(longitude)
            # Basic validation - check if coordinates are in Bulgaria region
            if 41.0 <= lat <= 44.0 and 22.0 <= lng <= 29.0:
                return lat, lng
        except (ValueError, TypeError):
            pass
    
    # Try store-specific coordinates
    if store_name and store_name in STORE_DEFAULT_COORDINATES:
        coords = STORE_DEFAULT_COORDINATES[store_name]
        return coords['lat'], coords['lng']
    
    # Ultimate fallback - Sofia city center
    return 42.6977, 23.3219

def add_coordinate_spread(latitude, longitude, existing_coordinates, min_distance=0.001):
    """
    Add small random offset to coordinates if they're too close to existing ones.
    This prevents overlapping markers on the map.
    """
    import random
    
    # Check if coordinates are too close to existing ones
    for existing_lat, existing_lng in existing_coordinates:
        if existing_lat and existing_lng:
            distance = ((latitude - existing_lat) ** 2 + (longitude - existing_lng) ** 2) ** 0.5
            if distance < min_distance:
                # Add small random offset
                latitude += random.uniform(-0.002, 0.002)
                longitude += random.uniform(-0.002, 0.002)
                break
    
    return latitude, longitude

def find_nearby_stores(latitude, longitude, radius=1000):
    """Find nearby stores using Google Places API (New) via direct HTTP requests."""
    if not GOOGLE_MAPS_API_KEY:
        return []
    
    try:
        # Use the new Places API (New) Nearby Search endpoint
        url = "https://places.googleapis.com/v1/places:searchNearby"
        
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': GOOGLE_MAPS_API_KEY,
            'X-Goog-FieldMask': 'places.displayName,places.rating,places.formattedAddress'
        }
        
        data = {
            "includedTypes": ["grocery_store", "supermarket"],
            "maxResultCount": 10,
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": latitude,
                        "longitude": longitude
                    },
                    "radius": radius
                }
            },
            "languageCode": "bg"
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code != 200:
            print(f"Places API error: {response.status_code} - {response.text}")
            return []
        
        result = response.json()
        nearby_stores = []
        
        for place in result.get('places', []):
            store_name = place.get('displayName', {}).get('text', '').lower()
            
            # Check if the store name matches our known store list
            for store_option in STORE_OPTIONS:
                if store_option.lower() in store_name:
                    nearby_stores.append({
                        'name': store_option,
                        'distance': place.get('formattedAddress', ''),
                        'rating': place.get('rating', 0)
                    })
                    break
        
        # Sort by rating (higher is better)
        nearby_stores.sort(key=lambda x: x['rating'], reverse=True)
        return nearby_stores
        
    except Exception as e:
        print(f"Error finding nearby stores: {e}")
        return []

def extract_store_name_from_text(ocr_text, latitude=None, longitude=None):
    """Extract store name from OCR text, with optional geolocation fallback."""
    if not ocr_text: 
        return "Неизвестен магазин"
    
    lines = ocr_text.split('\n')
    store_keywords = ['kaufland', 'lidl', 'metro', 'billa', 'fantastico', 't-market', 'пикадили', 'cba']
    
    # First try to find store name in OCR text
    for line in lines:
        for keyword in store_keywords:
            if keyword in line.strip().lower(): 
                return keyword.title()
    
    # If not found in OCR and we have geolocation, try nearby stores
    if latitude is not None and longitude is not None:
        nearby_stores = find_nearby_stores(latitude, longitude)
        if nearby_stores:
            print(f"Found nearby stores: {[store['name'] for store in nearby_stores]}")
            return nearby_stores[0]['name']  # Return the highest-rated nearby store
    
    return "Неизвестен магазин"

def save_to_sheets(product_name, store_name, bgn_price, eur_price, expected_eur, status, latitude=None, longitude=None):
    """Save the price verification result to Google Sheets."""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        difference = round(eur_price - expected_eur, 4)
        values = [[timestamp, product_name, store_name, bgn_price, eur_price, expected_eur, difference, status, latitude, longitude]]
        body = {'values': values, 'majorDimension': 'ROWS'}
        
        sheets_service.spreadsheets().values().append(
            spreadsheetId=SPREADSHEET_ID, 
            range=RANGE_NAME, 
            valueInputOption='RAW', 
            insertDataOption='INSERT_ROWS', 
            body=body
        ).execute()
        
        return True
    except Exception as e:
        print(f"Error saving to sheets: {e}")
        return False

def setup_sheet_headers():
    """Setup the headers for the Google Sheet."""
    try:
        headers = [['Timestamp', 'Product Name', 'Store Name', 'BGN Price', 'EUR Price', 'Expected EUR', 'Difference', 'Status', 'Latitude', 'Longitude']]
        body = {'values': headers, 'majorDimension': 'ROWS'}
        
        sheets_service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID, 
            range='Sheet1!A1:J1', 
            valueInputOption='RAW', 
            body=body
        ).execute()
        
        return True
    except Exception as e:
        print(f"Error setting up headers: {e}")
        return False

# --- END: UTILITY FUNCTIONS ---

# --- START: API ENDPOINTS ---

@app.route('/api/stores', methods=['GET'])
def get_stores():
    """Get the list of available stores."""
    return jsonify({"stores": STORE_OPTIONS})

@app.route('/api/prices', methods=['GET'])
def get_prices():
    """Get price data for map visualization."""
    try:
        # Get all data from Google Sheets
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID, 
            range=RANGE_NAME
        ).execute()
        values = result.get('values', [])
        
        if not values or len(values) <= 1:
            return jsonify({"prices": []})
        
        prices = []
        existing_coordinates = []
        
        # Skip header row
        for row in values[1:]:
            if len(row) >= 8:  # Ensure we have all required columns
                try:
                    # Get coordinates using smart fallback system
                    stored_lat = row[8] if len(row) >= 10 and row[8] else None
                    stored_lng = row[9] if len(row) >= 10 and row[9] else None
                    store_name = row[2]
                    
                    # Use fallback coordinate system
                    latitude, longitude = get_fallback_coordinates(store_name, stored_lat, stored_lng)
                    
                    # Add slight spread to prevent overlapping markers
                    latitude, longitude = add_coordinate_spread(latitude, longitude, existing_coordinates)
                    existing_coordinates.append((latitude, longitude))
                    
                    price_data = {
                        "product_name": row[1],
                        "store_name": store_name,
                        "price_bgn": float(row[3]),
                        "price_eur": float(row[4]),
                        "expected_eur": float(row[5]),
                        "scan_date": row[0].split(' ')[0],  # Extract date part
                        "status": row[7],
                        "latitude": latitude,
                        "longitude": longitude,
                        "is_best_price": row[7] == "CORRECT"
                    }
                    prices.append(price_data)
                except (ValueError, IndexError) as e:
                    print(f"Error parsing row: {row}, error: {e}")
                    continue
        
        # Limit to recent entries (last 100 for performance)
        prices = prices[-100:]
        
        return jsonify({"prices": prices})
        
    except Exception as e:
        print(f"Error getting price data: {e}")
        return jsonify({"prices": []})

@app.route('/api/verify-prices', methods=['POST'])
def verify_prices():
    """Main endpoint for price verification."""
    if 'file' not in request.files:
        return jsonify({"status": "ГРЕШКА", "message": "Не е качен файл."}), 400
    
    file = request.files['file']
    image_content = file.read()
    
    selected_store = request.form.get('store', 'Неизвестен магазин')
    latitude = request.form.get('latitude')
    longitude = request.form.get('longitude')
    
    # Convert coordinates to float if provided
    if latitude and longitude:
        try:
            latitude = float(latitude)
            longitude = float(longitude)
            print(f"Received geolocation: {latitude}, {longitude}")
        except ValueError:
            latitude = None
            longitude = None
            print("Invalid geolocation coordinates received")
    
    try:
        image_for_dims = Image.open(io.BytesIO(image_content))
        image_width, image_height = image_for_dims.size
    except Exception as e:
        return jsonify({"status": "ГРЕШКА", "message": f"Невалиден файл с изображение: {e}"})

    # Perform OCR using Google Vision API
    image = vision.Image(content=image_content)
    image_context = vision.ImageContext(language_hints=['bg', 'bg-BG'])
    #response = vision_client.text_detection(image=image, image_context=image_context) - old text_detection
    response = vision_client.document_text_detection(image=image, image_context=image_context)
    annotations = response.text_annotations

    if not annotations:
        return jsonify({"status": "ГРЕШКА", "message": "Не можах да разчета текст."})

    # Extract product name using advanced spatial analysis
    product_name, product_name_box = extract_product_name_advanced(annotations)
    
    print(f"Extracted product name: {product_name}")
    print(f"Product name bounding box: {product_name_box}")
    
    # Store detection
    if selected_store == "Неизвестен магазин" or not selected_store:
        full_ocr_text = annotations[0].description
        store_name = extract_store_name_from_text(full_ocr_text, latitude, longitude)
    else:
        store_name = selected_store

    # Price extraction
    price_candidates = []
    price_pattern = re.compile(r'^\d+([.,]\d{1,2})?$')
    
    for annotation in annotations[1:]:
        if price_pattern.match(annotation.description):
            value = parse_price_string(annotation.description)
            if value is not None:
                price_candidates.append({
                    'value': value,
                    'area': calculate_area(annotation.bounding_poly.vertices),
                    'box': [{'x': v.x, 'y': v.y} for v in annotation.bounding_poly.vertices]
                })

    if len(price_candidates) < 2:
        return jsonify({"status": "ERROR", "message": f"Намерени са по-малко от две цени."})

    # Sort price candidates by area (visual prominence)
    price_candidates.sort(key=lambda p: p['area'], reverse=True)
    
    # Check if the image is taken from appropriate distance
    largest_price_box = price_candidates[0]['box']
    box_height = max(v['y'] for v in largest_price_box) - min(v['y'] for v in largest_price_box)
    relative_height = box_height / image_height
    
    if relative_height < MIN_RELATIVE_HEIGHT_THRESHOLD:
        return jsonify({"status": "TOO-FAR", "message": "Приближете се до етикета и опитайте отново."})

    # Determine BGN and EUR prices (assume larger price is BGN)
    largest_price_1, largest_price_2 = price_candidates[0], price_candidates[1]
    if largest_price_1['value'] > largest_price_2['value']:
        bgn_price_data, eur_price_data = largest_price_1, largest_price_2
    else:
        bgn_price_data, eur_price_data = largest_price_2, largest_price_1
        
    price_bgn = bgn_price_data['value']
    price_eur = eur_price_data['value']
    expected_eur = price_bgn / BGN_TO_EUR_RATE
    price_difference = price_eur - expected_eur

    # Determine if the price is correct
    if price_difference > 0.01:
        status, message = "INCORRECT", "Цената в EUR изглежда несправедливо завишена."
    else:
        status, message = "CORRECT", "Цената е правилна и съответства на официалния курс."

    # Get historical price information
    historical_price_info = None
    if product_name != "Неизвестен продукт":
        historical_data = find_last_price_in_sheets(product_name)
        if historical_data:
            last_price = historical_data['last_price_bgn']
            price_change = price_bgn - last_price
            historical_price_info = {
                "last_price_bgn": last_price, 
                "last_scan_date": historical_data['last_scan_date'], 
                "price_change_bgn": round(price_change, 2)
            }

    # Get fallback coordinates if GPS is not available
    final_latitude, final_longitude = get_fallback_coordinates(store_name, latitude, longitude)
    
    # Save results to Google Sheets
    save_success = save_to_sheets(product_name, store_name, price_bgn, price_eur, round(expected_eur, 2), status, final_latitude, final_longitude)

    # Prepare response
    response_data = {
        "status": status, 
        "message": message,
        "data": {
            "product_name": product_name, 
            "store_name": store_name, 
            "found_bgn": price_bgn,
            "found_eur": price_eur, 
            "expected_eur": round(expected_eur, 2), 
            "difference_eur": round(price_difference, 4),
            "bgn_box": bgn_price_data['box'], 
            "eur_box": eur_price_data['box'], 
            "product_name_box": product_name_box,
            "saved_to_sheets": save_success,
            "historical_price": historical_price_info
        }
    }
    return jsonify(response_data)

@app.route('/api/setup-sheet', methods=['POST'])
def setup_sheet():
    """Setup the Google Sheet with proper headers."""
    success = setup_sheet_headers()
    return jsonify({
        "status": "SUCCESS" if success else "ERROR", 
        "message": "Sheet headers created successfully" if success else "Failed to create sheet headers"
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "OK",
        "message": "Price verification service is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/maps-key', methods=['GET'])
def get_maps_key():
    """Get Google Maps API key for frontend."""
    if not GOOGLE_MAPS_API_KEY:
        return jsonify({"error": "Google Maps API key not configured"}), 500
    return jsonify({"api_key": GOOGLE_MAPS_API_KEY})

# --- END: API ENDPOINTS ---

# Load product dictionary on startup
load_product_dictionary()

if __name__ == '__main__':
    app.run(debug=True)