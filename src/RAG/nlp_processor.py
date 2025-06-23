"""
NLP Processor for text analysis and sentiment processing
Handles NLTK operations, sentiment analysis, and keyword extraction
"""

import nltk
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Set
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.tag import pos_tag
from pysentimiento import create_analyzer
from keybert import KeyBERT

logger = logging.getLogger(__name__)

class NLPProcessor:
    """Handles all NLP operations including sentiment analysis and keyword extraction"""
    
    def __init__(self):
        """Initialize NLP components"""
        self.keybert_model = KeyBERT()
        self.sentiment_analyzer = create_analyzer(task="sentiment", lang="es")
        self.stemmer = None
        self.spanish_stopwords = set()
        
        # Tourism-related keywords for enhanced processing
        self.tourism_keywords = {
            'attractions': ['museo', 'catedral', 'iglesia', 'palacio', 'castillo', 'parque', 'plaza', 'mercado'],
            'activities': ['visitar', 'ver', 'conocer', 'explorar', 'caminar', 'disfrutar', 'admirar'],
            'sentiments': ['hermoso', 'bonito', 'increíble', 'impresionante', 'maravilloso', 'espectacular'],
            'time_expressions': ['mañana', 'tarde', 'noche', 'día', 'hora', 'tiempo', 'duración'],
            'locations': ['centro', 'histórico', 'antiguo', 'moderno', 'tradicional', 'típico']
        }
        
        self._init_nlp_components()
    
    def _init_nlp_components(self):
        """Initialize essential NLP components for query processing."""
        try:
            # Download required NLTK data
            self._download_nltk_data()
            
            # Get stopwords using NLTK
            self.spanish_stopwords = self._get_nltk_stopwords()

            # Initialize NLTK stemmer for Spanish
            self.stemmer = SnowballStemmer('spanish')
            
            logger.info("NLP components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing NLP components: {e}")
            self.stemmer = None

    def _download_nltk_data(self):
        """Download required NLTK data packages."""
        nltk_downloads = ['stopwords', 'punkt', 'vader_lexicon', 'wordnet', 'averaged_perceptron_tagger']
        
        for package in nltk_downloads:
            try:
                if package == 'vader_lexicon':
                    nltk.data.find('vader_lexicon')
                elif package in ['stopwords', 'punkt']:
                    nltk.data.find(f'tokenizers/{package}')
                else:
                    nltk.data.find(f'corpora/{package}')
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                    logger.info(f"Downloaded NLTK package: {package}")
                except Exception as e:
                    logger.warning(f"Failed to download NLTK package {package}: {e}")

    def _get_nltk_stopwords(self) -> Set[str]:
        """Get Spanish and English stopwords using NLTK."""
        try:
            # Get Spanish stopwords
            spanish_stops = set(stopwords.words('spanish'))
            
            # Get English stopwords as fallback
            try:
                english_stops = set(stopwords.words('english'))
            except:
                english_stops = set()
            
            # Combine both sets
            combined_stops = spanish_stops.union(english_stops)
            
            # Add some custom tourism-related stopwords
            custom_stops = {
                'www', 'http', 'https', 'com', 'org', 'es', 'html', 'php',
                'turismo', 'tourism', 'tourist', 'guide', 'guía', 'información',
                'info', 'página', 'page', 'web', 'site', 'sitio'
            }
            
            combined_stops.update(custom_stops)
            
            logger.info(f"Loaded {len(combined_stops)} stopwords from NLTK")
            return combined_stops
            
        except Exception as e:
            logger.error(f"Error loading NLTK stopwords: {e}")
            # Fallback to a minimal set
            return {'de', 'la', 'el', 'en', 'y', 'a', 'que', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'un', 'una'}
    
    def extract_aspects(self, text: str, n_aspects: int = 3) -> List[Tuple[str, float]]:
        """Extrae los aspectos clave de un texto con sus puntuaciones"""
        try:
            aspects = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                top_n=n_aspects,
                stop_words=list(self.spanish_stopwords))
            return aspects
        except Exception as e:
            logger.error(f"Error extracting aspects: {e}")
            return []

    def analyze_aspect_sentiment(self, aspect: str, context: str) -> Dict[str, Any]:
        """Analiza el sentimiento hacia un aspecto específico en un contexto"""
        combined_text = f"{aspect} : {context}"
        sentiment_result = self.analyze_query_sentiment(combined_text)
        
        return {
            'sentiment': sentiment_result['sentiment'].upper()[:3],  # POS, NEG, NEU
            'probas': {
                'POS': sentiment_result['scores']['pos'],
                'NEG': sentiment_result['scores']['neg'],
                'NEU': sentiment_result['scores']['neu']
            },
            'score': sentiment_result['scores']['compound']
        }

    def get_sentiment_vector(self, text: str) -> np.ndarray:
        """
        Método optimizado para obtener solo el vector de sentimiento (4 dimensiones)
        """
        if not text or not isinstance(text, str) or not text.strip():
            return np.array([0.0, 0.0, 1.0, 0.0])  # neutral por defecto
        
        try:
            analysis = self.sentiment_analyzer.predict(text)
            probas = analysis.probas
            
            return np.array([
                probas.get('POS', 0.0),
                probas.get('NEG', 0.0),
                probas.get('NEU', 1.0),
                probas.get('POS', 0.0) - probas.get('NEG', 0.0)  # compound
            ])
        except Exception as e:
            logger.error(f"Error in sentiment vector calculation: {e}")
            return np.array([0.0, 0.0, 1.0, 0.0])

    def analyze_query_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment using pysentimiento for Spanish.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {
                'sentiment': 'neutral', 
                'confidence': 1.0, 
                'scores': {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
            }

        try:
            # Use pysentimiento analyzer
            analysis = self.sentiment_analyzer.predict(text)
            
            # Map sentiment to a standard format
            sentiment_map = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
            sentiment = sentiment_map.get(analysis.output, 'neutral')
            
            # Get probabilities
            probas = analysis.probas
            confidence = probas.get(analysis.output, 0.0)
            
            # Create a VADER-like score structure for consistency
            scores = {
                'pos': probas.get('POS', 0.0),
                'neg': probas.get('NEG', 0.0),
                'neu': probas.get('NEU', 0.0),
                # Compound score: weighted difference between positive and negative
                'compound': probas.get('POS', 0.0) - probas.get('NEG', 0.0)
            }
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'scores': scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment with pysentimiento: {e}")
            return {
                'sentiment': 'neutral', 
                'confidence': 1.0, 
                'scores': {'pos': 0.0, 'neg': 0.0, 'neu': 1.0, 'compound': 0.0}
            }

    def preprocess_text(self, text: str) -> str:
        """Preprocess text using advanced NLTK techniques."""
        if not text:
            return ""
        
        try:
            # Tokenize with NLTK
            tokens = word_tokenize(text, language='spanish')
            
            # POS tagging to identify meaningful words
            pos_tags = pos_tag(tokens)
            
            # Keep only meaningful parts of speech and filter stopwords
            processed_tokens = []
            for token, pos in pos_tags:
                # Keep nouns, adjectives, verbs, and proper nouns
                if (pos.startswith(('NN', 'JJ', 'VB', 'NNP')) and 
                    token.isalpha() and 
                    len(token) > 2 and
                    token.lower() not in self.spanish_stopwords):
                    
                    # Apply stemming for better matching
                    if self.stemmer:
                        stemmed = self.stemmer.stem(token.lower())
                        processed_tokens.append(stemmed)
                    else:
                        processed_tokens.append(token.lower())
            
            # Join tokens back into text
            processed_text = ' '.join(processed_tokens)
            
            # If processing resulted in empty text, return original lowercased
            return processed_text if processed_text.strip() else text.lower()
            
        except Exception as e:
            logger.error(f"Error preprocessing text with NLTK: {e}")
            # Fallback to simple preprocessing
            try:
                tokens = word_tokenize(text, language='spanish')
                filtered_tokens = [
                    token.lower() for token in tokens 
                    if token.isalpha() and 
                    len(token) > 2 and 
                    token.lower() not in self.spanish_stopwords
                ]
                return ' '.join(filtered_tokens) if filtered_tokens else text.lower()
            except:
                return text.lower()

    def extract_query_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract relevant keywords using NLTK POS tagging and NER."""
        if not text:
            return []
        
        try:
            # Tokenize and POS tag
            tokens = word_tokenize(text, language='spanish')
            pos_tags = pos_tag(tokens)
            
            # Extract nouns, adjectives, and proper nouns
            keywords = []
            for word, pos in pos_tags:
                if (pos.startswith('NN') or pos.startswith('JJ') or pos.startswith('NNP')) and \
                   len(word) > 3 and word.lower() not in self.spanish_stopwords:
                    
                    # Check if word is tourism-related or significant
                    is_tourism_related = any(
                        word.lower() in category_words 
                        for category_words in self.tourism_keywords.values()
                    )
                    
                    if is_tourism_related or len(word) > 5:
                        keywords.append(word.lower())
            
            # Remove duplicates and limit
            unique_keywords = list(dict.fromkeys(keywords))
            return unique_keywords[:max_keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords with NLTK: {e}")
            # Fallback to simple method
            words = text.split()
            return [w.lower() for w in words if len(w) > 4 and w.lower() not in self.spanish_stopwords][:max_keywords]

    def create_search_query_from_preferences(self, user_preferences: Dict) -> str:
        """Create a search query from user preferences using NLP processing with sentiment analysis."""
        query_parts = []
        
        if 'city' in user_preferences:
            query_parts.append(f"tourist attractions in {user_preferences['city']}")
            
        if 'category_interest' in user_preferences:
            high_interest_categories = [
                category for category, score in user_preferences['category_interest'].items()
                if score >= 4
            ]
            if high_interest_categories:
                query_parts.append(f"interested in {', '.join(high_interest_categories)}")
                
        if 'user_notes' in user_preferences and user_preferences['user_notes']:
            # Process user notes with NLP to extract relevant keywords and sentiment
            processed_notes = self.preprocess_text(user_preferences['user_notes'])
            keywords = self.extract_query_keywords(processed_notes, max_keywords=5)
            sentiment = self.analyze_query_sentiment(user_preferences['user_notes'])
            
            # Build query based on keywords and sentiment
            if keywords:
                keyword_query = ' '.join(keywords)
                
                # Enhance query based on sentiment
                if sentiment['sentiment'] == 'positive' and sentiment['confidence'] > 0.3:
                    # User is enthusiastic, boost positive descriptors
                    keyword_query += " amazing wonderful excellent"
                elif sentiment['sentiment'] == 'negative' and sentiment['confidence'] > 0.3:
                    # User has concerns, focus on quality and avoid negative aspects
                    keyword_query += " quality recommended peaceful"
                
                query_parts.append(keyword_query)
            else:
                query_parts.append(processed_notes)
                
        return ". ".join(query_parts)