"""
Similarity Calculator for embeddings and place ranking
Handles embedding generation and similarity calculations
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """Handles embedding generation and similarity calculations"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", nlp_processor=None):
        """Initialize similarity calculator with sentence transformer model"""
        self.model = SentenceTransformer(model_name)
        self.nlp_processor = nlp_processor
    
    def generate_place_embeddings(self, places: List[Dict]) -> List[np.ndarray]:
        """Genera embeddings enriquecidos con información de aspectos y sentimiento"""
        enriched_embeddings = []
        for place in places:
            # Extraer aspectos clave
            aspects = self.nlp_processor.extract_aspects(place['description'])
            aspects_text = " ".join([aspect[0] for aspect in aspects]) if aspects else ""
            
            # Texto enriquecido para el embedding
            text_representation = f"""
Nombre: {place['name']}
Aspectos clave: {aspects_text}
Descripción: {place.get('description', '')}
Categoría: {place.get('category', '')}
            """.strip()
            
            # Generar embedding semántico
            semantic_embedding = self.model.encode(text_representation)
            
            # Usar el método optimizado para obtener directamente el vector de sentimiento
            sentiment_vector = self.nlp_processor.get_sentiment_vector(place.get('description', ''))
            
            # Combinar embedding semántico y de sentimiento
            combined_embedding = np.concatenate([semantic_embedding, sentiment_vector])
            enriched_embeddings.append(combined_embedding)
            
        return enriched_embeddings
        
    def generate_user_embedding(self, user_preferences: Dict) -> np.ndarray:
        """Generate embedding with sentiment enhancement for user query"""
        preferences_text = []
        
        # Procesar preferencias de categoría con sentimiento explícito y pesos
        if 'category_interest' in user_preferences:
            for category, score in user_preferences['category_interest'].items():
                if score == 5:
                    preferences_text.append(f"Me encanta completamente {category}")
                    # Repetir para dar más peso a las preferencias altas
                    preferences_text.append(f"Busco especialmente {category}")
                    preferences_text.append(f"Prefiero {category}")
                elif score == 4:
                    preferences_text.append(f"Realmente me gusta {category}")
                    preferences_text.append(f"Me interesa mucho {category}")
                elif score == 3:
                    preferences_text.append(f"Tengo interés moderado en {category}")
                elif score == 2:
                    preferences_text.append(f"No me interesa mucho {category}")
                    preferences_text.append(f"Evito un poco {category}")
                elif score == 1:
                    preferences_text.append(f"Odio {category}")
                    preferences_text.append(f"Evito completamente {category}")
                    preferences_text.append(f"No quiero {category}")
        
        # Procesar notas del usuario con análisis de sentimiento
        user_notes_text = ""
        if 'user_notes' in user_preferences and user_preferences['user_notes']:
            notes = user_preferences['user_notes']
            preferences_text.append(notes)
            # Dar más peso a las notas del usuario repitiéndolas
            preferences_text.append(notes)
            user_notes_text = notes
        
        # Si no hay texto de preferencias, crear uno más específico basado en las categorías
        if not preferences_text:
            # Crear texto específico basado en las puntuaciones de categorías
            if 'category_interest' in user_preferences:
                high_cats = [cat for cat, score in user_preferences['category_interest'].items() if score >= 4]
                low_cats = [cat for cat, score in user_preferences['category_interest'].items() if score <= 2]
                
                if high_cats and low_cats:
                    preferences_text.append(f"Busco específicamente {' y '.join(high_cats)}")
                    preferences_text.append(f"Evito completamente {' y '.join(low_cats)}")
                elif high_cats:
                    preferences_text.append(f"Solo me interesan {' y '.join(high_cats)}")
                elif low_cats:
                    preferences_text.append(f"No quiero {' y '.join(low_cats)}")
                else:
                    preferences_text.append("Busco lugares turísticos interesantes")
            else:
                preferences_text.append("Busco lugares turísticos interesantes")
        
        # Generar embedding semántico
        full_text = ". ".join(preferences_text)
        logger.info(f"🔍 User preference text for embedding: {full_text[:200]}...")
        semantic_embedding = self.model.encode(full_text)
        
        # Usar el método optimizado para obtener el vector de sentimiento
        # Si hay notas del usuario, usar esas; si no, usar todo el texto de preferencias
        sentiment_text = user_notes_text if user_notes_text else full_text
        sentiment_vector = self.nlp_processor.get_sentiment_vector(sentiment_text)
        
        logger.info(f"📊 User embedding dimensions: semantic={len(semantic_embedding)}, sentiment={len(sentiment_vector)}")
        return np.concatenate([semantic_embedding, sentiment_vector])

    def calculate_category_penalty(self, user_prefs: Dict, place: Dict) -> float:
        """
        Calcula una penalización basada en las categorías que el usuario no le gustan
        """
        if 'category_interest' not in user_prefs:
            return 1.0
        
        place_category = place.get('category', '').lower()
        if not place_category:
            return 1.0
        
        # Mapear categorías de lugares a categorías de preferencias
        category_mapping = {
            'history': ['history', 'sitio histórico', 'histórico', 'museo', 'catedral', 'iglesia', 'castillo'],
            'religion': ['religion', 'religioso', 'catedral', 'iglesia', 'basílica', 'monasterio', 'convento'],
            'culture': ['culture', 'cultural', 'museo', 'catedral', 'iglesia', 'arte', 'teatro'],
            'urban': ['urban', 'urbano', 'ciudad', 'plaza', 'calle', 'barrio', 'centro'],
            'nature': ['nature', 'natural', 'sitio natural', 'parque', 'meadows', 'jardín', 'bosque'],
            'beach': ['beach', 'playa', 'sea', 'beaches', 'costa', 'mar'],
            'food': ['food', 'mercado', 'market', 'restaurante', 'gastronomía', 'comida'],
            'shopping': ['shopping', 'mercado', 'market', 'tienda', 'comercio', 'compras'],
            'entertainment': ['entertainment', 'entretenimiento', 'diversión', 'espectáculo', 'ocio']
        }
        
        # Encontrar qué categoría de preferencia corresponde al lugar
        place_pref_category = None
        for pref_cat, keywords in category_mapping.items():
            if any(keyword in place_category for keyword in keywords):
                place_pref_category = pref_cat
                break
        
        if place_pref_category and place_pref_category in user_prefs['category_interest']:
            score = user_prefs['category_interest'][place_pref_category]
            
            # Convertir puntuación a multiplicador
            if score == 5:
                return 1.5  # Boost para categorías favoritas
            elif score == 4:
                return 1.2
            elif score == 3:
                return 1.0  # Neutral
            elif score == 2:
                return 0.5  # Penalización moderada
            elif score == 1:
                return 0.1  # Penalización fuerte
        
        return 1.0  # Sin penalización si no hay mapeo

    def calculate_enhanced_similarity(self, user_prefs: Dict, place: Dict) -> float:
        """
        Calcula una puntuación mejorada que combina:
        - Similitud semántica de aspectos clave
        - Alineamiento de sentimiento
        """
        # Extraer aspectos del lugar
        place_aspects = self.nlp_processor.extract_aspects(place['description'])
        if not place_aspects:
            return 0.0
        
        # Extraer aspectos de las preferencias del usuario con mayor énfasis en preferencias negativas
        category_texts = []
        if 'category_interest' in user_prefs:
            for category, score in user_prefs.get('category_interest', {}).items():
                if score == 5:
                    category_texts.append(f"me encanta {category}")
                    category_texts.append(f"busco {category}")
                elif score == 4:
                    category_texts.append(f"me gusta mucho {category}")
                elif score == 3:
                    category_texts.append(f"me interesa {category}")
                elif score == 2:
                    category_texts.append(f"no me interesa mucho {category}")
                elif score == 1:
                    category_texts.append(f"odio {category}")
                    category_texts.append(f"evito {category}")
        
        user_text = f"{user_prefs.get('user_notes', '')} {' '.join(category_texts)}"
        user_aspects = self.nlp_processor.extract_aspects(user_text)
        
        # Calcular similitud semántica entre aspectos
        semantic_sim = self._aspect_similarity(user_aspects, place_aspects)
        
        # Calcular alineamiento de sentimiento
        sentiment_alignment = self._sentiment_alignment(user_prefs, place, place_aspects)
        
        # Calcular penalización por categoría
        category_penalty = self.calculate_category_penalty(user_prefs, place)
        
        # Combinar con pesos y aplicar penalización
        base_similarity = 0.6 * semantic_sim + 0.4 * sentiment_alignment
        final_similarity = base_similarity * category_penalty
        
        return max(0.0, min(1.0, final_similarity))  # Asegurar rango [0, 1]

    def _aspect_similarity(self, user_aspects: List[Tuple[str, float]], 
                          place_aspects: List[Tuple[str, float]]) -> float:
        """Calcula la similitud semántica entre aspectos"""
        if not user_aspects or not place_aspects:
            return 0.0
        
        # Extraer solo los términos de los aspectos
        user_terms = [aspect[0] for aspect in user_aspects]
        place_terms = [aspect[0] for aspect in place_aspects]
        
        # Generar embeddings
        user_embs = self.model.encode(user_terms)
        place_embs = self.model.encode(place_terms)
        
        # Matriz de similitud
        sim_matrix = np.dot(user_embs, place_embs.T)
        
        # Promedio de máximas similitudes (soft alignment)
        max_sim_user = np.max(sim_matrix, axis=1).mean()
        max_sim_place = np.max(sim_matrix, axis=0).mean()
        
        return (max_sim_user + max_sim_place) / 2

    def _sentiment_alignment(self, user_prefs: Dict, place: Dict, 
                            place_aspects: List[Tuple[str, float]]) -> float:
        """Calcula el alineamiento de sentimiento entre usuario y lugar"""
        # Analizar sentimiento del usuario
        user_sentiment = self.nlp_processor.analyze_query_sentiment(user_prefs.get('user_notes', ''))
        user_score = user_sentiment['scores']['pos'] - user_sentiment['scores']['neg']
        
        # Analizar sentimiento del lugar (promedio ponderado por aspectos)
        place_sentiments = []
        for aspect, weight in place_aspects:
            aspect_sentiment = self.nlp_processor.analyze_aspect_sentiment(aspect, place['description'])
            place_sentiments.append(aspect_sentiment['score'] * weight)
        
        if not place_sentiments:
            return 0.0
            
        place_score = np.mean(place_sentiments)
        
        # Calcular alineamiento (1 - distancia)
        return 1 - abs(user_score - place_score)

    def calculate_cosine_similarity(self, user_embedding: np.ndarray, place_embeddings: List[np.ndarray], 
                                   user_preferences: Dict = None, places: List[Dict] = None) -> List[float]:
        """Enhanced cosine similarity with sentiment adjustment and category penalties"""
        similarities = []
        
        logger.info(f"🔍 Calculating similarity for {len(place_embeddings)} places")
        logger.info(f"📊 User embedding shape: {user_embedding.shape}")
        
        # Verificar que el embedding del usuario tenga el tamaño correcto
        if len(user_embedding) < 4:
            logger.error(f"❌ User embedding too small: {len(user_embedding)} dimensions")
            # Fallback: usar solo similitud semántica básica
            for place_emb in place_embeddings:
                if len(place_emb) >= len(user_embedding):
                    semantic_sim = cosine_similarity(
                        user_embedding.reshape(1, -1),
                        place_emb[:len(user_embedding)].reshape(1, -1)
                    )[0][0]
                    similarities.append(max(0.0, semantic_sim))
                else:
                    similarities.append(0.0)
            return similarities
        
        # Separar componentes del embedding del usuario
        user_semantic = user_embedding[:-4]  # Primeras N-4 dimensiones (MiniLM)
        user_sentiment = user_embedding[-4:]  # Últimas 4 dimensiones (sentimiento)
        
        logger.info(f"📊 User semantic dimensions: {len(user_semantic)}, sentiment: {len(user_sentiment)}")
        
        for i, place_emb in enumerate(place_embeddings):
            try:
                # Verificar que el embedding del lugar tenga el tamaño correcto
                if len(place_emb) < 4:
                    logger.warning(f"⚠️ Place {i} embedding too small: {len(place_emb)} dimensions")
                    similarities.append(0.0)
                    continue
                
                # Separar componentes del embedding del lugar
                place_semantic = place_emb[:-4]
                place_sentiment = place_emb[-4:]
                
                # Verificar que las dimensiones semánticas coincidan
                if len(user_semantic) != len(place_semantic):
                    logger.warning(f"��️ Dimension mismatch for place {i}: user={len(user_semantic)}, place={len(place_semantic)}")
                    # Usar la dimensión menor
                    min_dim = min(len(user_semantic), len(place_semantic))
                    user_sem_truncated = user_semantic[:min_dim]
                    place_sem_truncated = place_semantic[:min_dim]
                else:
                    user_sem_truncated = user_semantic
                    place_sem_truncated = place_semantic
                
                # 1. Calcular similitud semántica tradicional
                semantic_sim = cosine_similarity(
                    user_sem_truncated.reshape(1, -1),
                    place_sem_truncated.reshape(1, -1)
                )[0][0]
                
                # Asegurar que la similitud esté en rango válido
                semantic_sim = max(-1.0, min(1.0, semantic_sim))
                
                # 2. Calcular alineamiento de sentimiento
                user_pos, user_neg = user_sentiment[0], user_sentiment[1]
                place_pos, place_neg = place_sentiment[0], place_sentiment[1]

                # Premiar la coincidencia de sentimientos (positivo con positivo, negativo con negativo)
                sentiment_similarity = (user_pos * place_pos) + (user_neg * place_neg)
                
                # Penalizar la discrepancia de sentimientos (positivo con negativo)
                sentiment_opposition = (user_pos * place_neg) + (user_neg * place_pos)
                
                # El alineamiento va de -1 (totalmente opuesto) a 1 (totalmente alineado)
                alignment = sentiment_similarity - sentiment_opposition
                
                # Normalizar a rango [0, 1]
                sentiment_alignment = (alignment + 1) / 2
                sentiment_alignment = max(0.0, min(1.0, sentiment_alignment))
                
                # 3. Combinar con peso balanceado (70% semántica, 30% sentimiento)
                # Si el sentimiento es muy bajo, usar más peso semántico
                if sentiment_alignment < 0.3:
                    combined_sim = 0.9 * semantic_sim + 0.1 * sentiment_alignment
                else:
                    combined_sim = 0.7 * semantic_sim + 0.3 * sentiment_alignment
                
                # Aplicar penalización por categoría si está disponible
                if user_preferences and places and i < len(places):
                    category_penalty = self.calculate_category_penalty(user_preferences, places[i])
                    combined_sim = combined_sim * category_penalty
                
                # Asegurar que el resultado esté en rango válido
                combined_sim = max(0.0, min(1.0, combined_sim))
                similarities.append(combined_sim)
                
            except Exception as e:
                logger.error(f"❌ Error calculating similarity for place {i}: {e}")
                similarities.append(0.0)
        
        logger.info(f"✅ Calculated {len(similarities)} similarities, range: {min(similarities):.3f} - {max(similarities):.3f}")
        return similarities

    def filter_places_by_similarity(self, places: List[Dict], place_embeddings: List[np.ndarray],
                                   user_embedding: np.ndarray, user_preferences: Dict = None, 
                                   similarity_threshold: float = 0.1) -> Tuple[List[Dict], List[np.ndarray], List[float]]:
        """Filter places by enhanced cosine similarity"""
        similarities = self.calculate_cosine_similarity(user_embedding, place_embeddings, user_preferences, places)
        
        # Ordenar por similitud descendente
        sorted_data = sorted(zip(places, place_embeddings, similarities), 
                           key=lambda x: x[2], reverse=True)
        
        # Filtrar por umbral y desempaquetar
        filtered_data = [(p, e, s) for p, e, s in sorted_data if s >= similarity_threshold]
        
        if filtered_data:
            filtered_places, filtered_embeddings, filtered_similarities = zip(*filtered_data)
            return list(filtered_places), list(filtered_embeddings), list(filtered_similarities)
        return [], [], []