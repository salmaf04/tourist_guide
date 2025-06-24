import numpy as np
import datetime
from sklearn.metrics.pairwise import cosine_similarity


def is_incomplete_or_outdated(rag_data, query=None, min_places=10, min_description_length=30, min_cosine_threshold=0.25):
    """
    Evalúa si los datos RAG son insuficientes o desactualizados según múltiples criterios, incluyendo similitud de cosenos.
    Args:
        rag_data (dict): Datos devueltos por el sistema RAG
        query (str, optional): Consulta original del usuario para validación contextual
        min_places (int): Mínimo de lugares esperados
        min_description_length (int): Caracteres mínimos para descripción
        min_cosine_threshold (float): Umbral mínimo de similitud de coseno
    Returns:
        tuple: (bool, str) donde el bool indica si hay problemas y el str el motivo
    """
    if not isinstance(rag_data, dict) or 'filtered_places' not in rag_data:
        return True, "Estructura de datos inválida"
    
    # Check if RAG explicitly indicates crawler is needed
    if rag_data.get('needs_crawler', False):
        return True, rag_data.get('llm_response', 'No data available - crawler needed')
    
    places = rag_data.get('filtered_places', [])
    if len(places) < min_places:
        return True, f"Sólo se encontraron {len(places)} lugares (mínimo {min_places} esperado)"
    empty_descriptions = 0
    outdated_count = 0
    current_year = datetime.datetime.now().year
    for place in places:
        description = place.get('description', '')
        if not description or description == 'No disponible' or len(description) < min_description_length:
            empty_descriptions += 1
        last_updated = place.get('last_updated')
        if last_updated:
            try:
                update_year = int(last_updated.split('-')[0]) if isinstance(last_updated, str) else last_updated.year
                if current_year - update_year > 2:
                    outdated_count += 1
            except (AttributeError, ValueError):
                pass
    reasons = []
    if empty_descriptions > len(places) * 0.3:
        reasons.append(f"{empty_descriptions} lugares con descripción incompleta")
    if outdated_count > len(places) * 0.5:
        reasons.append(f"{outdated_count} lugares con información desactualizada")
    if query:
        query_terms = set(query.lower().split())
        matched_terms = 0
        for place in places:
            place_text = ' '.join([str(v) for k, v in place.items() if k != 'metadata']).lower()
            if any(term in place_text for term in query_terms):
                matched_terms += 1
        if matched_terms < min(len(query_terms), len(places)):
            reasons.append("baja coincidencia con términos de búsqueda")
    user_emb = rag_data.get('user_embedding')
    place_embs = rag_data.get('place_embeddings')
    if user_emb is not None and place_embs is not None and len(place_embs) == len(places):
        user_emb = np.array(user_emb).reshape(1, -1)
        place_embs = np.array(place_embs)
        similarities = cosine_similarity(user_emb, place_embs)[0]
        avg_similarity = float(np.mean(similarities))
        if avg_similarity < min_cosine_threshold:
            reasons.append(f"baja similitud promedio usuario-lugares ({avg_similarity:.2f} < {min_cosine_threshold})")
    return (bool(reasons), "; ".join(reasons)) if reasons else (False, "Datos completos y actualizados")


def ensure_fresh_rag_data(rag_data, user_query, fetch_tourism_data, city, RAGPlanner, user_preferences, lat, lon, transport_mode):
    """
    Valida los datos RAG y, si son insuficientes, activa el crawler y reprocesa.
    Devuelve (rag_data_final, reason, triggered_crawler:bool)
    """
    import logging
    logger = logging.getLogger(__name__)
    
    is_incomplete, reason = is_incomplete_or_outdated(rag_data, user_query)
    triggered_crawler = False
    if is_incomplete:
        logger.info(f"Datos insuficientes para {city}. Activando crawler...")
        triggered_crawler = True
        crawler_success = fetch_tourism_data(city)
        if crawler_success:
            logger.info(f"Crawler completado con éxito para {city}")
            rag_planner = RAGPlanner()  # Usa la ruta por defecto que ya está configurada correctamente
            rag_data = rag_planner.process_user_request(user_preferences, lat, lon, transport_mode)
            is_incomplete, reason = is_incomplete_or_outdated(rag_data, user_query)
        else:
            logger.error(f"Crawler falló para {city}")
    return rag_data, reason, triggered_crawler
