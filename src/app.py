import streamlit as st
import json
import os
from geopy.geocoders import Nominatim
from RAG.rag import RAGPlanner

# Opciones de categorías según mock
CATEGORIES = [
    ("engineering", "Ingeniería"),
    ("history", "Historia"),
    ("food", "Comida"),
    ("culture", "Cultura"),
    ("beach", "Playas"),
    ("shopping", "Compras"),
    ("nature", "Naturaleza"),
]

TRANSPORT_MODES = [
    "A pie",
    "Bicicleta",
    "Transporte público",
    "Coche/taxi",
    "Otro"
]

# Interpretación de las puntuaciones
SCORE_LABELS = {
    1: "No me gusta nada",
    2: "No me atrae",
    3: "Me es indiferente",
    4: "Me gusta",
    5: "Me encanta"
}

def get_available_cities():
    # Obtiene las ciudades únicas del archivo JSON
    try:
        # Buscar el archivo places.json en diferentes ubicaciones posibles
        possible_paths = [
            "places.json",
            "src/places.json",
            os.path.join(os.path.dirname(__file__), "places.json"),
            os.path.join(os.path.dirname(__file__), "..", "places.json")
        ]
        
        places_data = None
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    places_data = json.load(f)
                break
        
        if not places_data:
            st.error("No se pudo encontrar el archivo places.json")
            return []
        
        # Extraer ciudades únicas de los sitios turísticos
        cities = set()
        tourist_sites = places_data.get('touristSites', [])
        
        for site in tourist_sites:
            location = site.get('location', {})
            city = location.get('city', '')
            if city:
                cities.add(city)
        
        return sorted(list(cities))
        
    except Exception as e:
        st.warning(f"No se pudieron cargar las ciudades desde el archivo JSON: {e}")
        return []

def geocode_address(address, city):
    geolocator = Nominatim(user_agent="tourist_planner")
    try:
        location = geolocator.geocode(f"{address}, {city}")
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        st.warning(f"No se pudo geolocalizar la dirección: {e}")
    return None, None

def app():
    st.set_page_config(page_title="Planificador Turístico Inteligente", page_icon=":earth_africa:")
    st.title("Planificador Turístico Inteligente")
    st.markdown("¡Bienvenido! Personaliza tu experiencia turística en Europa.")

    # Selección de ciudad dinámica
    cities = get_available_cities()
    if not cities:
        st.error("No hay ciudades disponibles en el archivo de datos.")
        return
    city = st.selectbox("¿A qué ciudad viajas?", cities)

    # Tiempo disponible en horas
    available_hours = st.number_input("¿Cuántas horas tienes disponibles para turismo?", min_value=1, max_value=168, value=24, step=1, help="Ejemplo: 8 horas por día durante 3 días = 24 horas")

    # Preferencias de categorías con sliders de 1 a 5 y explicación
    st.markdown("### ¿Qué tipo de actividades prefieres? (Indica tu nivel de interés)")
    category_interest = {}
    for key, label in CATEGORIES:
        col1, col2 = st.columns([2, 3])
        with col1:
            interest = st.slider(
                f"{label}",
                min_value=1, max_value=5, value=3, step=1,
                help="1 = No me gusta nada, 5 = Me encanta"
            )
        with col2:
            st.markdown(f"**{SCORE_LABELS[interest]}**")
        category_interest[key] = interest

    # Medios de transporte
    st.markdown("### ¿Qué medios de transporte puedes utilizar?")
    selected_transport = st.multiselect(
        "Selecciona todos los que apliquen:",
        TRANSPORT_MODES,
        default=["A pie", "Transporte público"]
    )

    # Distancia máxima dispuesta a recorrer
    st.markdown("### ¿Cuánto estás dispuesto a alejarte de tu punto de partida?")
    max_distance = st.slider(
        "Distancia máxima (en kilómetros)",
        min_value=1, max_value=50, value=5, step=1
    )

    # Dirección de partida
    st.markdown("### ¿Cuál es tu punto de partida? (dirección o lugar conocido)")
    user_address = st.text_input(
        "Introduce tu dirección o punto de referencia (ejemplo: 'Gare de Lyon', 'Plaza Mayor', etc.)"
    )

    # Párrafo opcional
    st.markdown("### ¿Algo más que quieras contarnos? (opcional)")
    user_notes = st.text_area(
        "Puedes escribir un breve párrafo sobre tus intereses, necesidades especiales, etc.",
        placeholder="Me interesa la arquitectura moderna y la comida local..."
    )

    # Botón de envío
    if st.button("Buscar actividades recomendadas"):
        st.success("¡Gracias! Tus preferencias han sido registradas.")
        st.write("**Ciudad:**", city)
        st.write("**Horas disponibles:**", available_hours)
        st.write("**Preferencias por categoría:**", {label: f"{category_interest[key]} ({SCORE_LABELS[category_interest[key]]})" for key, label in CATEGORIES})
        st.write("**Medios de transporte:**", selected_transport)
        st.write("**Distancia máxima:**", f"{max_distance} km")
        st.write("**Dirección de partida:**", user_address if user_address else "(no especificada)")
        if user_notes.strip():
            st.write("**Notas adicionales:**", user_notes)
        else:
            st.write("**Notas adicionales:** (ninguna)")

        # Geocodificación de la dirección
        if user_address:
            lat, lon = geocode_address(user_address, city)
            if lat is not None and lon is not None:
                st.write(f"**Coordenadas de partida:** Lat: {lat:.5f}, Lon: {lon:.5f}")
            else:
                st.warning("No se pudo obtener la ubicación de la dirección proporcionada.")
        else:
            lat, lon = None, None

        # Initialize RAG system and process user request
        if lat is not None and lon is not None:
            try:
                from RAG.rag import RAGPlanner
                
                # Prepare user preferences for RAG
                user_preferences = {
                    'city': city,
                    'available_hours': available_hours,
                    'category_interest': category_interest,
                    'transport_modes': selected_transport,
                    'max_distance': max_distance,
                    'user_notes': user_notes.strip() if user_notes.strip() else ""
                }
                
                # Process the request
                with st.spinner("Procesando recomendaciones..."):
                    try:
                        # Initialize RAG planner (API keys loaded from .env)
                        rag_planner = RAGPlanner()
                        rag_data = rag_planner.process_user_request(user_preferences, lat, lon)
                    except ValueError as e:
                        st.error(f"Error de configuración: {e}")
                        st.error("Por favor, configura las API keys en tu archivo .env:")
                        st.code("""
OPENROUTER_API_KEY=tu_openrouter_key
OPENROUTESERVICE_API_KEY=tu_ors_key
                        """)
                        return
                    except RuntimeError as e:
                        st.error(f"Error de API: {e}")
                        st.error("Verifica que tus API keys sean válidas y tengas créditos disponibles.")
                        return
                
                if rag_data['filtered_places']:
                    st.success(f"¡Encontradas {len(rag_data['filtered_places'])} recomendaciones!")
                    
                    # Display LLM time estimates
                    if rag_data.get('llm_time_estimates'):
                        st.markdown("### ⏰ Tiempo recomendado por el LLM:")
                        for i, (place, time_hours) in enumerate(zip(rag_data['filtered_places'], rag_data['llm_time_estimates'])):
                            st.write(f"**{i+1}. {place['name']}:** {time_hours} horas")
                    
                    # Display filtered places
                    st.markdown("### 📍 Lugares recomendados:")
                    for i, place in enumerate(rag_data['filtered_places']):
                        time_estimate = rag_data['llm_time_estimates'][i] if rag_data.get('llm_time_estimates') and i < len(rag_data['llm_time_estimates']) else 'N/A'
                        with st.expander(f"{i+1}. {place['name']} ({time_estimate} horas recomendadas)"):
                            st.write(f"**Tipo:** {place.get('type', 'No especificado')}")
                            st.write(f"**Descripción:** {place.get('description', 'No disponible')}")
                            st.write(f"**Atractivo:** {place.get('visitorAppeal', 'No disponible')}")
                            st.write(f"**Clasificación:** {place.get('touristClassification', 'No especificada')}")
                            st.write(f"**Duración estimada original:** {place.get('estimatedVisitDuration', 'No especificada')}")
                            st.write(f"**Tiempo recomendado por LLM:** {time_estimate} horas")
                    
                    # Display distance matrix info
                    if rag_data['distance_matrix'].size > 0:
                        st.markdown("### 🚗 Matriz de distancias (Adyacencia):")
                        st.write(f"Matriz de tiempos de viaje calculada para {len(rag_data['filtered_places'])} lugares")
                        st.write("(Tiempos en minutos desde tu ubicación)")
                        
                        # Show travel times from user location to each place
                        if len(rag_data['distance_matrix']) > 1:
                            travel_times = rag_data['distance_matrix'][0][1:]  # First row, excluding user-to-user
                            for i, (place, time_minutes) in enumerate(zip(rag_data['filtered_places'], travel_times)):
                                st.write(f"- {place['name']}: {time_minutes:.1f} minutos")
                        
                        # Show full distance matrix
                        with st.expander("Ver matriz completa de distancias"):
                            st.write("Matriz de adyacencia (tiempos en minutos):")
                            st.dataframe(rag_data['distance_matrix'])
                    
                    # Display embeddings info
                    st.markdown("### 🧠 Información de Embeddings:")
                    st.write(f"- Embeddings de lugares generados: {len(rag_data.get('place_embeddings', []))}")
                    st.write(f"- Dimensiones del embedding del usuario: {len(rag_data.get('user_embedding', []))}")
                    
                    # Display LLM response
                    if rag_data.get('llm_response'):
                        with st.expander("Ver respuesta completa del LLM"):
                            st.text_area("Respuesta del LLM:", rag_data['llm_response'], height=300)
                    
                    # Display data for metaheuristic
                    with st.expander("Datos para Metaheurística (avanzado)"):
                        st.write("**Datos disponibles para algoritmos de optimización:**")
                        st.write(f"- Lugares filtrados: {len(rag_data['filtered_places'])}")
                        st.write(f"- Matriz de adyacencia: {rag_data['distance_matrix'].shape if hasattr(rag_data['distance_matrix'], 'shape') else 'N/A'}")
                        st.write(f"- Embeddings de lugares: {len(rag_data.get('place_embeddings', []))}")
                        st.write(f"- Embedding del usuario: {len(rag_data.get('user_embedding', []))}")
                        st.write(f"- Tiempos estimados por LLM: {len(rag_data.get('llm_time_estimates', []))}")
                        
                else:
                    st.warning("No se encontraron lugares que coincidan con tus criterios. Intenta aumentar la distancia máxima o cambiar tus preferencias.")
                    
            except Exception as e:
                st.error(f"Error al procesar las recomendaciones: {str(e)}")
                st.write("Por favor, verifica que todos los datos estén correctos e intenta nuevamente.")
        else:
            st.warning("No se pudo procesar la solicitud sin una ubicación válida.")

if __name__ == "__main__":
    app()
