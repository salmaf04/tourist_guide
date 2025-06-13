import streamlit as st
import json
import os
import numpy as np
from geopy.geocoders import Nominatim
from RAG.rag import RAGPlanner
# from crawler.crawler_manager import CrawlerManager  # Comentado - usando datos estáticos
from BestRoutes.meta_routes import RouteOptimizer

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
    "Caminar",
    "Bicicleta", 
    "Bus",
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

AVAILEABLE_CITIES = [
    'Madrid',
    'Barcelona',
    'Valencia',
    'Seville',
    'Bilbao',
    'Granada',
    'Toledo',
    'Salamanca',
    'Málaga',
    'San Sebastián'
]


def geocode_address(address, city):
    """Geocodifica una dirección"""
    geolocator = Nominatim(user_agent="tourist_planner")
    try:
        location = geolocator.geocode(f"{address}, {city}")
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        st.warning(f"No se pudo geolocalizar la dirección: {e}")
    return None, None

def calculate_route_metrics(route, adjacency_matrix, node_params, max_time_minutes, tourist_param, route_optimizer):
    """
    Calcula métricas detalladas para una ruta
    
    :param route: Lista de índices de nodos que representan la ruta
    :param adjacency_matrix: Matriz de adyacencia con tiempos de viaje
    :param node_params: Parámetros de los nodos
    :param max_time_minutes: Tiempo máximo disponible en minutos
    :param tourist_param: Parámetros del turista
    :param route_optimizer: Instancia del optimizador de rutas
    :return: Diccionario con métricas de la ruta
    """
    if not route or len(route) <= 1:
        return {
            'total_time': 0,
            'travel_time': 0,
            'visit_time': 0,
            'num_places': 0,
            'efficiency': 0,
            'within_time_limit': True,
            'goal_value': 0
        }
    
    # Calcular tiempo total usando el método del RouteFinder
    total_time = route_optimizer.RouteFinder.get_time(route, len(route))
    
    # Calcular tiempo de viaje (tiempo entre nodos)
    travel_time = 0
    for i in range(1, len(route)):
        if route[i-1] < len(adjacency_matrix) and route[i] < len(adjacency_matrix[0]):
            travel_time += adjacency_matrix[route[i-1]][route[i]]
    
    # Agregar tiempo de regreso al inicio
    if len(route) > 1 and route[-1] < len(adjacency_matrix) and route[0] < len(adjacency_matrix[0]):
        travel_time += adjacency_matrix[route[-1]][route[0]]
    
    # Calcular tiempo de visita (tiempo gastado en cada ubicación)
    visit_time = 0
    num_places = 0
    for i in range(1, len(route)):  # Saltar nodo de inicio
        if route[i] < len(node_params):
            visit_time += node_params[route[i]]['time']
            num_places += 1
    
    # Calcular eficiencia (valor objetivo por unidad de tiempo)
    goal_value = route_optimizer.RouteFinder.goal_func(route, max_time_minutes, tourist_param)
    efficiency = goal_value / max(total_time, 1)  # Evitar división por cero
    
    # Verificar si está dentro del límite de tiempo
    within_time_limit = total_time <= max_time_minutes
    
    return {
        'total_time': total_time,
        'travel_time': travel_time,
        'visit_time': visit_time,
        'num_places': num_places,
        'efficiency': efficiency,
        'within_time_limit': within_time_limit,
        'goal_value': goal_value
    }

def prepare_metaheuristic_data(rag_data, user_preferences):
    """
    Prepara los datos del RAG para ser usados por las metaheurísticas
    
    :param rag_data: Datos procesados por el RAG
    :param user_preferences: Preferencias del usuario
    :return: Diccionario con datos preparados para metaheurísticas
    """
    # Matriz de adyacencia (tiempos de viaje entre nodos)
    adjacency_matrix = rag_data['time_matrix'].tolist()
    
    # Preparar parámetros de nodos
    node_params = []
    
    # Añadir ubicación del usuario como nodo 0
    user_node = {
        'vector': np.array(rag_data['user_embedding']),
        'time': 0  # No se gasta tiempo en el punto de inicio
    }
    node_params.append(user_node)
    
    # Añadir cada lugar como un nodo
    for i, place in enumerate(rag_data['filtered_places']):
        place_embedding = np.array(rag_data['place_embeddings'][i])
        visit_time_hours = rag_data['llm_time_estimates'][i] if i < len(rag_data['llm_time_estimates']) else 2.0
        visit_time_minutes = visit_time_hours * 60  # Convertir a minutos
        
        node = {
            'vector': place_embedding,
            'time': visit_time_minutes,
            'place_name': place['name'],
            'place_data': place
        }
        node_params.append(node)
    
    # Convertir horas disponibles a minutos
    max_time_minutes = user_preferences['available_hours'] * 60
    
    # Parámetro del turista (usar embedding del usuario)
    tourist_param = np.array(rag_data['user_embedding'])
    
    return {
        'adjacency_matrix': adjacency_matrix,
        'node_params': node_params,
        'max_time_minutes': max_time_minutes,
        'tourist_param': tourist_param
    }

def evaluate_route(route, route_optimizer, rag_data):
    """
    Evalúa una ruta y calcula métricas de rendimiento
    
    :param route: Lista de índices de nodos que representan la ruta
    :param route_optimizer: Instancia del optimizador de rutas
    :param rag_data: Datos del RAG para obtener información de lugares
    :return: Diccionario con métricas de la ruta
    """
    if not route or len(route) <= 1:
        return {
            'total_time': 0,
            'num_places': 0,
            'goal_value': 0,
            'efficiency': 0,
            'within_time_limit': True,
            'places_info': []
        }
    
    # Calcular tiempo total y valor objetivo usando las funciones de la metaheurística
    total_time = route_optimizer.RouteFinder.get_time(route, len(route))
    goal_value = route_optimizer.RouteFinder.goal_func(route, route_optimizer.max_time, route_optimizer.tourist_param)
    
    # Calcular eficiencia
    efficiency = goal_value / max(total_time, 1)
    
    # Verificar si está dentro del límite de tiempo
    within_time_limit = total_time <= route_optimizer.max_time
    
    # Obtener información de los lugares en la ruta
    places_info = []
    for i, node_idx in enumerate(route[1:], 1):  # Saltar nodo de inicio
        if node_idx <= len(rag_data['filtered_places']):
            place_idx = node_idx - 1
            place = rag_data['filtered_places'][place_idx]
            visit_time = rag_data['llm_time_estimates'][place_idx] if place_idx < len(rag_data['llm_time_estimates']) else 2.0
            
            places_info.append({
                'name': place['name'],
                'description': place.get('description', ''),
                'visit_time': visit_time,
                'coordinates': place.get('coordinates'),
                'category': place.get('category', 'general')
            })
    
    return {
        'total_time': total_time,
        'num_places': len(places_info),
        'goal_value': goal_value,
        'efficiency': efficiency,
        'within_time_limit': within_time_limit,
        'places_info': places_info
    }

def display_route_optimization_results(rag_data, user_preferences, user_lat, user_lon, transport_mode):
    """Muestra los resultados de la optimización de rutas"""
    
    st.markdown("### 🚀 Optimización de Rutas con Metaheurística")
    
    with st.spinner("Calculando las mejores rutas..."):
        try:
            # Preparar datos para la metaheurística
            meta_data = prepare_metaheuristic_data(rag_data, user_preferences)
            
            # IMPRIMIR MATRIZ DE TIEMPOS EN TERMINAL PARA REVISIÓN
            print("\n" + "="*80)
            print("🕒 MATRIZ DE TIEMPOS DE VIAJE (minutos)")
            print("="*80)
            adjacency_matrix = meta_data['adjacency_matrix']
            
            # Crear nombres de nodos para mejor legibilidad
            node_names = ['Inicio']
            for i, place in enumerate(rag_data['filtered_places']):
                node_names.append(f"{place['name'][:15]}...")  # Truncar nombres largos
            
            # Imprimir encabezados
            print(f"{'':>15}", end="")
            for i, name in enumerate(node_names):
                print(f"{name:>15}", end="")
            print()
            
            # Imprimir matriz con nombres de filas
            for i, row in enumerate(adjacency_matrix):
                print(f"{node_names[i]:>15}", end="")
                for j, time_val in enumerate(row):
                    print(f"{time_val:>15.1f}", end="")
                print()
            
            print("="*80)
            print(f"📊 Dimensiones: {len(adjacency_matrix)}x{len(adjacency_matrix[0])}")
            print(f"🎯 Lugares incluidos: {len(rag_data['filtered_places'])}")
            print("\n📍 ÍNDICES DE NODOS:")
            for i, name in enumerate(node_names):
                if i == 0:
                    print(f"   {i}: {name} (Punto de partida)")
                else:
                    place = rag_data['filtered_places'][i-1]
                    category = place.get('touristClassification', 'N/A')
                    print(f"   {i}: {place['name']} ({category})")
            print("="*80)
            
            # IMPRIMIR INFORMACIÓN DE NODOS
            print("\n🎯 PARÁMETROS DE NODOS:")
            print("="*80)
            node_params = meta_data['node_params']
            for i, node in enumerate(node_params):
                if i == 0:
                    print(f"Nodo {i} (Inicio):")
                    print(f"   - Vector: {node['vector'][:5]}... (dim: {len(node['vector'])})")
                    print(f"   - Tiempo de visita: {node['time']} min")
                else:
                    place_name = node.get('place_name', f'Lugar {i}')
                    print(f"Nodo {i} ({place_name}):")
                    print(f"   - Vector: {node['vector'][:5]}... (dim: {len(node['vector'])})")
                    print(f"   - Tiempo de visita: {node['time']} min")
                    if 'place_data' in node:
                        category = node['place_data'].get('touristClassification', 'N/A')
                        print(f"   - Categoría: {category}")
                print()
            
            print(f"🧭 Vector del turista: {meta_data['tourist_param'][:5]}... (dim: {len(meta_data['tourist_param'])})")
            print(f"⏰ Tiempo máximo: {meta_data['max_time_minutes']} minutos ({meta_data['max_time_minutes']/60:.1f} horas)")
            print("="*80 + "\n")
            
            # Inicializar optimizador de rutas
            route_optimizer = RouteOptimizer(
                adjacency_matrix=meta_data['adjacency_matrix'],
                node_params=meta_data['node_params'],
                max_time=meta_data['max_time_minutes'],
                tourist_param=meta_data['tourist_param']
            )
            
            # Obtener rutas optimizadas de la metaheurística
            all_routes = route_optimizer.get_routes()
            
            # Evaluar todas las rutas con la función objetivo tradicional
            route_evaluations = []
            for route in all_routes:
                goal_value = route_optimizer.RouteFinder.goal_func(route, meta_data['max_time_minutes'], meta_data['tourist_param'])
                route_evaluations.append({
                    'route': route,
                    'goal_value': goal_value
                })
            
            # Ordenar por función objetivo (mayor es mejor) y tomar las 10 mejores
            route_evaluations.sort(key=lambda x: x['goal_value'], reverse=True)
            top_10_routes = [eval_data['route'] for eval_data in route_evaluations[:10]]
            
            st.info(f"📊 Evaluadas {len(all_routes)} rutas por metaheurística. Simulando las 10 mejores...")
            
            # Simular solo las 10 mejores rutas para obtener las mejores por satisfacción
            from route_simulator import simulate_and_rank_routes
            optimized_routes_with_details = simulate_and_rank_routes(
                routes=top_10_routes,
                node_params=meta_data['node_params'],
                top_n=3,
                simulation_steps=3,
                tourist_name="Turista_Simulado"
            )
            optimized_routes = [result['route'] for result in optimized_routes_with_details]
            
            if optimized_routes:
                st.success(f"¡Se encontraron {len(optimized_routes)} rutas optimizadas!")
                
                # Evaluar cada ruta
                route_evaluations = []
                for route in optimized_routes:
                    metrics = evaluate_route(route, route_optimizer, rag_data)
                    route_evaluations.append({
                        'route': route,
                        'metrics': metrics
                    })
                
                # Ordenar por valor objetivo (mayor es mejor)
                route_evaluations.sort(key=lambda x: x['metrics']['goal_value'], reverse=True)
                
                # Mostrar las mejores 3 rutas
                best_routes = route_evaluations[:3]
                
                # Mostrar las mejores rutas con información de simulación
                for idx, (route_data, sim_data) in enumerate(zip(best_routes, optimized_routes_with_details)):
                    route = route_data['route']
                    metrics = route_data['metrics']
                    
                    # Obtener datos de simulación
                    satisfaction_score = sim_data['satisfaction_score']
                    simulation_success = sim_data.get('simulation_success', True)
                    
                    # Calcular puntuación de calidad combinada
                    quality_score = metrics['goal_value']
                    time_compliance = "✅" if metrics['within_time_limit'] else "⚠️"
                    sim_status = "🎭" if simulation_success else "⚠️"
                    
                    with st.expander(f"🏆 Ruta {idx + 1} - Satisfacción: {satisfaction_score:.1f}/10 {sim_status} | Valor: {quality_score:.2f} {time_compliance}"):
                        # Información general de la ruta con simulación
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Satisfacción", f"{satisfaction_score:.1f}/10")
                        with col2:
                            st.metric("Tiempo Total", f"{metrics['total_time']/60:.1f}h")
                        with col3:
                            st.metric("Lugares", f"{metrics['num_places']}")
                        with col4:
                            st.metric("Interacciones", f"{sim_data['total_interactions']}")
                        
                        # Información de simulación
                        st.write("**🎭 Resultados de Simulación:**")
                        st.write(f"- 😊 Satisfacción del turista: {satisfaction_score:.1f}/10")
                        st.write(f"- 🗣️ Interacciones totales: {sim_data['total_interactions']}")
                        st.write(f"- 📍 Lugares visitados: {sim_data['num_places_visited']}")
                        
                        # Mostrar recuerdos significativos si existen
                        if sim_data['significant_memories']:
                            st.write("**💭 Recuerdos más significativos:**")
                            for memory in sim_data['significant_memories'][:3]:  # Mostrar solo los primeros 3
                                st.write(f"   • {memory}")
                        
                        # Información de tiempo
                        st.write("**⏰ Información de tiempo:**")
                        st.write(f"- Tiempo total de ruta: {metrics['total_time']/60:.1f}h")
                        st.write(f"- Tiempo disponible: {user_preferences['available_hours']}h")
                        st.write(f"- Eficiencia metaheurística: {metrics['efficiency']:.3f}")
                        
                        # Secuencia de lugares
                        st.write("**📍 Secuencia de lugares:**")
                        st.write("🏠 **Inicio:** Tu ubicación")
                        
                        for place_info in metrics['places_info']:
                            st.write(f"📍 **{place_info['name']}** - {place_info['visit_time']}h")
                            if place_info['description']:
                                st.write(f"   _{place_info['description'][:100]}..._")
                            st.write(f"   🏷️ Categoría: {place_info['category']}")
                        
                        st.write("🏠 **Regreso:** Tu ubicación")
                        
                        # Indicadores de calidad
                        col1, col2 = st.columns(2)
                        with col1:
                            if metrics['within_time_limit']:
                                st.success("✅ Tiempo: Se ajusta a tu disponibilidad")
                            else:
                                st.warning("⚠️ Tiempo: Excede tu disponibilidad")
                        
                        with col2:
                            if simulation_success:
                                if satisfaction_score >= 7:
                                    st.success("🎭 Simulación: Experiencia excelente")
                                elif satisfaction_score >= 5:
                                    st.info("🎭 Simulación: Experiencia buena")
                                else:
                                    st.warning("🎭 Simulación: Experiencia mejorable")
                            else:
                                st.error("🎭 Simulación: Error en evaluación")
                        
                        # Mostrar lugares en mapa (si es posible)
                        try:
                            route_places = []
                            for place_info in metrics['places_info']:
                                if place_info['coordinates']:
                                    route_places.append({
                                        'name': place_info['name'],
                                        'lat': place_info['coordinates'][0],
                                        'lon': place_info['coordinates'][1]
                                    })
                            
                            if route_places:
                                st.write("**🗺️ Ubicaciones en el mapa:**")
                                st.map(route_places)
                        except Exception as e:
                            st.write("No se pudo mostrar el mapa de la ruta")
                
                # Mostrar detalles de optimización
                with st.expander("📊 Detalles de la Optimización y Simulación"):
                    st.write("**🔬 Proceso de Optimización:**")
                    st.write("1. **Metaheurística:** Simulated Annealing genera rutas candidatas")
                    st.write("2. **Evaluación:** Se evalúan todas las rutas con función objetivo")
                    st.write("3. **Preselección:** Se eligen las 10 mejores rutas por función objetivo")
                    st.write("4. **Simulación:** Las 10 mejores se evalúan con agentes virtuales")
                    st.write("5. **Selección final:** Se eligen las 3 rutas con mayor satisfacción simulada")
                    
                    st.write("**📈 Parámetros del proceso:**")
                    st.write(f"- Lugares considerados: {len(rag_data['filtered_places'])}")
                    st.write(f"- Tiempo máximo: {user_preferences['available_hours']} horas")
                    st.write(f"- Modo de transporte: {transport_mode}")
                    st.write(f"- Rutas generadas por metaheurística: {len(all_routes)}")
                    st.write(f"- Rutas preseleccionadas por función objetivo: 10")
                    st.write(f"- Rutas simuladas: {len(top_10_routes)}")
                    st.write(f"- Mejores rutas mostradas: {len(optimized_routes)}")
                    
                    # Información sobre los parámetros de la metaheurística
                    st.write("**🎯 Parámetros de la metaheurística:**")
                    st.write(f"- Nodos totales: {len(meta_data['node_params'])}")
                    st.write(f"- Dimensión del embedding: {len(meta_data['tourist_param'])}")
                    st.write(f"- Tiempo máximo (minutos): {meta_data['max_time_minutes']}")
                    
                    # Información sobre la simulación
                    st.write("**🎭 Parámetros de la simulación:**")
                    st.write("- Pasos de simulación por lugar: 3")
                    st.write("- Tipos de agentes: guías, meseros, curadores, etc.")
                    st.write("- Sistema de satisfacción: 0-10 puntos")
                    st.write("- Interacciones por lugar: 1-2 por paso")
                    
                    # Mostrar estadísticas de simulación si están disponibles
                    if optimized_routes_with_details:
                        st.write("**📊 Estadísticas de simulación:**")
                        satisfactions = [r['satisfaction_score'] for r in optimized_routes_with_details]
                        interactions = [r['total_interactions'] for r in optimized_routes_with_details]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Satisfacción promedio", f"{sum(satisfactions)/len(satisfactions):.1f}/10")
                        with col2:
                            st.metric("Interacciones promedio", f"{sum(interactions)/len(interactions):.0f}")
                        with col3:
                            st.metric("Rutas exitosas", f"{sum(1 for r in optimized_routes_with_details if r.get('simulation_success', True))}/{len(optimized_routes_with_details)}")
                    
                    # Mostrar matriz de tiempos (solo una muestra si es muy grande)
                    st.write("**⏱️ Matriz de tiempos de viaje (minutos):**")
                    if len(rag_data['time_matrix']) <= 10:
                        st.dataframe(rag_data['time_matrix'])
                    else:
                        st.write(f"Matriz de {rag_data['time_matrix'].shape[0]}x{rag_data['time_matrix'].shape[1]} (muy grande para mostrar)")
            
            else:
                st.warning("No se pudieron generar rutas optimizadas. Intenta ajustar tus preferencias.")
                
        except Exception as e:
            st.error(f"Error en la optimización de rutas: {str(e)}")
            
def app():
    st.set_page_config(page_title="Planificador Turístico Inteligente", page_icon="🌍")
    st.title("🌍 Planificador Turístico Inteligente")
    st.markdown("¡Bienvenido! Descubre las mejores rutas turísticas personalizadas con IA.")

    # Selección de ciudad con carga dinámica
    st.markdown("### 📍 Destino")
    cities = AVAILEABLE_CITIES
    
    if not cities:
        st.error("❌ No hay ciudades disponibles. Por favor, verifica la configuración del sistema.")
        return
    
    # Selección única de ciudad
    city = st.selectbox(
        "¿A qué ciudad viajas?", 
        cities, 
        help="Selecciona tu destino turístico. Más ciudades están disponibles para elegir."
    )
    
    if not city:
        st.warning("⚠️ Por favor selecciona una ciudad.")
        return
    
    # Configuración del viaje
    st.markdown("Configuración del Viaje")
    
    col1, col2 = st.columns(2)
    
    with col1:
        available_hours = st.number_input(
            "Horas disponibles para turismo", 
            min_value=1, max_value=168, value=8, step=1,
            help="Ejemplo: 8 horas por día"
        )
    
    with col2:
        max_distance = st.slider(
            "Distancia máxima (km)", 
            min_value=1, max_value=50, value=10, step=1,
            help="Qué tan lejos estás dispuesto a viajar desde tu punto de partida"
        )

    # Preferencias de categorías
    st.markdown("Preferencias de Actividades")
    st.markdown("*Indica tu nivel de interés en cada tipo de actividad:*")
    
    category_interest = {}
    
    # Mostrar categorías en dos columnas
    col1, col2 = st.columns(2)
    
    for i, (key, label) in enumerate(CATEGORIES):
        col = col1 if i % 2 == 0 else col2
        
        with col:
            interest = st.slider(
                f"{label}",
                min_value=1, max_value=5, value=3, step=1,
                help=f"1 = {SCORE_LABELS[1]}, 5 = {SCORE_LABELS[5]}",
                key=f"category_{key}"
            )
            category_interest[key] = interest

    # Transporte y ubicación
    st.markdown("Transporte y Ubicación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_transport = st.multiselect(
            "Medios de transporte disponibles:",
            TRANSPORT_MODES,
            default=["Caminar"],
            help="Selecciona todos los medios que puedes utilizar"
        )
    
    with col2:
        user_address = st.text_input(
            "Punto de partida:",
            placeholder="Ej: Plaza Mayor, Estación Central...",
            help="Introduce tu dirección o punto de referencia"
        )

    # Notas adicionales
    st.markdown("### 📝 Preferencias Adicionales")
    user_notes = st.text_area(
        "Cuéntanos más sobre tus intereses (opcional):",
        placeholder="Ej: Me interesa la arquitectura moderna, comida local, evitar multitudes...",
        help="Cualquier información adicional que nos ayude a personalizar tu experiencia"
    )

    # Botón principal de búsqueda
    st.markdown("---")
    
    if st.button("🔍 Buscar Rutas Turísticas Optimizadas", type="primary", use_container_width=True):
        
        # Validaciones
        if not selected_transport:
            st.error("❌ Por favor selecciona al menos un medio de transporte.")
            return
        
        if not user_address:
            st.warning("⚠️ Se recomienda especificar un punto de partida para mejores resultados.")
        
        # Geocodificación
        if user_address:
            with st.spinner("📍 Localizando tu punto de partida..."):
                lat, lon = geocode_address(user_address, city)
                
                if lat is not None and lon is not None:
                    st.success(f"✅ Ubicación encontrada: {lat:.5f}, {lon:.5f}")
                else:
                    st.error("❌ No se pudo localizar la dirección. Usando centro de la ciudad.")
                    
        else:
            # Usar coordenadas del centro de la ciudad
            city_coords = {
                'Madrid': (40.4168, -3.7038),
                'Barcelona': (41.3851, 2.1734),
                'Valencia': (39.4699, -0.3763),
                'Sevilla': (37.3891, -5.9845),
                'Bilbao': (43.2627, -2.9253)
            }
            lat, lon = city_coords.get(city, (40.4168, -3.7038))
            st.info(f"📍 Usando centro de {city} como punto de partida")

        # Procesamiento con RAG
        st.markdown("### 🤖 Procesando Recomendaciones con IA")
        
        try:
            
            # PASO 2: Preparar preferencias del usuario
            user_preferences = {
                'city': city,
                'available_hours': available_hours,
                'category_interest': category_interest,
                'transport_modes': selected_transport,
                'max_distance': max_distance,
                'user_notes': user_notes.strip() if user_notes.strip() else ""
            }
            
            # PASO 3: Procesar con RAG usando datos actualizados
            st.markdown("#### 🧠 Análisis Inteligente de Preferencias")
            
            with st.spinner("🔍 Analizando tus preferencias y buscando lugares relevantes..."):
                try:
                    rag_planner = RAGPlanner(chroma_db_path="db/")
                    transport_mode = selected_transport[0] if selected_transport else "A pie"
                    
                    rag_data = rag_planner.process_user_request(user_preferences, lat, lon, transport_mode)
                    
                except Exception as e:
                    st.error(f"❌ Error en el procesamiento RAG: {str(e)}")
                    return
            
            # Mostrar resultados del RAG
            if rag_data['filtered_places']:
                st.success(f"✅ ¡Encontrados {len(rag_data['filtered_places'])} lugares que coinciden con tus preferencias!")
                
                # Información del RAG
                with st.expander("📊 Información del Análisis IA"):
                    st.write(f"**Fuente de datos:** {rag_data.get('data_source', 'ChromaDB')}")
                    st.write(f"**Lugares analizados:** {len(rag_data['filtered_places'])}")
                    st.write(f"**Matriz de tiempos:** {rag_data['time_matrix'].shape}")
                    
                    if rag_data.get('llm_response'):
                        st.write("**Análisis del LLM:**")
                        st.text_area("", rag_data['llm_response'][:500] + "...", height=100)
                
                # Mostrar lugares recomendados
                st.markdown("### 📍 Lugares Recomendados")
                
                for i, place in enumerate(rag_data['filtered_places']):
                    time_estimate = rag_data['llm_time_estimates'][i] if i < len(rag_data['llm_time_estimates']) else 2.0
                    similarity = rag_data['similarity_scores'][i] if i < len(rag_data['similarity_scores']) else 0.0
                    
                    with st.expander(f"📍 {place['name']} ({time_estimate}h recomendadas - Afinidad: {similarity:.2f})"):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Categoría:** {place.get('touristClassification', 'No especificada')}")
                            st.write(f"**Descripción:** {place.get('description', 'No disponible')}")
                            st.write(f"**Atractivo:** {place.get('visitorAppeal', 'No disponible')}")
                        
                        with col2:
                            st.metric("Tiempo recomendado", f"{time_estimate}h")
                            st.metric("Afinidad", f"{similarity:.2f}")
                
                # OPTIMIZACIÓN DE RUTAS CON METAHEURÍSTICA
                display_route_optimization_results(rag_data, user_preferences, lat, lon, transport_mode)
                
            else:
                st.warning("⚠️ No se encontraron lugares que coincidan con tus criterios.")
                st.info("💡 Intenta:")
                st.write("- Aumentar la distancia máxima")
                st.write("- Ajustar tus preferencias de categorías")
                st.write("- Verificar que hay datos disponibles para la ciudad seleccionada")
                
        except Exception as e:
            st.error(f"❌ Error procesando la solicitud: {str(e)}")
            st.info("Por favor, verifica tu configuración e intenta nuevamente.")

if __name__ == "__main__":
    app()