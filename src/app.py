import streamlit as st
from geopy.geocoders import Nominatim

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

CITIES = ["Paris", "Roma"]

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

    # Selección de ciudad
    city = st.selectbox("¿A qué ciudad viajas?", CITIES)

    # Número de días
    days = st.number_input("¿Cuántos días te vas a quedar?", min_value=1, max_value=30, value=3, step=1)

    # Preferencias de categorías con sliders de interés
    st.markdown("### ¿Qué tipo de actividades prefieres? (Indica tu nivel de interés)")
    category_interest = {}
    for key, label in CATEGORIES:
        interest = st.slider(
            f"{label}",
            min_value=0, max_value=10, value=5, step=1,
            help="0 = Nada interesado, 10 = Máximo interés"
        )
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
        st.write("**Días de estancia:**", days)
        st.write("**Preferencias por categoría:**", {label: category_interest[key] for key, label in CATEGORIES})
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

        # Aquí puedes pasar category_interest (como dict de pesos), lat, lon, max_distance, etc. a tu lógica RAG