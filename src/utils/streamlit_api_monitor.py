"""
Componente de Streamlit para monitorear las llamadas a la API de Gemini en tiempo real
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json

from .gemini_api_counter import GeminiAPICounter, api_counter

class StreamlitAPIMonitor:
    """Monitor de API integrado con Streamlit"""
    
    def __init__(self):
        self.counter = api_counter
    
    def display_api_stats_sidebar(self):
        """Muestra estadísticas básicas en la barra lateral"""
        stats = self.counter.get_stats()
        
        with st.sidebar:
            st.markdown("### 📊 API Gemini Stats")
            
            # Métricas principales
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Calls", stats['call_counts']['total_calls'])
                st.metric("Success Rate", f"{stats['call_counts']['success_rate']:.1f}%")
            
            with col2:
                st.metric("Failed Calls", stats['call_counts']['failed_calls'])
                st.metric("Avg Duration", f"{stats['timing']['average_call_duration']:.2f}s")
            
            # Uso de datos
            st.markdown("**Data Usage:**")
            st.write(f"📤 Sent: {stats['data_usage']['total_prompt_characters']:,} chars")
            st.write(f"📥 Received: {stats['data_usage']['total_response_characters']:,} chars")
            
            # Botón para ver detalles
            if st.button("🔍 View Detailed Stats"):
                st.session_state.show_api_details = True
    
    def display_detailed_stats(self):
        """Muestra estadísticas detalladas en la página principal"""
        if not st.session_state.get('show_api_details', False):
            return
        
        st.markdown("## 📊 Detailed Gemini API Statistics")
        
        stats = self.counter.get_stats()
        calls = self.counter.get_detailed_calls()
        
        # Pestañas para organizar la información
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📞 Call Details", "⏱️ Timeline", "💾 Export"])
        
        with tab1:
            self._display_overview_tab(stats)
        
        with tab2:
            self._display_calls_tab(calls)
        
        with tab3:
            self._display_timeline_tab(calls)
        
        with tab4:
            self._display_export_tab(stats, calls)
        
        # Botón para cerrar
        if st.button("❌ Close Detailed Stats"):
            st.session_state.show_api_details = False
    
    def _display_overview_tab(self, stats: Dict[str, Any]):
        """Pestaña de resumen general"""
        
        # Métricas principales en columnas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total API Calls", 
                stats['call_counts']['total_calls'],
                help="Número total de llamadas realizadas a la API de Gemini"
            )
        
        with col2:
            st.metric(
                "Success Rate", 
                f"{stats['call_counts']['success_rate']:.1f}%",
                delta=f"+{stats['call_counts']['successful_calls']} successful",
                help="Porcentaje de llamadas exitosas"
            )
        
        with col3:
            st.metric(
                "Total Characters", 
                f"{stats['data_usage']['total_characters']:,}",
                help="Total de caracteres enviados y recibidos"
            )
        
        with col4:
            st.metric(
                "API Time", 
                stats['timing']['total_api_time_formatted'],
                help="Tiempo total gastado en llamadas a la API"
            )
        
        # Gráficos de resumen
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de éxito/fallo
            success_data = {
                'Status': ['Successful', 'Failed'],
                'Count': [stats['call_counts']['successful_calls'], stats['call_counts']['failed_calls']],
                'Color': ['#00CC96', '#FF6B6B']
            }
            
            fig_success = px.pie(
                success_data, 
                values='Count', 
                names='Status',
                title="API Call Success Rate",
                color_discrete_sequence=['#00CC96', '#FF6B6B']
            )
            st.plotly_chart(fig_success, use_container_width=True)
        
        with col2:
            # Desglose por módulo
            caller_breakdown = stats['caller_breakdown']
            if caller_breakdown:
                modules = []
                counts = []
                for module, functions in caller_breakdown.items():
                    total_calls = sum(functions.values())
                    modules.append(module.split('.')[-1])  # Solo el nombre del módulo
                    counts.append(total_calls)
                
                fig_modules = px.bar(
                    x=counts,
                    y=modules,
                    orientation='h',
                    title="API Calls by Module",
                    labels={'x': 'Number of Calls', 'y': 'Module'}
                )
                fig_modules.update_layout(height=400)
                st.plotly_chart(fig_modules, use_container_width=True)
        
        # Información de sesión
        st.markdown("### 📅 Session Information")
        session_col1, session_col2 = st.columns(2)
        
        with session_col1:
            st.write(f"**Session Start:** {stats['session_info']['start_time']}")
            st.write(f"**Session Duration:** {stats['session_info']['session_duration_formatted']}")
        
        with session_col2:
            st.write(f"**Calls per Minute:** {stats['timing']['calls_per_minute']:.1f}")
            st.write(f"**Average Call Duration:** {stats['timing']['average_call_duration']:.2f}s")
    
    def _display_calls_tab(self, calls: List[Dict[str, Any]]):
        """Pestaña con detalles de las llamadas"""
        
        if not calls:
            st.info("No API calls recorded yet.")
            return
        
        st.markdown("### 📞 Individual API Calls")
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_only_failed = st.checkbox("Show only failed calls")
        
        with col2:
            min_duration = st.number_input("Min duration (s)", min_value=0.0, value=0.0, step=0.1)
        
        with col3:
            max_calls = st.number_input("Max calls to show", min_value=1, value=50, step=10)
        
        # Filtrar llamadas
        filtered_calls = calls
        if show_only_failed:
            filtered_calls = [call for call in filtered_calls if not call['success']]
        
        filtered_calls = [call for call in filtered_calls if call['duration_seconds'] >= min_duration]
        filtered_calls = filtered_calls[-max_calls:]  # Mostrar las más recientes
        
        # Crear DataFrame para mostrar
        if filtered_calls:
            df_calls = pd.DataFrame(filtered_calls)
            df_calls['timestamp'] = pd.to_datetime(df_calls['timestamp'])
            df_calls = df_calls.sort_values('timestamp', ascending=False)
            
            # Mostrar tabla
            st.dataframe(
                df_calls[['timestamp', 'success', 'duration_seconds', 'prompt_length', 
                         'response_length', 'caller_module', 'caller_function', 'error_message']],
                use_container_width=True,
                column_config={
                    'timestamp': st.column_config.DatetimeColumn('Time'),
                    'success': st.column_config.CheckboxColumn('Success'),
                    'duration_seconds': st.column_config.NumberColumn('Duration (s)', format="%.2f"),
                    'prompt_length': st.column_config.NumberColumn('Prompt Length'),
                    'response_length': st.column_config.NumberColumn('Response Length'),
                    'caller_module': st.column_config.TextColumn('Module'),
                    'caller_function': st.column_config.TextColumn('Function'),
                    'error_message': st.column_config.TextColumn('Error')
                }
            )
            
            # Mostrar detalles de una llamada específica
            if st.checkbox("Show call details"):
                selected_idx = st.selectbox(
                    "Select call to inspect:",
                    range(len(filtered_calls)),
                    format_func=lambda x: f"Call {x+1}: {filtered_calls[x]['timestamp']} ({'✅' if filtered_calls[x]['success'] else '❌'})"
                )
                
                if selected_idx is not None:
                    call = filtered_calls[selected_idx]
                    
                    st.markdown("#### Call Details")
                    st.json(call)
        else:
            st.info("No calls match the current filters.")
    
    def _display_timeline_tab(self, calls: List[Dict[str, Any]]):
        """Pestaña con timeline de llamadas"""
        
        if not calls:
            st.info("No API calls recorded yet.")
            return
        
        st.markdown("### ⏱️ API Calls Timeline")
        
        # Crear DataFrame
        df = pd.DataFrame(calls)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Gráfico de llamadas por tiempo
        fig_timeline = px.scatter(
            df,
            x='timestamp',
            y='duration_seconds',
            color='success',
            size='prompt_length',
            hover_data=['caller_module', 'caller_function', 'response_length'],
            title="API Calls Over Time",
            labels={
                'timestamp': 'Time',
                'duration_seconds': 'Duration (seconds)',
                'success': 'Success'
            },
            color_discrete_map={True: '#00CC96', False: '#FF6B6B'}
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Histograma de duraciones
        fig_hist = px.histogram(
            df,
            x='duration_seconds',
            nbins=20,
            title="Distribution of Call Durations",
            labels={'duration_seconds': 'Duration (seconds)', 'count': 'Number of Calls'}
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Llamadas por minuto
        df['minute'] = df['timestamp'].dt.floor('min')  # Redondear a minuto
        calls_per_minute = df.groupby('minute').size().reset_index(name='calls')
        
        if len(calls_per_minute) > 1:
            fig_rate = px.line(
                calls_per_minute,
                x='minute',
                y='calls',
                title="API Calls Rate (per minute)",
                labels={'minute': 'Time', 'calls': 'Calls per Minute'}
            )
            
            st.plotly_chart(fig_rate, use_container_width=True)
    
    def _display_export_tab(self, stats: Dict[str, Any], calls: List[Dict[str, Any]]):
        """Pestaña para exportar datos"""
        
        st.markdown("### 💾 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Options")
            
            export_stats = st.checkbox("Include statistics summary", value=True)
            export_calls = st.checkbox("Include detailed call data", value=True)
            
            if st.button("📥 Download JSON Report"):
                data = {}
                if export_stats:
                    data['statistics'] = stats
                if export_calls:
                    data['calls'] = calls
                
                # Crear archivo JSON
                json_str = json.dumps(data, indent=2, ensure_ascii=False, default=str)
                
                st.download_button(
                    label="📄 Download JSON File",
                    data=json_str,
                    file_name=f"gemini_api_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.markdown("#### Quick Stats")
            st.code(f"""
Total API Calls: {stats['call_counts']['total_calls']}
Success Rate: {stats['call_counts']['success_rate']:.1f}%
Total Characters: {stats['data_usage']['total_characters']:,}
Total API Time: {stats['timing']['total_api_time_formatted']}
Session Duration: {stats['session_info']['session_duration_formatted']}
            """)
        
        # Botón para resetear contador
        st.markdown("#### Reset Counter")
        st.warning("⚠️ This will clear all recorded API call data!")
        
        if st.button("🗑️ Reset API Counter", type="secondary"):
            self.counter.reset()
            st.success("API counter has been reset!")
    
    def display_live_counter(self):
        """Muestra un contador en vivo en la parte superior de la página"""
        stats = self.counter.get_stats()
        
        # Crear métricas en línea
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🔥 API Calls", stats['call_counts']['total_calls'])
        
        with col2:
            st.metric("✅ Success", f"{stats['call_counts']['success_rate']:.0f}%")
        
        with col3:
            st.metric("📊 Characters", f"{stats['data_usage']['total_characters']:,}")
        
        with col4:
            st.metric("⏱️ API Time", stats['timing']['total_api_time_formatted'])
        
        with col5:
            if st.button("📈 Details"):
                st.session_state.show_api_details = True

# Instancia global del monitor
streamlit_monitor = StreamlitAPIMonitor()