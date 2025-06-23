# Wrapper para compatibilidad hacia atrás con la nueva estructura modular
from .rag_planner import RAGPlanner

# Si hay funciones utilitarias que deban exponerse, se pueden importar aquí
default_app_class = RAGPlanner
