"""
======================================================
 Sistema Multiagente - Generación de Blog de Tecnología
 Arquitectura Horizontal Colaborativa
======================================================
Agentes Inteligentes
Otoño 2025

Bernardo Bojalil Lorenzini - 195908

DESCRIPCIÓN:
Este sistema implementa una arquitectura multiagente horizontal donde
tres agentes especializados colaboran para generar artículos de blog
de tecnología de manera autónoma:
  1. Agente Investigador: Busca y recopila información sobre el tema
  2. Agente Redactor: Crea el borrador del artículo
  3. Agente Editor: Revisa y produce la versión final

CARACTERÍSTICAS:
- Sistema de mensajería descentralizado (peer-to-peer)
- Cada agente tiene especialización y temperature específica
- Usa Google Gemini vía LangChain para generación de contenido
- Historial de comunicación entre agentes
- Guardado automático de artículos y logs
"""

import os
import json
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

# Importaciones de LangChain 1.x (compatible con langchain-core)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==================== CONFIGURACIÓN INICIAL ====================

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Obtener API key de Google Gemini
# Acepta tanto GOOGLE_API_KEY como GEMINI_API_KEY para flexibilidad
google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not google_key:
    raise ValueError("Error: Configura GOOGLE_API_KEY o GEMINI_API_KEY en tu archivo .env")


# ==================== SISTEMA DE MENSAJERÍA ====================

class MensajeAgente:
    """
    Representa un mensaje entre agentes del sistema.
    
    Encapsula toda la información necesaria para la comunicación:
    - Remitente y destinatario (nombres de agentes)
    - Contenido del mensaje (datos, borrador, etc.)
    - Tipo de mensaje (investigacion, borrador, datos, etc.)
    - Timestamp para trazabilidad
    """
    def __init__(self, remitente: str, destinatario: str, contenido: str, tipo: str = "datos"):
        self.remitente = remitente
        self.destinatario = destinatario
        self.contenido = contenido
        self.tipo = tipo
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        """Convierte el mensaje a diccionario para serialización JSON"""
        return {
            "remitente": self.remitente,
            "destinatario": self.destinatario,
            "tipo": self.tipo,
            "timestamp": self.timestamp
        }


class BuzonMensajes:
    """
    Sistema de mensajería descentralizado para comunicación entre agentes.
    
    Implementa un patrón de buzón (mailbox) donde:
    - Los agentes envían mensajes sin bloqueo
    - Los mensajes se almacenan hasta que el destinatario los recupera
    - Se mantiene un historial completo para auditoría
    
    ARQUITECTURA: Peer-to-peer horizontal (no hay controlador central)
    """
    def __init__(self):
        self.mensajes: List[MensajeAgente] = []  # Cola de mensajes pendientes
        self.historial: List[Dict] = []          # Historial completo
    
    def enviar_mensaje(self, remitente: str, destinatario: str, contenido: str, tipo: str = "datos"):
        """
        Envía un mensaje de un agente a otro.
        
        Args:
            remitente: Nombre del agente que envía
            destinatario: Nombre del agente receptor
            contenido: Datos o información a transmitir
            tipo: Categoría del mensaje (investigacion, borrador, datos)
        
        Returns:
            MensajeAgente: El mensaje creado
        """
        mensaje = MensajeAgente(remitente, destinatario, contenido, tipo)
        self.mensajes.append(mensaje)
        self.historial.append(mensaje.to_dict())
        print(f"\n📨 {remitente} → {destinatario}")
        return mensaje
    
    def obtener_mensajes(self, destinatario: str) -> List[MensajeAgente]:
        """
        Obtiene y elimina mensajes pendientes para un destinatario.
        
        Implementa un patrón "consumir y borrar" para evitar procesamiento duplicado.
        
        Args:
            destinatario: Nombre del agente que recupera mensajes
        
        Returns:
            Lista de mensajes pendientes para ese agente
        """
        mensajes_pendientes = [m for m in self.mensajes if m.destinatario == destinatario]
        self.mensajes = [m for m in self.mensajes if m.destinatario != destinatario]
        return mensajes_pendientes
    
    def guardar_historial(self, ruta: str = "historial_mensajes.json"):
        """
        Guarda el historial completo de mensajes en JSON.
        
        Útil para:
        - Debugging del flujo de comunicación
        - Auditoría de la colaboración entre agentes
        - Análisis de rendimiento del sistema
        """
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(self.historial, f, ensure_ascii=False, indent=2)


# ==================== AGENTE BASE ====================

class AgenteBase:
    """
    Clase base para todos los agentes del sistema.
    
    Define la funcionalidad común:
    - Conexión al LLM (Google Gemini)
    - Métodos de comunicación (enviar/recibir mensajes)
    - Logging para trazabilidad
    
    PATRÓN: Template Method - Las subclases implementan comportamientos específicos
    """
    def __init__(self, nombre: str, rol: str, buzon: BuzonMensajes, temperature: float = 0.3):
        """
        Inicializa un agente con sus propiedades y modelo LLM.
        
        Args:
            nombre: Identificador único del agente
            rol: Descripción de la especialización
            buzon: Referencia al sistema de mensajería compartido
            temperature: Creatividad del LLM (0.0=determinista, 1.0=creativo)
        """
        self.nombre = nombre
        self.rol = rol
        self.buzon = buzon
        # Cada agente tiene su propia instancia de LLM con temperatura específica
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # Modelo rápido y eficiente
            temperature=temperature,     # Controla aleatoriedad/creatividad
            google_api_key=google_key,
        )
    
    def log(self, mensaje: str):
        """
        Registra un mensaje del agente con formato identificable.
        
        Útil para debugging y seguimiento del flujo de trabajo.
        """
        print(f"[{self.nombre}] {mensaje}")
    
    def enviar_mensaje(self, destinatario: str, contenido: str, tipo: str = "datos"):
        """
        Envía un mensaje a otro agente a través del buzón compartido.
        
        Args:
            destinatario: Nombre del agente receptor
            contenido: Información a transmitir
            tipo: Categoría del mensaje
        
        Returns:
            MensajeAgente: El mensaje enviado
        """
        return self.buzon.enviar_mensaje(self.nombre, destinatario, contenido, tipo)
    
    def recibir_mensajes(self) -> List[MensajeAgente]:
        """
        Recupera mensajes pendientes desde el buzón.
        
        Returns:
            Lista de mensajes dirigidos a este agente
        """
        return self.buzon.obtener_mensajes(self.nombre)


# ==================== AGENTE INVESTIGADOR ====================

class ResearchAgent(AgenteBase):
    """
    Agente especializado en búsqueda e investigación de información.
    
    RESPONSABILIDADES:
    - Recibir solicitudes de investigación sobre temas de tecnología
    - Buscar información objetiva, actual y verificable
    - Estructurar datos en formato útil para redacción
    - Enviar resultados al Agente Redactor
    
    CARACTERÍSTICAS:
    - Temperature: 0.3 (bajo para mantener objetividad)
    - Enfoque: Hechos, datos, estadísticas, tendencias
    """
    
    def __init__(self, buzon: BuzonMensajes):
        super().__init__("Agente Investigador", "Experto en búsqueda de información", buzon, temperature=0.3)
        # Prompt template diseñado para investigación estructurada
        self.prompt_template = PromptTemplate(
            input_variables=["tema"],
            template="""
Eres un investigador de tecnología. Busca información objetiva y actual sobre el tema solicitado.

TEMA: {tema}

Proporciona:
1. Definición y contexto del tema (2 párrafos)
2. Datos y estadísticas relevantes (3-5 puntos)
3. Aplicaciones prácticas actuales
4. Tendencias y desarrollos recientes
5. Empresas o proyectos destacados

Sé objetivo, conciso y preciso. Enfócate en hechos verificables.
"""
        )
    
    def investigar(self, tema: str) -> str:
        """
        Acción principal: investigar(tema)
        
        Ejecuta el proceso de investigación:
        1. Genera prompt específico para el tema
        2. Invoca el LLM (Gemini) para obtener información
        3. Envía resultados al siguiente agente en el flujo
        
        Args:
            tema: Tema de tecnología a investigar
        
        Returns:
            str: Datos e información recopilada
        """
        self.log(f"Investigando: {tema}")
        
        # Construir cadena de procesamiento LangChain
        # prompt_template | llm | parser = pipeline completo
        chain = self.prompt_template | self.llm | StrOutputParser()
        datos_encontrados = chain.invoke({"tema": tema})
        
        # Enviar mensaje al Agente Redactor con los datos
        self.enviar_mensaje("Agente Redactor", datos_encontrados, tipo="investigacion")
        
        return datos_encontrados


# ==================== AGENTE REDACTOR ====================

class WriterAgent(AgenteBase):
    """
    Agente especializado en redacción técnica y creación de contenido.
    
    RESPONSABILIDADES:
    - Recibir datos de investigación
    - Transformar datos en artículo estructurado
    - Aplicar formato de blog profesional
    - Mantener tono técnico pero accesible
    - Enviar borrador al Agente Editor
    
    CARACTERÍSTICAS:
    - Temperature: 0.4 (moderado para balance entre claridad y creatividad)
    - Enfoque: Estructura, claridad, narrativa coherente
    """
    
    def __init__(self, buzon: BuzonMensajes):
        super().__init__("Agente Redactor", "Experto en redacción técnica", buzon, temperature=0.4)
        # Prompt template diseñado para redacción estructurada
        self.prompt_template = PromptTemplate(
            input_variables=["datos_investigacion"],
            template="""
Eres un redactor técnico. Escribe un artículo de blog claro y objetivo basado en los datos proporcionados.

DATOS DE INVESTIGACIÓN:
{datos_investigacion}

ESTRUCTURA:
1. Título descriptivo
2. Introducción (2 párrafos): qué es y por qué es importante
3. Sección 1: Fundamentos y contexto
4. Sección 2: Aplicaciones y casos de uso
5. Sección 3: Estado actual y tendencias
6. Conclusión: resumen y perspectiva

ESTILO:
- Tono profesional e informativo
- Párrafos cortos y directos
- Enfoque en hechos y datos
- Evita lenguaje promocional o especulativo
- 700-900 palabras

Genera el borrador completo del artículo.
"""
        )
    
    def redactar(self, datos: str) -> str:
        """
        Acción principal: redactar(datos)
        
        Transforma datos de investigación en artículo estructurado:
        1. Analiza los datos recibidos
        2. Genera estructura de artículo
        3. Crea contenido siguiendo guía editorial
        4. Envía borrador al Editor
        
        Args:
            datos: Información recopilada por el Agente Investigador
        
        Returns:
            str: Borrador del artículo
        """
        self.log("Redactando artículo...")
        
        # Cadena LangChain: Prompt → LLM → Parseo de salida
        chain = self.prompt_template | self.llm | StrOutputParser()
        borrador_articulo = chain.invoke({"datos_investigacion": datos})
        
        # Enviar borrador al Agente Editor para revisión
        self.enviar_mensaje("Agente Editor", borrador_articulo, tipo="borrador")
        
        return borrador_articulo
    
    def procesar_mensajes(self):
        """
        Procesa mensajes entrantes de tipo 'investigacion'.
        
        Implementa el patrón de procesamiento asíncrono:
        - Revisa buzón de mensajes
        - Filtra por tipo 'investigacion'
        - Ejecuta acción de redacción
        
        Returns:
            str o None: Artículo redactado, o None si no hay mensajes
        """
        mensajes = self.recibir_mensajes()
        for mensaje in mensajes:
            if mensaje.tipo == "investigacion":
                return self.redactar(mensaje.contenido)
        return None


# ==================== AGENTE EDITOR ====================

class EditorAgent(AgenteBase):
    """
    Agente especializado en revisión y edición final de contenido.
    
    RESPONSABILIDADES:
    - Recibir borradores del Agente Redactor
    - Corregir errores ortográficos y gramaticales
    - Mejorar claridad y coherencia
    - Verificar estructura y flujo
    - Producir versión final lista para publicación
    
    CARACTERÍSTICAS:
    - Temperature: 0.2 (bajo para mantener precisión y consistencia)
    - Enfoque: Calidad, corrección, profesionalismo
    """
    
    def __init__(self, buzon: BuzonMensajes):
        super().__init__("Agente Editor", "Experto en revisión y edición", buzon, temperature=0.2)
        # Prompt template enfocado en revisión y mejora
        self.prompt_template = PromptTemplate(
            input_variables=["borrador"],
            template="""
Eres un editor profesional. Revisa y mejora el borrador para producir la versión final.

BORRADOR:
{borrador}

TAREAS:
1. Corrige errores ortográficos y gramaticales
2. Mejora la claridad y coherencia
3. Verifica la estructura y flujo
4. Optimiza títulos y subtítulos
5. Asegura tono profesional consistente

IMPORTANTE: Genera ÚNICAMENTE el artículo final corregido y mejorado, sin comentarios adicionales.
No incluyas notas, resúmenes de cambios ni sugerencias. Solo el artículo listo para publicar.
"""
        )
    
    def revisar(self, borrador: str) -> str:
        """
        Acción principal: revisar(borrador)
        
        Ejecuta proceso de revisión y edición:
        1. Analiza el borrador recibido
        2. Aplica correcciones ortográficas y gramaticales
        3. Mejora estructura y coherencia
        4. Genera versión final pulida
        
        Args:
            borrador: Artículo preliminar del Agente Redactor
        
        Returns:
            str: Artículo final listo para publicación
        """
        self.log("Revisando y generando versión final...")
        
        # Cadena LangChain para edición
        chain = self.prompt_template | self.llm | StrOutputParser()
        articulo_final = chain.invoke({"borrador": borrador})
        
        print("\n✅ ¡TAREA COMPLETADA! Artículo Finalizado.")
        
        return articulo_final
    
    def procesar_mensajes(self):
        """
        Procesa mensajes entrantes de tipo 'borrador'.
        
        Implementa el patrón de procesamiento asíncrono:
        - Revisa buzón de mensajes
        - Filtra por tipo 'borrador'
        - Ejecuta acción de revisión y edición
        
        Returns:
            str o None: Artículo final, o None si no hay mensajes
        """
        mensajes = self.recibir_mensajes()
        for mensaje in mensajes:
            if mensaje.tipo == "borrador":
                return self.revisar(mensaje.contenido)
        return None


# ==================== COORDINADOR DEL SISTEMA ====================

class CoordinadorMultiagente:
    """
    Coordina la colaboración entre los agentes del sistema.
    
    RESPONSABILIDADES:
    - Inicializar el sistema de mensajería y agentes
    - Orquestar el flujo de trabajo completo
    - Gestionar las 3 fases del proceso (Investigación → Redacción → Edición)
    - Guardar artículos finales y logs
    
    ARQUITECTURA:
    - No es un controlador centralizado autoritario
    - Actúa como facilitador que inicia el proceso
    - Los agentes colaboran de manera autónoma vía mensajería
    
    FLUJO DE TRABAJO:
    1. Usuario solicita tema
    2. Coordinador inicia fase de investigación
    3. Agente Investigador → envía datos → Agente Redactor
    4. Agente Redactor → envía borrador → Agente Editor
    5. Agente Editor → produce versión final
    6. Coordinador guarda artículo y logs
    """
    
    def __init__(self):
        """
        Inicializa el coordinador y todos los agentes del sistema.
        
        Crea:
        - Sistema de mensajería compartido (BuzonMensajes)
        - Los tres agentes especializados con acceso al buzón
        """
        self.buzon = BuzonMensajes()
        self.agente_investigador = ResearchAgent(self.buzon)
        self.agente_redactor = WriterAgent(self.buzon)
        self.agente_editor = EditorAgent(self.buzon)
    
    def generar_articulo(self, tema: str) -> str:
        """
        Flujo completo de generación de artículo.
        
        Ejecuta las tres fases de manera secuencial:
        - FASE 1: Investigación (buscar información)
        - FASE 2: Redacción (crear borrador)
        - FASE 3: Edición (versión final)
        
        Args:
            tema: Tema de tecnología para el artículo
        
        Returns:
            str: Artículo final completo
        
        Raises:
            Exception: Si no se puede completar el artículo
        """
        print("\n" + "="*60)
        print(f"SISTEMA MULTIAGENTE - BLOG DE TECNOLOGÍA")
        print("="*60)
        print(f"Tema: {tema}\n")
        
        # ===== FASE 1: INVESTIGACIÓN =====
        # El Agente Investigador busca información sobre el tema
        # y envía los datos al Agente Redactor automáticamente
        print("FASE 1: INVESTIGACIÓN")
        print("-"*60)
        datos = self.agente_investigador.investigar(tema)
        
        # ===== FASE 2: REDACCIÓN =====
        # El Agente Redactor procesa los mensajes recibidos,
        # crea el borrador y lo envía al Agente Editor
        print("\nFASE 2: REDACCIÓN")
        print("-"*60)
        self.agente_redactor.procesar_mensajes()
        
        # ===== FASE 3: EDICIÓN =====
        # El Agente Editor revisa el borrador y produce la versión final
        print("\nFASE 3: REVISIÓN Y EDICIÓN")
        print("-"*60)
        articulo_final = self.agente_editor.procesar_mensajes()
        
        # Validar que el proceso se completó exitosamente
        if articulo_final:
            # Guardar artículo final en archivo de texto
            self._guardar_articulo(tema, articulo_final)
            # Guardar historial de mensajes para auditoría
            self.buzon.guardar_historial()
            return articulo_final
        else:
            raise Exception("Error: No se pudo completar el artículo")
    
    def _guardar_articulo(self, tema: str, articulo: str):
        """
        Guarda el artículo final en un archivo de texto.
        
        Formato del archivo:
        - Nombre: articulo_YYYYMMDD_HHMMSS.txt (timestamp único)
        - Contenido: Tema, fecha y artículo completo
        
        Args:
            tema: Tema del artículo
            articulo: Contenido completo del artículo final
        """
        # Generar nombre de archivo único con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"articulo_{timestamp}.txt"
        
        # Escribir artículo con metadatos
        with open(nombre_archivo, "w", encoding="utf-8") as f:
            f.write(f"TEMA: {tema}\n")
            f.write(f"FECHA: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write(articulo)
        
        print(f"\n💾 Artículo guardado: {nombre_archivo}")


# ==================== FUNCIÓN PRINCIPAL ====================

def main():
    """
    Función principal del sistema - Punto de entrada.
    
    Maneja dos modos de operación:
    1. MODO AUTOMÁTICO (AUTO_RUN=1 en .env):
       - Genera artículo sin interacción del usuario
       - Útil para testing y ejecución en CI/CD
    
    2. MODO INTERACTIVO (por defecto):
       - Solicita tema al usuario
       - Permite generar múltiples artículos en una sesión
       - Provee sugerencias de temas
    
    Variables de entorno opcionales:
    - AUTO_RUN: "1" para modo automático
    - TEMA_BLOG: Tema predefinido en modo automático
    """
    
    # Inicializar coordinador (crea agentes y sistema de mensajería)
    coordinador = CoordinadorMultiagente()
    
    # ===== MODO AUTOMÁTICO =====
    # Útil para pruebas automatizadas o despliegue
    if os.getenv("AUTO_RUN") == "1":
        tema = os.getenv("TEMA_BLOG", "Inteligencia Artificial en la Medicina")
        print(f"\n[MODO AUTOMÁTICO] Tema: {tema}\n")
        coordinador.generar_articulo(tema)
        return
    
    # ===== MODO INTERACTIVO =====
    # Interfaz amigable para el usuario
    print("\n¿Qué tema de tecnología deseas explorar?")
    print("\nEjemplos:")
    print("  - Inteligencia Artificial en la Medicina")
    print("  - Blockchain y sus Aplicaciones")
    print("  - Computación Cuántica")
    print("  - Internet de las Cosas (IoT)")
    print("  - Ciberseguridad Moderna\n")
    
    # Solicitar tema al usuario (con valor por defecto)
    tema = input("Tema: ").strip()
    if not tema:
        tema = "Inteligencia Artificial en la Medicina"
    
    # Generar el primer artículo
    coordinador.generar_articulo(tema)
    
    # ===== BUCLE INTERACTIVO =====
    # Permitir generar múltiples artículos en la misma sesión
    while True:
        respuesta = input("\n¿Generar otro artículo? (s/n): ").strip().lower()
        if respuesta not in ['s', 'si', 'sí']:
            print("\n¡Hasta luego!")
            break
        
        tema = input("\nNuevo tema: ").strip()
        if tema:
            coordinador.generar_articulo(tema)


# ==================== PUNTO DE ENTRADA ====================
if __name__ == "__main__":
    """
    Ejecuta el sistema cuando el script se ejecuta directamente.
    
    Para ejecutar:
    - Windows: python main.py
    - Linux/Mac: python3 main.py
    
    Requisitos previos:
    1. Archivo .env con GOOGLE_API_KEY o GEMINI_API_KEY
    2. Dependencias instaladas (ver requirements.txt)
    """
    main()