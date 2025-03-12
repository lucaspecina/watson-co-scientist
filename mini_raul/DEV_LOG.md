# Introduccion general:

En este documento, tienes que ir haciendo un seguimiento del plan de desarrollo para cada iteracion y lo que se fue haciendo, resolviendo, problemas, comentarios, contexto, etc. Este documento de DEV deberia ser suficiente para volver en cualquier momento, sin memoria de lo hecho y entender todo lo que se fue haciendo, los pasos, iteraciones, pruebas, etc. SIEMPRE ACTUALIZALO!

# Desarrollo de mini-RAUL

## Iteración 1: Esqueleto del sistema multi-agente

### Fecha: 2023-12-03

### Objetivos:
- Crear la estructura básica del proyecto
- Implementar el framework multi-agente básico
- Implementar la gestión de sesiones
- Implementar los diferentes tipos de agentes
- Crear una interfaz de línea de comandos básica

### Implementación:

1. **Estructura del proyecto**:
   - Creada estructura modular con separación clara de responsabilidades
   - Organización en módulos: core, agents, models, utils, data, cli

2. **Core**:
   - Implementada clase `Session` para gestionar el estado de la investigación
   - Implementada clase `Coordinator` para orquestar los diferentes agentes
   - Definidos estados de sesión (CREATED, PLANNING, GENERATING, etc.)

3. **Agentes**:
   - Implementada clase base `Agent` con interfaz común
   - Implementados agentes específicos:
     - `GenerationAgent`: Genera hipótesis (literatura, debate)
     - `RankingAgent`: Compara y clasifica hipótesis en torneos
     - `ReflectionAgent`: Analiza hipótesis (revisión completa, verificación)
     - `EvolutionAgent`: Mejora hipótesis (mejora, simplificación, extensión)
     - `MetaReviewAgent`: Sintetiza resultados de investigación

4. **Modelos**:
   - Implementada clase base `LLMClient` para interactuar con modelos de lenguaje
   - Implementada clase `LLMProvider` como factory para diferentes proveedores
   - Implementado cliente para Azure OpenAI

5. **CLI**:
   - Implementados comandos básicos para interactuar con el sistema
   - Creada interfaz de línea de comandos con subcomandos para diferentes acciones

### Decisiones de diseño:

1. **Arquitectura modular**:
   - Separación clara entre componentes para facilitar extensiones futuras
   - Interfaces bien definidas entre módulos

2. **Gestión de estado**:
   - Estado persistente en archivos JSON
   - Estructura de datos clara para sesiones e iteraciones

3. **Comunicación entre agentes**:
   - Coordinador central que orquesta el flujo de trabajo
   - Contexto compartido para pasar información entre agentes

4. **Interacción con LLMs**:
   - Abstracción para diferentes proveedores de LLM
   - Configuración flexible a través de variables de entorno

### Limitaciones actuales:

1. No hay herramientas externas (búsqueda web, ejecución de código)
2. Sistema de memoria básico (archivos JSON)
3. No hay paralelización real de agentes (excepto en la fase de generación)
4. Interfaz de usuario limitada a línea de comandos

### Próximos pasos:

1. Probar el sistema con casos de uso reales
2. Mejorar la gestión de errores y recuperación
3. Optimizar prompts para los diferentes agentes
4. Implementar clientes para otros proveedores de LLM (OpenAI, Ollama)
5. Mejorar la documentación y ejemplos de uso

# ORIGINAL Dev steps:

## Step 1: inicial → esqueleto del sistema

En primer lugar tiene que construirse el FRAMEWORK BASICO MULTI AGENTE, como esta descripto en el paper. Tienen que haber multiples agentes que actuan, el supervisor que orquestra y demas componentes. TODOS los tipos de agentes tienen que estar presentes.

Tambien tiene que existir el concepto de session o project para cada una de las sesiones y de estados que tienen las sessions. Tiene que haber una forma de listar las sessiones creadas y ver su estado actual. La interaccion con el usuario tiene que ser mediante consola pero no de forma interactiva sino ejecutando comandos que sirvan para las partes del proceso. 

El user input tiene que ser solo texto, tienen que generarse multiples agentes que se ejecuten en paralelo con hipotesis y todo como esta descrito en el paper, y tienen que devolver hipotesis, ser evaluadas, scoreadas, competidas, rankeadas, etc, tal cual esta en el paper. NO SE VAN A USAR TOOLS POR EL MOMENTO!! NO VAMOS A HACER WEB SEARCH NI EJECUCION DE CODIGO NI NADA DE ESO. Las llms que generen hipotesis como las que evaluen, rankeen y mejoren las hipotesis van a hacerlo solo con su conocimiento propio, no van a buscar cosas externas.

Luego tiene que estar la parte del rankeo de las hipotesis y cuando termina la iteracion, se sale de la ejecucion y queda en el estado de espera de feedback del usuario. Esto se hace a traves de otro comando por consola, donde el input sera solo texto. Y asi se repetira el proceso, tal cual como se describe en el paper, logrando evolucionar las hipotesis y crear nuevas y mejores.

NO TIENE QUE HABER UN SISTEMA DE MEMORIA INTELIGENTE pero si una memoria basica, quizas un archivo de texto plano o algo asi, que siga los lineamientos del paper pero que sea algo muy simple y que funcione.

Entonces, la arquitectura deberia tener los componentes basicos: 

- user (scientist) interactua con el sistema, comenzando con un research goal en texto (Scientist describes a research goal along with preferences, experiment constraints, and other attributes. Add idea. Review idea. Discuss research)
- research plan configuration
- generation agents (Literature exploration. Simulated scientific debate)
- ranking agent tournaments (Research hypotheses comparison and ranking is performed through scientific debate in tournaments. Limitations and top win-loss patterns are summarized and provided as feedback to other agents. This enables iterative improvement in the quality of research hypothesis generation, creating a self-improving loop.)
- reflection agent (Full review (later with web search). Simulation review. Tournament review. Deep verification)
- evolution agents (Inspiration from other ideas. Simplification. Research extension)
- proximity agents
- meta-review agents (Research overview formulation)
- Tiene que tener un sistema basico de memoria,
- Tienen que haber sessions e iteraciones con feedback del usuario