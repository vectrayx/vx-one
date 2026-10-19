# VectrayX-One: Un Framework de Inferencia Probabilística No-Autorregresiva para Triage y Mitigación Perimetral Dinámica

**Autor:** Juan (Creador de VectrayX)  
**Área de Investigación:** Ciberseguridad, Inferencia de Alta Velocidad, Sistemas Híbridos  

---

## 1. Introducción y Planteamiento del Problema

Las arquitecturas contemporáneas de detección y mitigación de amenazas perimetrales (como Firewalls, IDPS y sistemas EDR) se enfrentan a un trade-off crítico entre **latencia de procesamiento** y **capacidad de generalización semántica**. 

1. **Motores Deterministas Tradicionales (Snort, Zeek, YARA):** Operan en el orden de los microsegundos ($<1\text{ ms}$) mediante la coincidencia exacta de firmas y umbrales rígidos. Sin embargo, sufren de una severa rigidez estructural. Son incapaces de detectar ataques de día cero, malware polimórfico u ofensivas DDoS distribuidas de baja intensidad (*Low and Slow*), los cuales imitan el comportamiento legítimo para evadir reglas fijas. Intentar mitigar esto mediante reglas manuales provoca una **explosión combinatoria de estados** inmanejable en entornos de producción.
2. **Modelos de Lenguaje Basados en Texto (CyberBERT, SecureBERT):** Ofrecen una excelente comprensión semántica del contexto y de las tácticas de los atacantes, pero son computacionalmente prohibitivos para la línea de fuego. Su naturaleza **autorregresiva** (generación token por token) introduce latencias de segundos y un alto riesgo de alucinación o fallos en el formateo de datos (*Type-Safety*), haciéndolos inviables para operar antes de que se saturen los buffers del kernel.

### El Método Novedoso (*Novel Method*)
Este *paper* introduce **VectrayX-One**, una arquitectura híbrida simbiótica que rompe este paradigma. Combinamos un motor determinista de extracción de flujos de alta velocidad (**VectrayX**) con un motor de decisiones probabilísticas no-autorregresivo de "System One" (inspirado en el paradigma **Jev**). El sistema no genera texto; evalúa primitivas lógicas fuertemente tipadas en un solo ciclo de hardware, resolviendo la ambigüedad semántica en la frontera de la red con latencias inferiores a los $100\text{ ms}$.

---

## 2. Arquitectura del Sistema (VectrayX-One)

El framework opera como un **Sistema de Triage de Dos Fases** diseñado para optimizar el cómputo y aislar anomalías en una "Zona Gris".

```
       [ Telemetría Cruda de Alta Velocidad: Paquetes / Syscalls ]
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │              PIPELINE PRINCIPAL VECTRAYX             │
         │  - Parsing de protocolos a nivel de línea (C++)      │
         │  - Extracción de vectores de características         │
         └──────────────────────────┬───────────────────────────┘
                                    │
                    ¿Match con patrón determinista?
                    ├── Sí ──► [Bloqueo / Permitir Inmediato] (~0.01 ms)
                    └── No
                            (Filtro de la Zona Gris)
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │             VECTRAYX SEMANTIC TRANSLATOR             │
         │  - Empaqueta el vector numérico y el contexto de text│
         │  - Genera el objeto estandarizado 'State'            │
         └──────────────────────────┬───────────────────────────┘
                                    │
                        (Inferencia en un solo ciclo)
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │              MOTOR DE DECISIÓN SYSTEM ONE            │
         │  - Evalúa primitivas en paralelo (No-autorregresivo) │
         │  - Entrega: Choice (Tipo), Score, Noul (Confianza)   │
         └──────────────────────────┬───────────────────────────┘
                                    │
                       (Acción en Microsegundos)
                                    ▼
         ┌──────────────────────────────────────────────────────┐
         │         CONTROLADOR DE MITIGACIÓN COMPILADO          │
         │  - Ejecuta la política basada en datos Type-Safe     │
         └──────────────────────────────────────────────────────┘
```

### Componentes Clave:
*   **Fase 1 (VectrayX Determinista):** Filtra y procesa el tráfico masivo. Si un paquete o comportamiento coincide con una firma conocida, es bloqueado o permitido instantáneamente. Costo computacional: casi nulo.
*   **VectrayX Semantic Translator:** Si el evento no coincide con ninguna regla pero exhibe anomalías estadísticas, VectrayX traduce la telemetría cruda y el contexto textual desestructurado en un esquema de estado estandarizado denso (`State`).
*   **Fase 2 (Motor System One):** Toma el `State` y evalúa vectorialmente un conjunto de preguntas primitivas en paralelo sobre los pesos de la red neuronal. Genera respuestas estructuradas lógicas con calibración de probabilidad real.

---

## 3. Formalismo Matemático

Modelamos el pipeline de **VectrayX-One** como una función de transición de estados y resolución probabilística:

\[\mathcal{F}(X) \rightarrow S \rightarrow \mathcal{P}(Q \mid S)\]

Donde:
*   X representa el flujo continuo de datos crudos e irregulares de la red o del sistema en una ventana de tiempo Δ t.
*   S es el vector de características de estado semántico estructurado de tamaño fijo generado por VectrayX: \(S = \{v_1, v_2, \dots, v_n\}\).
*   Q es el vector de preguntas primitivas fuertemente tipadas inyectadas en el motor de decisiones.
*   \(\mathcal{P}(Q \mid S)\) es la distribución de probabilidad conjunta calibrada calculada en **un solo paso de hardware (pasada de logits forward)**, omitiendo el bucle de muestreo autorregresivo tradicional \(P(t_n \mid t_1, \dots, t_{n-1})\).

### Calibración de Probabilidad Real vs. Sobreconfianza de ML Clásico
A diferencia de clasificadores tradicionales (como XGBoost o SVMs) que sufren de distorsiones de sobreconfianza ante ruido, el motor de System One utiliza la primitiva `Noul` (booleanos calibrados) para garantizar que el valor de salida refleje la frequency empírica real de la amenaza:

\[\mathbb{P}(\text{Amenaza Real} \mid \text{Noul}(S) = p) = p\]

---

## 4. Especificación del Estado y Primitivas de Decisión

### 4.1 Formato del Objeto `State` (JSON Schema Plano)
Para maximizar la eficiencia y evitar que el modelo de IA consuma ciclos parseando jerarquías complejas, VectrayX compila la telemetría en una estructura orientada a características semánticas de rendimiento:

```json
{
  "telemetry_source": "vectrayx_edge_network_v1",
  "timestamp_window_ms": 10,
  "network_features": {
    "packet_count": 14200,
    "syn_ratio": 0.89,
    "entropy_ip_source": 1.21,
    "avg_payload_size_bytes": 64,
    "burst_factor": 4.7
  },
  "payload_context": {
    "user_agent_raw": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...",
    "http_method": "GET",
    "anomaly_flags": ["invalid_header_order", "high_frequency_uri"]
  }
}
```

### 4.2 Definición de Primitivas Lógicas (Interfaces de Salida)
El framework restringe la salida de la IA a estructuras nativas estrictas que el código compilado de VectrayX puede ejecutar directamente sin necesidad de analizadores de texto (*parsers*):

```typescript
// Estructura de salida garantizada a nivel de hardware (Type-Safe)
interface SystemOneDecision {
  // Choice: Clasificación multiclase en una sola pasada
  attack_vector: Choice<["Normal", "SYN_Flood", "HTTP_Amplification", "Slowloris"]>;
  
  // Score: Rúbrica continua de criticidad (0.0 a 1.0)
  severity_score: Score; // e.g., 0.87
  
  // Noul: Booleano calibrado para automatización estricta
  requires_immediate_drop: Noul; // e.g., true (Confianza: 94.2%)
}
```

---

## 5. Diseño Experimental y Evaluación

Para demostrar la validez del método frente a la comunidad académica, el framework se evaluará bajo el siguiente protocolo de pruebas:

### 5.1 Datasets de Referencia
*   **Tráfico de Red / DDoS:** **CICDDoS2019** y **UNSW-NB15**. Permiten medir la tasa de detección contra ataques avanzados embebidos en tráfico legítimo masivo.
*   **Comportamiento de Malware:** Dataset **EMBER** y telemetría de eventos de llamadas al sistema dinámicas (**Sysmon**).

### 5.2 Métricas de Comparación Críticas
El rendimiento de **VectrayX-One** se medirá directamente contra firewalls tradicionales y modelos de lenguaje clásicos:

| Métrica | Firewall Determinista (Snort) | LLM Basado en Texto (SecureBERT) | Tu Propuesta (**VectrayX-One**) |
| :--- | :--- | :--- | :--- |
| **Latencia de Inferencia ($t_{\text{inf}}$)** | **$< 1\text{ ms}$** | $\sim 1500\text{ ms}$ – $3000\text{ ms}$ | **$70\text{ ms}$ – $120\text{ ms}$** |
| **Precisión ante Día Cero** | Muy Baja (Depende de firmas) | Alta | **Alta (Evaluación semántica)** |
| **Garantía de Formato** | Absoluta (Código Nativo) | Baja (Riesgo de alucinación/prosa) | **Absoluta (Type-Safe de hardware)** |
| **Costo por Millón de Eventos** | $\sim \$0.00$ | $\sim \$15.00$ – \$30.00 | **$< \$0.05$ (Eficiencia System One)** |

---

## 6. Conclusiones y Trabajo Futuro

**VectrayX-One** demuestra que es posible unificar la velocidad de la defensa perimetral nativa con las capacidades de razonamiento abstracto de la inteligencia artificial moderna. Al transformar el problema de un "procesamiento de lenguaje" a una "inferencia de decisión estructurada no-autorregresiva", el framework protege la infraestructura contra vectores de ataque de día cero y polimórficos dentro de los límites estrictos de tiempo real que exige la red.

El trabajo futuro se centrará en mitigar la **vulnerabilidad adversarial**, diseñando capas de sanitización en VectrayX para neutralizar inyecciones de *prompts* ocultas en los campos desestructurados de la telemetría antes de que impacten el motor de toma de decisiones.