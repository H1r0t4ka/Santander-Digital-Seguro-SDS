# 🛡️ Tablero Predictivo de Seguridad — SafeSant

![Imagen de Portada](docs/portada.png)

[![Streamlit](https://img.shields.io/badge/Streamlit-1.33+-EE2E31?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Next.js](https://img.shields.io/badge/Next.js-13+-000000?logo=next.js&logoColor=white)](https://nextjs.org)
[![Prisma](https://img.shields.io/badge/Prisma-ORM-2D3748?logo=prisma&logoColor=white)](https://prisma.io)
[![tRPC](https://img.shields.io/badge/tRPC-TypeSafe-3982CE)](https://trpc.io)
[![HuggingFace Spaces](https://img.shields.io/badge/Spaces-HuggingFace-FFCC00?logo=huggingface&logoColor=black)](https://huggingface.co/spaces)
[![Vercel](https://img.shields.io/badge/Deploy-Vercel-000000?logo=vercel&logoColor=white)](https://vercel.com)

Proyecto integral que combina un tablero geoespacial interactivo y un chatbot comunitario para monitorizar, explicar y prevenir fenómenos de seguridad en los municipios del departamento de Santander, usando datos oficiales de la Policía Nacional.

**Objetivo General**
- Desarrollar un tablero web inteligente que integre visualización geoespacial, análisis predictivo y un chatbot comunitario, con base en datos oficiales.

**Objetivos Específicos**
- OE1. Modelos de ML para predecir zonas, horarios y modalidades de mayor riesgo.
- OE2. Tablero web con filtros por municipio, cuadrante, tipo de delito y temporalidad.
- OE3. Chatbot comunitario que traduzca cifras en explicaciones claras y recomendaciones.

---

## 🚀 Características
- Visualización geoespacial con mapa de calor y puntos por municipio.
- Indicadores clave y ranking de municipios con mayor riesgo.
- Modo futuro (t+1) con “municipios en alto riesgo” y lista descargable.
- Filtros interactivos: modalidad, fuente, tipo de delito, municipio, año/mes.
- Chatbot “Gabi” accesible desde el portal web (ruta `/chatbot`).

---

## 🤖 Modelo de Machine Learning
- Objetivo: clasificar riesgo (`riesgo` o `riesgo_futuro`) por municipio/tiempo.
- Algoritmo: Árbol de Decisión (`max_depth=8`, `min_samples_leaf=10`).
- Entrenamiento:
  - Si hay coordenadas: usa `latitud`, `longitud`, `anio`, `mes`, `dia`, `dia_semana`.
  - Si no: convierte categóricas (`MUNICIPIO`, `DEPARTAMENTO`, `fuente`, `tipo_delito`) a dummies y selecciona columnas numéricas.

---

## 🗂️ Estructura del Repositorio
- `Dashboard/`
  - `app_tablero_streamlit_futuro.py` — tablero principal (Streamlit).
  - `requirements.txt` — dependencias del tablero.
- `app/` — portal web (Next.js, T3 stack)
  - `src/pages/chatbot/` — página del chatbot “Gabi”.
  - `src/env.mjs` — validación de variables de entorno.
  - `prisma/schema.prisma` — modelo de datos (PostgreSQL).
  - `vercel.json` — crons de ejemplo.
- `.streamlit/` — configuración de proyecto (si aplica en Spaces).

---

## 📦 Prerrequisitos
- Python 3.10+ y `pip`.
- Node.js 18+ y `npm`.
- Base de datos PostgreSQL (para el portal web con NextAuth + Prisma).

---

## 🧭 Tablero (Streamlit)

**Instalación**
- `pip install -r Dashboard/requirements.txt`

**Ejecución**
- `streamlit run Dashboard/app_tablero_streamlit_futuro.py`
- Fuente por defecto: `Local CSVs (si existen)`; el selector de fuente está oculto temporalmente.

**Datos de Entrada (CSV)**
- Ubicación: raíz del repo o `Dashboard/`.
- Detección automática del CSV más reciente por patrón:
  - Hurto: contiene `hurto`.
  - Delitos sexuales: `sexual` o `sexuales`.
  - Violencia intrafamiliar: `violencia`.
- Alternativamente, usa nombres completos: `Reporte_Hurto_por_Modalidades_Policía_Nacional_YYYYMMDD.csv`, `Reporte__Delitos_sexuales_Policía_Nacional_YYYYMMDD.csv`, `Reporte_Delito_Violencia_Intrafamiliar_YYYYMMDD.csv`.
- Solo se consideran filas con fecha válida y `DEPARTAMENTO = SANTANDER`.

**APIs Opcionales**
- [Reporte Hurto por Modalidades Policía Nacional](https://www.datos.gov.co/Seguridad-y-Defensa/Reporte-Hurto-por-Modalidades-Polic-a-Nacional/d4fr-sbn2/about_data)
- [Reporte Delito Violencia Intrafamiliar Policía Nacional](https://www.datos.gov.co/Seguridad-y-Defensa/Reporte-Delito-Violencia-Intrafamiliar-Polic-a-Nac/vuyt-mqpw/about_data)
- [Reporte Delitos Sexuales Policía Nacional](https://www.datos.gov.co/Seguridad-y-Defensa/Reporte-Delitos-Sexuales-Polic-a-Nacional/fpe5-yrmw/about_data)

- Endpoints: `fpe5-yrmw` (delitos sexuales), `d4fr-sbn2` (hurto), `vuyt-mqpw` (violencia).
- Variables:
  - `SOCRATA_APP_TOKEN` para mejorar cuota/estabilidad.
  - `MAPBOX_API_KEY` para estilos Mapbox; sin token se usa base CARTO.

---

## 💬 Chatbot (Next.js)

**Instalación y arranque**
- `cd app`
- `npm install`
- Crear `.env.local` con:
```
DATABASE_URL=postgresql://user:pass@host:5432/db
NODE_ENV=development
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=changeme
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
AZURE_AD_CLIENT_ID=...
AZURE_AD_CLIENT_SECRET=...
AZURE_AD_TENANT_ID=...
```
- `npm run dev`
- Accede a `http://localhost:3000/chatbot` o usa el botón flotante “Gabi”.

**Notas**
- NextAuth configurado con Google y Azure AD (ver `src/server/auth.ts`).
- Prisma y NextAuth requieren base de datos (`DATABASE_URL`).
- API tRPC en `/api/trpc`; utilidades en `src/utils/api.ts`.

---

## 🧪 Desarrollo
- Lint Next.js: `cd app && npm run lint`
- Generación Prisma: `cd app && npm run postinstall` (o `npx prisma generate`)
- Opcional (migraciones): `npx prisma migrate dev --name init`

---

## 🚀 Despliegue
- Streamlit: HuggingFace Spaces o Streamlit Cloud.
- Next.js: Vercel (incluye `vercel.json` con crons de ejemplo).
- Sitio en producción: https://santanderdigitalseguro.vercel.app/

- Dashboard (Streamlit):

![Tablero de Seguridad — Santander](docs/dashboard.png)

- Chatbot “Gabi” (Next.js):

![Chatbot Gabi — Portal web](docs/gabi.png)

---

## 📜 Licencia
- Pendiente de definir.

---

## 📧 Autores
- Andres Felipe Gonzalez Rodriguez — [LinkedIn](https://www.linkedin.com/in/afgonzalezr1993/) | [andrfgr@gmail.com](mailto:andrfgr@gmail.com)
- Arellys De Jesus Correa Rodriguez — [LinkedIn](https://www.linkedin.com/in/arellys-de-jes%C3%BAs-correa-rodr%C3%ADguez-745b0434/) | [arellys.correa@gmail.com](mailto:arellys.correa@gmail.com)
- Juan Manuel Quevedo Gonzalez — [LinkedIn](https://www.linkedin.com/in/juan-manuel-quevedo-gonzalez-324442194/) | [juanm_quevedo@hotmail.com](mailto:juanm_quevedo@hotmail.com)
