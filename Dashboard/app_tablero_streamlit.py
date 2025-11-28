import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import pydeck as pdk
import json
from shapely.geometry import shape
import unicodedata
import re

st.set_page_config(layout="wide", page_title="Tablero Predictivo - Santander")

st.title("🚨 Tablero Predictivo de Seguridad — Santander Digital Seguro SDS")

@st.cache_data
def load_local_csvs(base_path='.'):
    files = {
        'delitos_sex': 'Reporte__Delitos_sexuales_Policía_Nacional_20251118.csv',
        'hurto': 'Reporte_Hurto_por_Modalidades_Policía_Nacional_20251118.csv',
        'violencia': 'Reporte_Delito_Violencia_Intrafamiliar_Policía_Nacional_20251118.csv'
    }
    fallback = {
        'delitos_sex': 'Reporte__Delitos_sexuales_Policía_Nacional_20251118.csv',
        'hurto': 'Reporte_Hurto_por_Modalidades_Policía_Nacional_20251118.csv',
        'violencia': 'Reporte_Delito_Violencia_Intrafamiliar_20251118.csv'
    }
    dfs = {}
    for k, fname in files.items():
        path = os.path.join(base_path, fname)
        if os.path.exists(path):
            try:
                dfs[k] = pd.read_csv(path, sep=",", encoding="latin-1")
            except Exception:
                dfs[k] = pd.DataFrame()
        else:
            fb = os.path.join(base_path, fallback.get(k, ''))
            if fb and os.path.exists(fb):
                try:
                    dfs[k] = pd.read_csv(fb, sep=",", encoding="latin-1")
                except Exception:
                    dfs[k] = pd.DataFrame()
            else:
                dfs[k] = pd.DataFrame()
    return dfs['delitos_sex'], dfs['hurto'], dfs['violencia']


# ---------- Funciones para cargar desde las APIs (Socrata) ----------
BASE_URL = "https://www.datos.gov.co/resource/"
# IDs fieles tomados de `app_sincronizada_api.py`
RECURSO_SEXUAL_ID = "fpe5-yrmw"
RECURSO_VIOLENCIA_ID = "vuyt-mqpw"
RECURSO_HURTO_ID = "d4fr-sbn2"


def cargar_datos_desde_api(recurso_id, limit=50000):
    url = f"{BASE_URL}{recurso_id}.json?$limit={limit}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return pd.DataFrame(data)
    except Exception:
        return pd.DataFrame()

def cargar_datos_desde_api_all(recurso_id, page_size=50000, max_pages=500):
    frames = []
    offset = 0
    for _ in range(max_pages):
        headers = {}
        try:
            token = st.secrets.get('SOCRATA_APP_TOKEN', None)
        except Exception:
            token = os.getenv('SOCRATA_APP_TOKEN')
        if token:
            headers['X-App-Token'] = token
        url = f"{BASE_URL}{recurso_id}.json?$limit={page_size}&$offset={offset}&$order=fecha_hecho"
        try:
            resp = requests.get(url, timeout=30, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                alt_url = f"{BASE_URL}{recurso_id}.json?$limit={page_size}&$offset={offset}&$order=:id"
                resp = requests.get(alt_url, timeout=30, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
            frames.append(pd.DataFrame(data))
            if len(data) < page_size:
                break
            offset += page_size
        except Exception:
            break
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


@st.cache_data
def cargar_y_preprocesar_datos_api():
    def cargar_api_agrupada(recurso_id, tipo_col=None):
        headers = {}
        try:
            token = st.secrets.get('SOCRATA_APP_TOKEN', None)
        except Exception:
            token = os.getenv('SOCRATA_APP_TOKEN')
        if token:
            headers['X-App-Token'] = token

        select_parts = [
            "upper(municipio) as MUNICIPIO",
            "sum(cast(cantidad as integer)) as CANTIDAD",
            "extract(year from fecha_hecho) as anio",
            "extract(month from fecha_hecho) as mes",
        ]
        group_parts_expr = ["upper(municipio)", "extract(year from fecha_hecho)", "extract(month from fecha_hecho)"]
        if tipo_col:
            select_parts.append(f"{tipo_col} as tipo_delito")
            group_parts_expr.append(f"{tipo_col}")
        select = ",".join(select_parts)
        group = ",".join(group_parts_expr)
        where = "upper(departamento)='SANTANDER'"
        url = f"{BASE_URL}{recurso_id}.json"
        try:
            params = {"$select": select, "$where": where, "$group": group, "$limit": 50000}
            resp = requests.get(url, timeout=30, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data)
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            return df
        if 'MUNICIPIO' in df.columns:
            df['MUNICIPIO'] = (
                fix_mojibake_series(df['MUNICIPIO'].astype(str))
                .str.strip()
                .str.upper()
                .str.replace(r"\s+\(CT\)$", "", regex=True)
            )
        for c in ['CANTIDAD', 'anio', 'mes']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
        if 'tipo_delito' in df.columns:
            df['tipo_delito'] = fix_mojibake_series(df['tipo_delito'])
        return df

    delitos_sex = cargar_api_agrupada(RECURSO_SEXUAL_ID, tipo_col='delito')
    hurto = cargar_api_agrupada(RECURSO_HURTO_ID, tipo_col='tipo_de_hurto')
    violencia = cargar_api_agrupada(RECURSO_VIOLENCIA_ID, tipo_col=None)

    if delitos_sex is None or delitos_sex.empty:
        delitos_sex = cargar_datos_desde_api_all(RECURSO_SEXUAL_ID)
    if hurto is None or hurto.empty:
        hurto = cargar_datos_desde_api_all(RECURSO_HURTO_ID)
    if violencia is None or violencia.empty:
        violencia = cargar_datos_desde_api_all(RECURSO_VIOLENCIA_ID)

    if delitos_sex is not None and not delitos_sex.empty:
        delitos_sex['fuente'] = 'delitos_sexuales'
    if hurto is not None and not hurto.empty:
        hurto['fuente'] = 'hurto'
    if violencia is not None and not violencia.empty:
        violencia['fuente'] = 'violencia_intrafamiliar'

    for df in [delitos_sex, hurto, violencia]:
        if df is not None and not df.empty and 'dia_semana' not in df.columns:
            df['dia_semana'] = 0

    return delitos_sex, hurto, violencia


def normalizar_dep_mun(df, cols=None):
    if cols is None:
        cols = ['DEPARTAMENTO', 'MUNICIPIO']
    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(r"\s+\(CT\)$", "", regex=True)
            )
    return df


# Normalización compatible con `extract_municipio_centroids.py`
re_non_alnum = re.compile(r"[^A-Z0-9]+")
def normalize_name_for_merge(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.upper()
    s = re_non_alnum.sub(' ', s)
    s = ' '.join(s.split())
    return s

def _fix_mojibake(s):
    if s is None:
        return s
    t = str(s)
    if 'Ã' in t or 'Â' in t:
        try:
            return t.encode('latin1').decode('utf-8')
        except Exception:
            return t
    return t

def fix_mojibake_series(series):
    try:
        return series.astype(str).map(_fix_mojibake)
    except Exception:
        return series

def extract_tipo_delito(df):
    if df is None or df.empty:
        return None
    cols = [c for c in df.columns]
    targets = [
        'delito',
        'modalidad_del_hecho',
        'tipo_de_hurto',
        'tipo de hurto',
        'modalidad'
    ]
    col = next((c for c in cols if c.lower() in targets), None)
    if col is None:
        col = next((c for c in cols if ('delito' in c.lower()) or ('hurto' in c.lower()) or ('modalidad' in c.lower())), None)
    if col is not None:
        try:
            return df[col].astype(str)
        except Exception:
            return None
    return None


def prepare_riesgo_santander(delitos_sex, hurto, violencia):
    # Normalizar y parsear fechas. Hacemos detección robusta de nombres de columna
    for df in [delitos_sex, hurto, violencia]:
        if df is None or df.empty:
            continue
        # Asegurar que todos los nombres de columnas sean strings
        df.columns = [c if isinstance(c, str) else str(c) for c in df.columns]

        # Detectar columna de municipio/departamento en cualquier variante y normalizar
        municipio_col = next((c for c in df.columns if 'municipio' == c.lower() or 'municipio' in c.lower()), None)
        departamento_col = next((c for c in df.columns if 'departamento' == c.lower() or 'departamento' in c.lower()), None)
        if municipio_col is not None:
            df['MUNICIPIO'] = fix_mojibake_series(df[municipio_col].astype(str))
        if departamento_col is not None:
            df['DEPARTAMENTO'] = fix_mojibake_series(df[departamento_col].astype(str))

        # Aplicar normalización de texto en nuevas columnas si existen
        for col in ['DEPARTAMENTO', 'MUNICIPIO']:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.replace(r"\s+\(CT\)$", "", regex=True)
                )

        # parseo de fecha - detectar variantes
        fecha_col = next((c for c in df.columns if c.lower() in ('fecha hecho', 'fecha_hecho', 'fecha') or 'fecha' == c.lower()), None)
        if fecha_col is not None:
            df['fecha_hecho'] = pd.to_datetime(df[fecha_col], dayfirst=True, errors='coerce')

    # Añadir fuente y concatenar
    if delitos_sex is not None and not delitos_sex.empty:
        delitos_sex['fuente'] = 'delitos_sexuales'
        ts = extract_tipo_delito(delitos_sex)
        if ts is not None:
            delitos_sex['tipo_delito'] = fix_mojibake_series(ts)
    if hurto is not None and not hurto.empty:
        hurto['fuente'] = 'hurto'
        ts = extract_tipo_delito(hurto)
        if ts is not None:
            hurto['tipo_delito'] = fix_mojibake_series(ts)
    if violencia is not None and not violencia.empty:
        violencia['fuente'] = 'violencia_intrafamiliar'
        ts = extract_tipo_delito(violencia)
        if ts is not None:
            violencia['tipo_delito'] = fix_mojibake_series(ts)

    parts = [d for d in [delitos_sex, hurto, violencia] if d is not None and not d.empty]
    if not parts:
        return pd.DataFrame()

    # Seleccionar columnas base como en el flujo CSV para mantener filtros
    cols_base = ['DEPARTAMENTO','MUNICIPIO','fecha_hecho','CANTIDAD','fuente','tipo_delito']
    df_total = pd.concat([p[[c for c in cols_base if c in p.columns]] for p in parts], ignore_index=True)

    # Asegurar existencia y formato de CANTIDAD
    if 'CANTIDAD' in df_total.columns:
        df_total['CANTIDAD'] = pd.to_numeric(df_total['CANTIDAD'], errors='coerce').fillna(0).astype(int)
    else:
        df_total['CANTIDAD'] = 1

    # Derivar temporalidad, soportando entradas agregadas sin `fecha_hecho`
    if 'fecha_hecho' in df_total.columns:
        df_total.dropna(subset=['fecha_hecho'], inplace=True)
        df_total['anio'] = df_total['fecha_hecho'].dt.year
        df_total['mes'] = df_total['fecha_hecho'].dt.month
        df_total['dia'] = df_total['fecha_hecho'].dt.day
        df_total['dia_semana'] = df_total['fecha_hecho'].dt.weekday
    else:
        if 'anio' not in df_total.columns:
            df_total['anio'] = 0
        if 'mes' not in df_total.columns:
            df_total['mes'] = 0
        df_total['dia'] = 1
        if 'dia_semana' not in df_total.columns:
            df_total['dia_semana'] = 0

    # Reparar mojibake en texto clave
    for col in ['DEPARTAMENTO','MUNICIPIO','tipo_delito']:
        if col in df_total.columns:
            df_total[col] = fix_mojibake_series(df_total[col])

    # Filtrar a Santander
    df_santander = df_total[df_total.get('DEPARTAMENTO','').astype(str).str.upper()=='SANTANDER'].copy()
    if df_santander.empty:
        return pd.DataFrame()

    # Calcular riesgo por umbral de CANTIDAD (como en CSV)
    umbral = df_santander['CANTIDAD'].quantile(0.75)
    df_santander['riesgo'] = (df_santander['CANTIDAD'] >= umbral).astype(int)
    return df_santander

@st.cache_data
def cargar_y_limpiar_local_fiel_notebook(delitos_sex, hurto, violencia):
    def normalizar_dep_mun_fiel(df):
        for col in ["DEPARTAMENTO", "MUNICIPIO"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .str.replace(r"\s+\(CT\)$", "", regex=True)
                )
        return df

    if delitos_sex is not None and not delitos_sex.empty:
        delitos_sex = normalizar_dep_mun_fiel(delitos_sex)
        if "FECHA HECHO" in delitos_sex.columns:
            delitos_sex["FECHA HECHO"] = pd.to_datetime(delitos_sex["FECHA HECHO"], dayfirst=True, errors='coerce')
    if hurto is not None and not hurto.empty:
        hurto = normalizar_dep_mun_fiel(hurto)
        if "FECHA HECHO" in hurto.columns:
            hurto["FECHA HECHO"] = pd.to_datetime(hurto["FECHA HECHO"], dayfirst=True, errors='coerce')
    if violencia is not None and not violencia.empty:
        violencia = normalizar_dep_mun_fiel(violencia)
        if "FECHA HECHO" in violencia.columns:
            violencia["FECHA HECHO"] = pd.to_datetime(violencia["FECHA HECHO"], dayfirst=True, errors='coerce')

    if delitos_sex is not None and not delitos_sex.empty:
        delitos_sex = delitos_sex.copy()
        delitos_sex['fuente'] = 'delitos_sexuales'
        if 'delito' in delitos_sex.columns:
            delitos_sex['tipo_delito'] = fix_mojibake_series(delitos_sex['delito'].astype(str))
    if hurto is not None and not hurto.empty:
        hurto = hurto.copy()
        hurto['fuente'] = 'hurto'
        if 'TIPO DE HURTO' in hurto.columns:
            hurto['tipo_delito'] = fix_mojibake_series(hurto['TIPO DE HURTO'].astype(str))
    if violencia is not None and not violencia.empty:
        violencia = violencia.copy()
        violencia['fuente'] = 'violencia_intrafamiliar'
        violencia['tipo_delito'] = 'VIOLENCIA INTRAFAMILIAR'

    parts = [d for d in [delitos_sex, hurto, violencia] if d is not None and not d.empty]
    if not parts:
        return pd.DataFrame()

    cols_base = ['DEPARTAMENTO','MUNICIPIO','FECHA HECHO','CANTIDAD','fuente','tipo_delito']
    df_total = pd.concat([p[[c for c in cols_base if c in p.columns]] for p in parts], ignore_index=True)
    for col in ['DEPARTAMENTO','MUNICIPIO','tipo_delito']:
        if col in df_total.columns:
            df_total[col] = fix_mojibake_series(df_total[col])
    df_total.dropna(subset=['FECHA HECHO'], inplace=True)
    df_total['anio'] = df_total['FECHA HECHO'].dt.year
    df_total['mes'] = df_total['FECHA HECHO'].dt.month
    df_total['dia'] = df_total['FECHA HECHO'].dt.day
    df_total['dia_semana'] = df_total['FECHA HECHO'].dt.weekday
    df_total.rename(columns={'FECHA HECHO':'fecha_hecho'}, inplace=True)

    df_santander = df_total[df_total.get('DEPARTAMENTO','')=='SANTANDER'].copy()
    if df_santander.empty:
        return pd.DataFrame()
    umbral = df_santander['CANTIDAD'].quantile(0.75)
    df_santander['riesgo'] = (df_santander['CANTIDAD'] >= umbral).astype(int)
    return df_santander


@st.cache_resource
def train_model(df_riesgo):
    if df_riesgo is None or df_riesgo.empty:
        return None, None, None

    df = df_riesgo.copy()
    y = df['riesgo']

    # Si tenemos coordenadas, entrenar usando lat/long + temporales
    if 'latitud' in df.columns and 'longitud' in df.columns and df['latitud'].notna().any() and df['longitud'].notna().any():
        feature_cols = []
        for c in ['latitud', 'longitud', 'anio', 'mes', 'dia', 'dia_semana']:
            if c in df.columns:
                feature_cols.append(c)
        X = df[feature_cols].fillna(0)
    else:
        # Fallback robusto: convertir todas las categóricas relevantes a dummies y filtrar no numéricas
        cat_cols = [c for c in ['MUNICIPIO', 'DEPARTAMENTO', 'fuente', 'tipo_delito'] if c in df.columns]
        df_model = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        # Seleccionar únicamente columnas numéricas y excluir campos no relevantes
        X = (
            df_model.select_dtypes(include=['number'])
            .drop(['riesgo', 'CANTIDAD'], axis=1, errors='ignore')
            .fillna(0)
        )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
    model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    f1 = f1_score(y_test, pred, average='weighted') if y.nunique()>1 else None
    return model, X.columns.tolist(), f1


def predict_grid(model, columns, N=2000, center=(7.1132, -73.1190), scale=0.05, df_riesgo=None):
    lats = center[0] + (np.random.rand(N) - 0.5) * scale
    lons = center[1] + (np.random.rand(N) - 0.5) * scale
    df_points = pd.DataFrame({'latitud': lats, 'longitud': lons})

    if model is None or columns is None:
        df_points['probabilidad_riesgo'] = np.random.rand(N)
    else:
        # Construir DataFrame X con las mismas columnas usadas en entrenamiento
        X = pd.DataFrame(0, index=range(N), columns=columns)
        # Si modelo fue entrenado con lat/long, rellenar esas columnas
        for c in X.columns:
            if c == 'latitud':
                X[c] = df_points['latitud'].values
            elif c == 'longitud':
                X[c] = df_points['longitud'].values
            elif c in ('anio', 'mes', 'dia', 'dia_semana'):
                # usar medianas/mode desde df_riesgo si se pasó
                if df_riesgo is not None and c in df_riesgo.columns:
                    X[c] = int(df_riesgo[c].median()) if df_riesgo[c].dtype.kind in 'biufc' else df_riesgo[c].mode().iloc[0]
                else:
                    X[c] = 0
            else:
                # otras columnas (p.ej. dummies) rellenar 0
                X[c] = 0
        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception:
            probs = model.predict(X)
        df_points['probabilidad_riesgo'] = probs

    df_points['intensidad_riesgo'] = df_points['probabilidad_riesgo'] * 100
    # Añadir franja horaria y modalidad para permitir filtros como en `app_sincronizada_api.py`
    franja_opts = ['00:00-06:00', '06:00-12:00', '12:00-18:00', '18:00-00:00']
    modalidad_opts = ['Hurto a Persona', 'Hurto de Vehículo', 'Lesiones', 'Otro']
    df_points['franja_horaria'] = np.random.choice(franja_opts, size=len(df_points))
    df_points['modalidad'] = np.random.choice(modalidad_opts, size=len(df_points))
    return df_points


# Sidebar - fuente de datos
st.sidebar.header('Fuente de datos')
# Toggle para mostrar/ocultar mensajes de diagnóstico en UI
debug_logs = st.sidebar.checkbox('Mostrar logs y diagnósticos', False)
# Por defecto, inicia en 'Simulación' para evitar llamadas pesadas a APIs durante el arranque en hosting
source = st.sidebar.radio('Selecciona fuente:', (
    'APIs (Policía Nacional)',
    'Local CSVs (si existen)',
    'Simulación'
), index=2)

delitos_sex, hurto, violencia = load_local_csvs(base_path='.')
df_riesgo = pd.DataFrame()
model = None
model_cols = None
f1 = None

if source == 'Local CSVs (si existen)':
    df_riesgo = cargar_y_limpiar_local_fiel_notebook(delitos_sex, hurto, violencia)
    if not df_riesgo.empty:
        if debug_logs:
            st.sidebar.success(f'Encontrados datos locales: {df_riesgo.shape[0]} filas en df_riesgo')
        model, model_cols, f1 = train_model(df_riesgo)
        if model is not None and debug_logs:
            st.sidebar.success('Modelo entrenado localmente')
    else:
        if debug_logs:
            st.sidebar.warning('No se encontraron CSV locales; usando simulación')

elif source == 'APIs (Policía Nacional)':
    if debug_logs:
        st.sidebar.info('Consultando APIs de Policía Nacional... (puede tardar unos segundos)')
    delitos_sex_api, hurto_api, violencia_api = cargar_y_preprocesar_datos_api()
    # Intentar preparar df_riesgo con datos obtenidos
    df_riesgo = prepare_riesgo_santander(delitos_sex_api, hurto_api, violencia_api)
    if not df_riesgo.empty:
        if debug_logs:
            st.sidebar.success(f'Datos API cargados: {df_riesgo.shape[0]} filas en df_riesgo')
            try:
                st.sidebar.write('Filas por API:',
                                 {'sexuales': int(len(delitos_sex_api)), 'hurto': int(len(hurto_api)), 'violencia': int(len(violencia_api))})
            except Exception:
                pass
        model, model_cols, f1 = train_model(df_riesgo)
        if model is not None and debug_logs:
            st.sidebar.success('Modelo entrenado sobre datos API')
    else:
        if debug_logs:
            st.sidebar.warning('Las APIs no devolvieron datos útiles; usando simulación')

if source == 'Simulación' or (df_riesgo.empty if not df_riesgo is None else True):
    if debug_logs:
        st.sidebar.info('Usando datos simulados para demo')

# --- Añadir coordenadas de municipios a df_riesgo usando municipio_centroids.json ---
centroids_file = 'municipio_centroids.json'
if df_riesgo is not None and not df_riesgo.empty:
    try:
        if os.path.exists(centroids_file):
            with open(centroids_file, 'r', encoding='utf-8') as fh:
                cent_map = json.load(fh)
            # mapear usando la normalización
            df_riesgo['MUNICIPIO_NORM'] = df_riesgo['MUNICIPIO'].map(normalize_name_for_merge)
            df_riesgo['latitud'] = df_riesgo['MUNICIPIO_NORM'].map(lambda k: cent_map.get(k, {}).get('lat'))
            df_riesgo['longitud'] = df_riesgo['MUNICIPIO_NORM'].map(lambda k: cent_map.get(k, {}).get('lon'))
            # marcar cuántos municipios obtuvieron coordenadas
            n_with_coords = df_riesgo['latitud'].notna().sum()
            if debug_logs:
                st.sidebar.info(f'Municipios con coordenadas asociadas: {n_with_coords} / {df_riesgo.shape[0]} filas')
        else:
            if debug_logs:
                st.sidebar.info('No se encontró `municipio_centroids.json`; se intentará usar GeoJSON durante visualización')
    except Exception as e:
        if debug_logs:
            st.sidebar.warning(f'Error al anexar coordenadas a df_riesgo: {e}')

# KPIs
st.header('📊 Indicadores')
col1, col2, col3 = st.columns(3)
N_cells = st.sidebar.slider('Número de celdas a generar', 500, 20000, 2000, step=500)

municipios_cubiertos = 0
rango_temporal = 'N/A'
total_hechos = 0
if df_riesgo is not None and not df_riesgo.empty:
    try:
        municipios_cubiertos = int(df_riesgo['MUNICIPIO'].dropna().nunique()) if 'MUNICIPIO' in df_riesgo.columns else 0
    except Exception:
        municipios_cubiertos = 0
    try:
        total_hechos = int(pd.to_numeric(df_riesgo['CANTIDAD'], errors='coerce').fillna(0).sum()) if 'CANTIDAD' in df_riesgo.columns else 0
    except Exception:
        total_hechos = 0
    try:
        if 'anio' in df_riesgo.columns and 'mes' in df_riesgo.columns and df_riesgo['anio'].max() > 0:
            a_min = int(df_riesgo['anio'].min())
            a_max = int(df_riesgo['anio'].max())
            m_min = int(df_riesgo[df_riesgo['anio'] == a_min]['mes'].min())
            m_max = int(df_riesgo[df_riesgo['anio'] == a_max]['mes'].max())
            rango_temporal = f"{a_min}-{m_min:02d} a {a_max}-{m_max:02d}"
    except Exception:
        rango_temporal = 'N/A'

with col1:
    st.metric('Municipios cubiertos', municipios_cubiertos)
with col2:
    st.metric('Rango temporal', rango_temporal)
with col3:
    st.metric('Total hechos (CANTIDAD)', total_hechos)

# Generar predicciones
df_pred = predict_grid(model, model_cols, N=N_cells, df_riesgo=df_riesgo)

# --- Filtros (como en app_sincronizada_api.py) ---
st.sidebar.header('🔍 Filtros de Predicción')
mostrar_poligonos = st.sidebar.checkbox('Mostrar polígonos municipales', value=True)
tema_mapa = st.sidebar.selectbox('Tema del mapa', options=['Claro','Oscuro','Satélite'], index=0)
paleta_mapa = st.sidebar.selectbox('Paleta de calor', options=['YlOrRd (accesible)','Viridis (accesible)'], index=0)

# Cargar mapping de centroides para permitir filtros por municipio (bounding box)
muni_centroids = {}
centroids_file = 'municipio_centroids.json'
if os.path.exists(centroids_file):
    try:
        with open(centroids_file, 'r', encoding='utf-8') as fh:
            mapping = json.load(fh)
        for k, v in mapping.items():
            try:
                muni_centroids[k] = (float(v.get('lat')), float(v.get('lon')))
            except Exception:
                continue
    except Exception as e:
        if debug_logs:
            st.sidebar.warning(f'No se pudo leer {centroids_file}: {e}')

# Si no hay predicciones, dejamos el df_pred_filtrado vacío
if df_pred.empty:
    df_pred_filtrado = df_pred
else:
    # Nota: el filtro de `franja_horaria` se mantiene _desactivado temporalmente_
    # porque los datasets reales no contienen esa granularidad. Se deja generación
    # interna de franja en `predict_grid` para pruebas, pero no se usa como filtro.

    modalidad_opts = sorted(df_pred['modalidad'].unique())
    fuente_opts = []
    if df_riesgo is not None and not df_riesgo.empty and 'fuente' in df_riesgo.columns:
        fuente_opts = sorted(df_riesgo['fuente'].dropna().unique().tolist())
    tipo_opts = []
    if df_riesgo is not None and not df_riesgo.empty and 'tipo_delito' in df_riesgo.columns:
        tipo_opts = sorted([t for t in df_riesgo['tipo_delito'].dropna().unique().tolist() if isinstance(t, str)])

    # Filtros por Municipio: afecta principalmente la vista de municipios y el recorte
    # de la cuadrícula de predicción (se hace por bounding box alrededor de centroides)
    municipio_opts = []
    if df_riesgo is not None and not df_riesgo.empty:
        municipio_opts = sorted(df_riesgo['MUNICIPIO'].dropna().unique().tolist())
    else:
        municipio_opts = sorted([v.get('orig') for v in (mapping.values() if 'mapping' in locals() else []) if v.get('orig')])

    if 'f_modalidades' not in st.session_state:
        st.session_state['f_modalidades'] = modalidad_opts
    if fuente_opts and 'f_fuentes' not in st.session_state:
        st.session_state['f_fuentes'] = fuente_opts
    if tipo_opts and 'f_tipos' not in st.session_state:
        st.session_state['f_tipos'] = tipo_opts
    if 'f_municipios' not in st.session_state:
        st.session_state['f_municipios'] = municipio_opts
    if 'f_anio' not in st.session_state:
        st.session_state['f_anio'] = None
    if 'f_mes' not in st.session_state:
        st.session_state['f_mes'] = None

    if st.sidebar.button('Reestablecer filtros'):
        st.session_state['f_modalidades'] = modalidad_opts
        if fuente_opts:
            st.session_state['f_fuentes'] = fuente_opts
        if tipo_opts:
            st.session_state['f_tipos'] = tipo_opts
        st.session_state['f_municipios'] = municipio_opts
        st.session_state['f_anio'] = None
        st.session_state['f_mes'] = None
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    def _sanitize_default(opts, current):
        try:
            valid = [x for x in (current or []) if x in opts]
            return valid if valid else opts
        except Exception:
            return opts
    st.session_state['f_modalidades'] = _sanitize_default(modalidad_opts, st.session_state.get('f_modalidades', modalidad_opts))
    if fuente_opts:
        st.session_state['f_fuentes'] = _sanitize_default(fuente_opts, st.session_state.get('f_fuentes', fuente_opts))
    if tipo_opts:
        st.session_state['f_tipos'] = _sanitize_default(tipo_opts, st.session_state.get('f_tipos', tipo_opts))
    st.session_state['f_municipios'] = _sanitize_default(municipio_opts, st.session_state.get('f_municipios', municipio_opts))

    modalidades_seleccionadas = st.sidebar.multiselect('Filtrar por Modalidad de Riesgo:', options=modalidad_opts, default=st.session_state['f_modalidades'], key='f_modalidades')
    fuentes_seleccionadas = []
    if fuente_opts:
        fuentes_seleccionadas = st.sidebar.multiselect('Filtrar por Tipo de delito (fuente):', options=fuente_opts, default=st.session_state['f_fuentes'], key='f_fuentes')
    tipo_seleccionados = []
    if tipo_opts:
        tipo_seleccionados = st.sidebar.multiselect('Filtrar por Tipo específico de delito:', options=tipo_opts, default=st.session_state['f_tipos'], key='f_tipos')
    municipios_seleccionados = st.sidebar.multiselect('Filtrar por Municipio (afecta mapa):', options=municipio_opts, default=st.session_state['f_municipios'], key='f_municipios')

    # Filtros temporales básicos: Año / Mes (siempre opcionales)
    anio_selected = None
    mes_selected = None
    if df_riesgo is not None and not df_riesgo.empty:
        anios = sorted(df_riesgo['anio'].dropna().unique().astype(int).tolist())
        meses = sorted(df_riesgo['mes'].dropna().unique().astype(int).tolist())
        if anios:
            anio_opts = [None] + anios
            if st.session_state.get('f_anio') not in anio_opts:
                st.session_state['f_anio'] = None
            anio_selected = st.sidebar.selectbox('Año (opcional):', options=anio_opts, index=anio_opts.index(st.session_state['f_anio']), key='f_anio')
        if meses:
            mes_opts = [None] + meses
            if st.session_state.get('f_mes') not in mes_opts:
                st.session_state['f_mes'] = None
            mes_selected = st.sidebar.selectbox('Mes (opcional):', options=mes_opts, index=mes_opts.index(st.session_state['f_mes']), key='f_mes')

    # Aplicar filtros sobre las predicciones: modalidad + recorte espacial por municipio(s)
    df_pred_filtrado = df_pred[df_pred['modalidad'].isin(modalidades_seleccionadas)].copy()

    # DEBUG: información sobre filtrado para depuración en tiempo real
    try:
        if debug_logs:
            st.sidebar.markdown('**Debug filtros**')
            st.sidebar.write('Modalidades seleccionadas:', modalidades_seleccionadas)
            st.sidebar.write('Municipios seleccionados:', municipios_seleccionados[:20] if isinstance(municipios_seleccionados, (list, tuple)) else municipios_seleccionados)
            st.sidebar.write('Filas df_pred antes:', int(df_pred.shape[0]), 'después:', int(df_pred_filtrado.shape[0]))
            try:
                st.sidebar.write('Lat range:', float(df_pred_filtrado['latitud'].min()), '-', float(df_pred_filtrado['latitud'].max()))
                st.sidebar.write('Lon range:', float(df_pred_filtrado['longitud'].min()), '-', float(df_pred_filtrado['longitud'].max()))
            except Exception:
                pass
            if df_pred_filtrado.empty:
                st.sidebar.write('df_pred_filtrado está vacío tras aplicar filtros')
    except Exception:
        pass

    # Si el usuario seleccionó municipios, recortamos la cuadrícula a su(s) bounding box
    if municipios_seleccionados and muni_centroids:
        # construir máscara booleana que marque puntos cerca de cualquiera de los municipios
        masks = []
        buffer_deg = 0.12  # ~12 km a escala latitude (aprox) — valor conservador
        for m in municipios_seleccionados:
            key = normalize_name_for_merge(m)
            if key in muni_centroids and muni_centroids[key] is not None:
                lat_c, lon_c = muni_centroids[key]
                mask = (
                    (df_pred_filtrado['latitud'] >= (lat_c - buffer_deg)) &
                    (df_pred_filtrado['latitud'] <= (lat_c + buffer_deg)) &
                    (df_pred_filtrado['longitud'] >= (lon_c - buffer_deg)) &
                    (df_pred_filtrado['longitud'] <= (lon_c + buffer_deg))
                )
                masks.append(mask)
        if masks:
            combined = masks[0]
            for m in masks[1:]:
                combined = combined | m
            df_pred_filtrado = df_pred_filtrado[combined]

    # No filtramos por franja_horaria aquí (desactivado temporalmente)

    # Asignar MUNICIPIO aproximando por centroides más cercanos
    if muni_centroids and not df_pred_filtrado.empty and 'latitud' in df_pred_filtrado.columns and 'longitud' in df_pred_filtrado.columns:
        try:
            keys = list(muni_centroids.keys())
            arr_lat = np.array([muni_centroids[k][0] for k in keys])
            arr_lon = np.array([muni_centroids[k][1] for k in keys])
            names = []
            for k in keys:
                try:
                    names.append(mapping.get(k, {}).get('orig', k))
                except Exception:
                    names.append(k)
            names = np.array(names)

            latp = df_pred_filtrado['latitud'].values
            lonp = df_pred_filtrado['longitud'].values
            dists = (latp[:, None] - arr_lat[None, :])**2 + (lonp[:, None] - arr_lon[None, :])**2
            idx = dists.argmin(axis=1)
            df_pred_filtrado['MUNICIPIO'] = names[idx]
        except Exception:
            pass

st.header('🗺️ Mapa de Riesgo (simulado / modelo)')
if df_pred.empty:
    st.warning('No hay datos de predicción')
else:
    # Calcular centroide del mapa: preferir cuadrícula filtrada, luego puntos por municipio, luego todas las predicciones
    if not df_pred_filtrado.empty:
        midpoint = (np.mean(df_pred_filtrado['latitud']), np.mean(df_pred_filtrado['longitud']))
    else:
        midpoint = (None, None)
    layers = []
    # Estilo del mapa y paleta accesible con fallback si no hay token de Mapbox
    carto_styles = {
        'Claro': 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
        'Oscuro': 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
        'Satélite': 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'
    }
    mapbox_key = None
    try:
        mapbox_key = st.secrets.get('MAPBOX_API_KEY', None)
    except Exception:
        mapbox_key = os.getenv('MAPBOX_API_KEY') or os.getenv('MAPBOX_TOKEN')
    if mapbox_key:
        map_style = 'mapbox://styles/mapbox/light-v9' if tema_mapa=='Claro' else ('mapbox://styles/mapbox/dark-v9' if tema_mapa=='Oscuro' else 'mapbox://styles/mapbox/satellite-streets-v12')
        try:
            pdk.settings.mapbox_api_key = mapbox_key
        except Exception:
            pass
    else:
        map_style = carto_styles.get(tema_mapa, carto_styles['Claro'])
        if debug_logs:
            st.sidebar.info('Sin token de Mapbox: usando base CARTO (libre)')
    if paleta_mapa.startswith('YlOrRd'):
        color_range = [[255,255,178],[254,204,92],[253,141,60],[240,59,32],[189,0,38]]
    else:
        color_range = [[68,1,84],[58,82,139],[32,144,140],[94,201,98],[253,231,37]]

    # Si existe df_riesgo y GeoJSON, agregamos por municipio usando centroides
    geojson_path = 'santander_municipios.geojson'
    muni_points = None
    # Cargar mapping de centroides si existe (generado por extract_municipio_centroids.py)
    muni_centroids = {}
    centroids_file = 'municipio_centroids.json'
    if os.path.exists(centroids_file):
        try:
            with open(centroids_file, 'r', encoding='utf-8') as fh:
                mapping = json.load(fh)
            # mapping keys son name_norm -> {orig, mpio, lat, lon}
            for k, v in mapping.items():
                try:
                    muni_centroids[k] = (float(v['lat']), float(v['lon']))
                except Exception:
                    continue
        except Exception as e:
            if debug_logs:
                st.sidebar.warning(f'No se pudo leer {centroids_file}: {e}')
    else:
        # Si no existe el JSON, no usamos GeoJSON: avisar y continuar (el usuario pidió quitar GeoJSON)
        if debug_logs:
            st.sidebar.info('`municipio_centroids.json` no encontrado — no se usará GeoJSON; las coordenadas no estarán disponibles.')

    # Si tenemos df_riesgo, agregamos por MUNICIPIO y unimos con centroides
    if df_riesgo is not None and not df_riesgo.empty:
        df_muni_base = df_riesgo.copy()
        if fuentes_seleccionadas:
            df_muni_base = df_muni_base[df_muni_base['fuente'].isin(fuentes_seleccionadas)]
        if tipo_seleccionados:
            df_muni_base = df_muni_base[df_muni_base['tipo_delito'].isin(tipo_seleccionados)]
        if anio_selected is not None:
            df_muni_base = df_muni_base[df_muni_base['anio'] == anio_selected]
        if mes_selected is not None:
            df_muni_base = df_muni_base[df_muni_base['mes'] == mes_selected]
        df_muni = df_muni_base.groupby('MUNICIPIO', as_index=False)['CANTIDAD'].sum()
        # Normalizar nombres con la misma función usada para generar centroides
        df_muni['MUNICIPIO_NORM'] = df_muni['MUNICIPIO'].map(normalize_name_for_merge)
        # Si el usuario filtró por municipios, aplicar la selección aquí también
        try:
            if 'municipios_seleccionados' in locals() and municipios_seleccionados:
                sel_norm = set(normalize_name_for_merge(m) for m in municipios_seleccionados)
                df_muni = df_muni[df_muni['MUNICIPIO_NORM'].isin(sel_norm)].copy()
        except Exception:
            pass

        coords = df_muni['MUNICIPIO_NORM'].map(lambda x: muni_centroids.get(x))
        df_muni['coords'] = coords
        df_muni = df_muni[df_muni['coords'].notna()].copy()
        if not df_muni.empty:
            df_muni['latitud'] = df_muni['coords'].map(lambda c: c[0])
            df_muni['longitud'] = df_muni['coords'].map(lambda c: c[1])
            # intensidad normalizada para visualización
            df_muni['intensidad_riesgo'] = (df_muni['CANTIDAD'] - df_muni['CANTIDAD'].min()) / max(1, (df_muni['CANTIDAD'].max() - df_muni['CANTIDAD'].min()))
            # columna para mostrar nombre con acentos corregidos (fallback si falla)
            try:
                df_muni['MUNICIPIO_DISPLAY'] = fix_mojibake_series(df_muni['MUNICIPIO'])
            except Exception:
                df_muni['MUNICIPIO_DISPLAY'] = df_muni.get('MUNICIPIO', '').astype(str)
            muni_points = df_muni
            # Mostrar municipios no emparejados en la barra lateral para depuración
            all_munis = set(df_riesgo['MUNICIPIO'].astype(str).map(normalize_name_for_merge))
            matched = set(df_muni['MUNICIPIO_NORM'].astype(str))
            unmatched = sorted(list(all_munis - matched))
            if unmatched and debug_logs:
                st.sidebar.markdown('**Municipios no emparejados (ejemplos):**')
                for u in unmatched[:20]:
                    st.sidebar.write(u)

    # Si tenemos puntos por municipio, usar ScatterplotLayer para mostrarlos
    if muni_points is not None and not muni_points.empty:
        scatter = pdk.Layer(
            'ScatterplotLayer',
            muni_points,
            get_position=['longitud', 'latitud'],
            get_radius= 'intensidad_riesgo * 5000 + 2000',
            get_fill_color="[255 * intensidad_riesgo, 50, 50, 180]",
            pickable=True,
        )
        layers.append(scatter)
        # Etiquetas para municipios
        text_color = [0,0,0,255] if tema_mapa=='Claro' else [255,255,255,255]
        text_layer = pdk.Layer(
            'TextLayer',
            muni_points,
            get_position=['longitud', 'latitud'],
            get_text='MUNICIPIO_DISPLAY',
            get_color=text_color,
            get_size=14,
            get_angle=0,
            size_scale=1,
            pickable=False,
        )
        layers.append(text_layer)
    else:
        heatmap_layer = pdk.Layer(
            'HeatmapLayer',
            df_pred_filtrado,
            get_position=['longitud', 'latitud'],
            get_weight='intensidad_riesgo',
            radius=180,
            opacity=0.88,
            threshold=0.2,
            color_range=color_range,
        )
        layers.append(heatmap_layer)

    # Si no calculamos midpoint antes, intentar usar centroides municipales si existen
    if midpoint[0] is None or midpoint[1] is None:
        if muni_points is not None and not muni_points.empty:
            midpoint = (np.mean(muni_points['latitud']), np.mean(muni_points['longitud']))
        else:
            midpoint = (np.mean(df_pred['latitud']), np.mean(df_pred['longitud']))

    if mostrar_poligonos and os.path.exists(geojson_path):
        try:
            with open(geojson_path, 'r', encoding='utf-8') as f:
                santander_data = json.load(f)
            geojson_layer = pdk.Layer(
                'GeoJsonLayer',
                santander_data,
                pickable=True,
                stroked=True,
                filled=False,
                extruded=False,
                line_width_min_pixels=1,
                get_line_color='[0,0,0,180]'
            )
            layers.insert(0, geojson_layer)
        except Exception:
            if debug_logs:
                st.sidebar.warning('No fue posible cargar polígonos de municipios')
    if not muni_centroids:
        if debug_logs:
            st.sidebar.info('No hay coordenadas de municipios disponibles; el mapa mostrará solo predicciones simuladas/por cuadrícula.')

    view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=9, pitch=0)
    r = pdk.Deck(layers=layers, initial_view_state=view_state, map_style=map_style, tooltip={"text": "Riesgo: {intensidad_riesgo}\nMunicipio: {MUNICIPIO}"})
    st.pydeck_chart(r)

    st.subheader('Municipios con mayor riesgo')
    if muni_points is not None and not muni_points.empty:
        top_m = muni_points.sort_values('CANTIDAD', ascending=False).head(10)
        st.bar_chart(top_m.set_index('MUNICIPIO')['CANTIDAD'])
    elif df_riesgo is not None and not df_riesgo.empty:
        top_m = (
            df_riesgo.groupby('MUNICIPIO', as_index=False)['CANTIDAD'].sum()
            .sort_values('CANTIDAD', ascending=False)
            .head(10)
        )
        st.bar_chart(top_m.set_index('MUNICIPIO')['CANTIDAD'])

    if df_riesgo is not None and not df_riesgo.empty and 'MUNICIPIO' in df_riesgo.columns:
        if 'municipios_seleccionados' in locals() and isinstance(municipios_seleccionados, list) and len(municipios_seleccionados) == 1:
            m_sel = municipios_seleccionados[0]
            ts = df_riesgo[df_riesgo['MUNICIPIO'] == m_sel].groupby(['anio','mes'], as_index=False)['CANTIDAD'].sum()
            ts['fecha'] = pd.to_datetime(dict(year=ts['anio'], month=ts['mes'], day=1))
            st.subheader(f'Serie temporal de {m_sel}')
            st.line_chart(ts.set_index('fecha')['CANTIDAD'])

    cols_export = [c for c in ['MUNICIPIO','modalidad','probabilidad_riesgo','intensidad_riesgo'] if c in df_pred_filtrado.columns]
    export_df = df_pred_filtrado[cols_export].copy()
    st.download_button('Descargar predicciones filtradas', export_df.to_csv(index=False), 'predicciones_filtradas.csv', 'text/csv')
    st.caption(f"Filtradas: {int(export_df.shape[0])} / Totales: {int(df_pred.shape[0])}")
    try:
        df_pred_all_export = df_pred.copy()
        if muni_centroids and 'latitud' in df_pred_all_export.columns and 'longitud' in df_pred_all_export.columns:
            keys = list(muni_centroids.keys())
            arr_lat = np.array([muni_centroids[k][0] for k in keys])
            arr_lon = np.array([muni_centroids[k][1] for k in keys])
            names = []
            for k in keys:
                try:
                    names.append(mapping.get(k, {}).get('orig', k))
                except Exception:
                    names.append(k)
            names = np.array(names)
            latp = df_pred_all_export['latitud'].values
            lonp = df_pred_all_export['longitud'].values
            dists = (latp[:, None] - arr_lat[None, :])**2 + (lonp[:, None] - arr_lon[None, :])**2
            idx = dists.argmin(axis=1)
            df_pred_all_export['MUNICIPIO'] = names[idx]
        cols_export_all = [c for c in ['MUNICIPIO','modalidad','probabilidad_riesgo','intensidad_riesgo'] if c in df_pred_all_export.columns]
        st.download_button('Descargar todas las predicciones (sin filtros)', df_pred_all_export[cols_export_all].to_csv(index=False), 'predicciones_todas.csv', 'text/csv')
    except Exception:
        pass

    # Leyenda simple para intensidades
    st.sidebar.markdown('**Leyenda de Intensidad de Riesgo**')
    if paleta_mapa.startswith('YlOrRd'):
        legend_cols = [('#FFEDA0', '<=20%'), ('#FEC44F', '20-40%'), ('#FD8D3C', '40-60%'), ('#E31A1C', '60-80%'), ('#BD0026', '> 80%')]
    else:
        legend_cols = [('#440154', '<=20%'), ('#3B528B', '20-40%'), ('#20908C', '40-60%'), ('#5EC962', '60-80%'), ('#FDEB25', '> 80%')]
    for color, label in legend_cols:
        st.sidebar.markdown(f"<div style='display:flex;align-items:center'><div style='width:18px;height:12px;background:{color};margin-right:8px;border:1px solid #333'></div><span style='font-size:13px'>{label}</span></div>", unsafe_allow_html=True)

st.markdown('---')
if debug_logs:
    st.subheader('Tabla de Predicciones (muestras)')
    st.dataframe(df_pred.head(200))

if debug_logs:
    st.sidebar.markdown('---')
    st.sidebar.write('Archivos en el workspace:')
    for f in os.listdir('.'):
        if f.lower().endswith('.csv') or f.lower().endswith('.geojson'):
            st.sidebar.write(f)
