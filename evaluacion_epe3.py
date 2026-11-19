# ==================================================
# EVALUACION EPE3 - PRIVACIDAD Y MOROSIDAD
# ==================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from cryptography.fernet import Fernet

import hashlib
import hmac
import json
import time
import tracemalloc
import os
import re
from datetime import datetime, timezone, timedelta
import getpass
from pathlib import Path
from collections import Counter
import secrets
import sys
import warnings


# ==================================================
# FUNCIONES DE AUDITORIA (DEBEN CREARSE ANTES DE CARGAR)
# ==================================================

RUTA_PROGRAMA = Path(__file__).resolve().parent
RUTA_AUDITORIA = RUTA_PROGRAMA / "auditoria.jsonl"
hash_previo_auditoria = "0" * 64


def registrar(accion, antes, despues):
    """Registra una accion y la enlaza con el hash del registro anterior."""
    global hash_previo_auditoria

    registro_sin_hash = {
        "fecha_hora_utc": datetime.now(timezone.utc).isoformat(),
        "usuario": getpass.getuser(),
        "accion": accion,
        "filas_antes": int(antes),
        "filas_despues": int(despues),
        "hash_previo": hash_previo_auditoria,
    }

    texto_estable = json.dumps(
        registro_sin_hash,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    hash_actual = hashlib.sha256(texto_estable.encode("utf-8")).hexdigest()
    registro = dict(registro_sin_hash)
    registro["hash_actual"] = hash_actual

    try:
        with open(RUTA_AUDITORIA, "a", encoding="utf-8") as archivo:
            archivo.write(json.dumps(registro, ensure_ascii=False, sort_keys=True) + "\n")
        hash_previo_auditoria = hash_actual
        return True
    except OSError as error:
        print("Error al escribir la auditoria:", error)
        return False


def verificar_cadena(ruta_archivo=None):
    """Verifica hashes, enlazamiento y integridad del archivo JSONL."""
    if ruta_archivo is None:
        ruta_archivo = RUTA_AUDITORIA

    try:
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            registros = [json.loads(linea) for linea in archivo if linea.strip()]
    except (OSError, json.JSONDecodeError) as error:
        print("No fue posible leer la auditoria:", error)
        return False, 0

    hash_esperado_previo = "0" * 64
    cadena_integra = True

    for indice, registro in enumerate(registros, start=1):
        hash_anunciado = registro.pop("hash_actual", None)

        texto_estable = json.dumps(
            registro,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        hash_calculado = hashlib.sha256(texto_estable.encode("utf-8")).hexdigest()

        if hash_anunciado != hash_calculado:
            cadena_integra = False
        if registro.get("hash_previo") != hash_esperado_previo:
            cadena_integra = False

        hash_esperado_previo = hash_anunciado

    return cadena_integra, len(registros)


# ==================================================
# PARTE I - VALIDACION, LIMPIEZA Y MODELO
# ==================================================

def validar_datos(df):
    """Devuelve una lista con los problemas encontrados en el DataFrame."""
    problemas = []
    columnas_necesarias = [
        "id_cliente", "rut", "nombre", "email", "edad", "sexo",
        "region", "tipo_deuda", "monto_deuda", "ingreso_mensual",
        "num_cuotas", "tasa_interes_anual", "dias_mora",
        "consentimiento", "fecha_origen_deuda", "moroso",
    ]

    faltantes = [columna for columna in columnas_necesarias if columna not in df.columns]
    if faltantes:
        problemas.append("Columnas faltantes: " + ", ".join(faltantes))

    cantidad_duplicados = int(df.duplicated().sum())
    if cantidad_duplicados:
        problemas.append("Filas duplicadas: " + str(cantidad_duplicados))

    columnas_existentes = [columna for columna in columnas_necesarias if columna in df.columns]
    nulos = int(df[columnas_existentes].isna().sum().sum())
    if nulos:
        problemas.append("Valores nulos: " + str(nulos))

    if "edad" in df.columns:
        edad = pd.to_numeric(df["edad"], errors="coerce")
        cantidad = int(((edad.isna()) | (edad < 18) | (edad > 100)).sum())
        if cantidad:
            problemas.append("Edad fuera de rango o invalida: " + str(cantidad))

    if "ingreso_mensual" in df.columns:
        ingreso = pd.to_numeric(df["ingreso_mensual"], errors="coerce")
        cantidad = int(((ingreso.isna()) | (ingreso < 1)).sum())
        if cantidad:
            problemas.append("Ingreso mensual invalido o menor que 1: " + str(cantidad))

    if "monto_deuda" in df.columns:
        monto = pd.to_numeric(df["monto_deuda"], errors="coerce")
        cantidad = int(((monto.isna()) | (monto < 1)).sum())
        if cantidad:
            problemas.append("Monto de deuda invalido o menor que 1: " + str(cantidad))

    if "dias_mora" in df.columns:
        dias = pd.to_numeric(df["dias_mora"], errors="coerce")
        cantidad = int(((dias.isna()) | (dias < 0)).sum())
        if cantidad:
            problemas.append("Dias de mora negativos o invalidos: " + str(cantidad))

    if "consentimiento" in df.columns:
        consentimiento = df["consentimiento"].astype("string").str.strip()
        cantidad = int((~consentimiento.isin(["Sí", "No"]) | consentimiento.isna()).sum())
        if cantidad:
            problemas.append("Consentimiento distinto de Sí o No: " + str(cantidad))

    return problemas


def limpiar(df):
    """Aplica las reglas de limpieza y devuelve el DataFrame y los recuentos."""
    resultado = df.copy()
    eliminaciones = {}

    antes = len(resultado)
    resultado = resultado.drop_duplicates()
    eliminaciones["duplicados"] = antes - len(resultado)

    if "consentimiento" in resultado.columns:
        antes = len(resultado)
        consentimiento = resultado["consentimiento"].astype("string").str.strip()
        resultado = resultado[consentimiento.isin(["Sí"])].copy()
        eliminaciones["sin_consentimiento"] = antes - len(resultado)
    else:
        eliminaciones["sin_consentimiento"] = 0
        print("No existe la columna consentimiento; no fue posible eliminar esos registros.")

    if "edad" in resultado.columns:
        resultado["edad"] = pd.to_numeric(resultado["edad"], errors="coerce")
        edad_valida = resultado["edad"].where(resultado["edad"].between(18, 100), np.nan)
        mediana_edad = edad_valida.median()
        reemplazar_edad = resultado["edad"].isna() | ~resultado["edad"].between(18, 100)
        cantidad = int(reemplazar_edad.sum())
        if pd.isna(mediana_edad):
            print("Problema de limpieza: no existe una mediana valida de edad.")
        else:
            resultado.loc[reemplazar_edad, "edad"] = mediana_edad
        eliminaciones["edad_reemplazada"] = cantidad
    else:
        eliminaciones["edad_reemplazada"] = 0

    if "ingreso_mensual" in resultado.columns:
        resultado["ingreso_mensual"] = pd.to_numeric(
            resultado["ingreso_mensual"], errors="coerce"
        )
        ingreso_valido = resultado["ingreso_mensual"].where(
            resultado["ingreso_mensual"] >= 1, np.nan
        )
        mediana_ingreso = ingreso_valido.median()
        reemplazar_ingreso = resultado["ingreso_mensual"].isna() | (resultado["ingreso_mensual"] < 1)
        cantidad = int(reemplazar_ingreso.sum())
        if pd.isna(mediana_ingreso):
            print("Problema de limpieza: no existe una mediana valida de ingreso mensual.")
        else:
            resultado.loc[reemplazar_ingreso, "ingreso_mensual"] = mediana_ingreso
        eliminaciones["ingreso_reemplazado"] = cantidad
    else:
        eliminaciones["ingreso_reemplazado"] = 0

    if "monto_deuda" in resultado.columns:
        resultado["monto_deuda"] = pd.to_numeric(resultado["monto_deuda"], errors="coerce")
        antes = len(resultado)
        monto_valido = resultado["monto_deuda"].notna() & (resultado["monto_deuda"] >= 1)
        resultado = resultado[monto_valido].copy()
        eliminaciones["monto_invalido"] = antes - len(resultado)
    else:
        eliminaciones["monto_invalido"] = 0

    if "dias_mora" in resultado.columns:
        resultado["dias_mora"] = pd.to_numeric(resultado["dias_mora"], errors="coerce")
        antes = len(resultado)
        dias_validos = resultado["dias_mora"].notna() & (resultado["dias_mora"] >= 0)
        resultado = resultado[dias_validos].copy()
        eliminaciones["dias_mora_invalidos"] = antes - len(resultado)
    else:
        eliminaciones["dias_mora_invalidos"] = 0

    return resultado, eliminaciones


def calcular_kanonimato(df, columnas):
    """Calcula k minimo, tamanos de grupo y porcentaje de grupos unicos."""
    if df.empty:
        return 0, [], 0.0

    for columna in columnas:
        if columna not in df.columns:
            return 0, [], 0.0

    tamanos = df.groupby(columnas, dropna=False).size()
    k_minimo = int(tamanos.min())
    grupos_unicos = int((tamanos == 1).sum())
    porcentaje_unicos = 100.0 * int(tamanos[tamanos == 1].sum()) / len(df)
    return k_minimo, list(tamanos.astype(int)), porcentaje_unicos


def obtener_rango_etario(edad):
    """Convierte una edad exacta en los rangos solicitados."""
    if pd.isna(edad):
        return "Desconocido"
    edad = float(edad)
    if 18 <= edad <= 29:
        return "18-29"
    if 30 <= edad <= 44:
        return "30-44"
    if 45 <= edad <= 59:
        return "45-59"
    if edad >= 60:
        return "60+"
    return "Desconocido"


def medir_etapa(nombre, funcion):
    """Mide tiempo y memoria pico de una funcion."""
    tracemalloc.start()
    inicio = time.perf_counter()
    resultado = funcion()
    tiempo = time.perf_counter() - inicio
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memoria_mb = pico / (1024 * 1024)
    print(f"{nombre}: tiempo={tiempo:.4f} s | memoria pico={memoria_mb:.2f} MB")
    return resultado, tiempo, memoria_mb


# ==================================================
# PARTE II - PRIVACIDAD, ANONIMIZACION Y CIFRADO
# ==================================================

HASH_ARCHIVO_CIFRADO_ESPERADO = None


def recibir(contenido, clave):
    """Verifica SHA-256 y luego descifra el contenido con Fernet."""
    global HASH_ARCHIVO_CIFRADO_ESPERADO

    if HASH_ARCHIVO_CIFRADO_ESPERADO is None:
        return None, "No existe un hash esperado"

    hash_recibido = hashlib.sha256(contenido).hexdigest()
    if hash_recibido != HASH_ARCHIVO_CIFRADO_ESPERADO:
        return None, "ARCHIVO ALTERADO: checksum SHA-256 no coincide"

    try:
        contenido_descifrado = Fernet(clave).decrypt(contenido)
        return contenido_descifrado, "RECEPCION NORMAL: checksum valido y clave correcta"
    except Exception as error:
        return None, "CLAVE INCORRECTA O CIFRADO INVALIDO: " + str(type(error).__name__)


# ==================================================
# PROGRAMA PRINCIPAL
# ==================================================

def main():
    global hash_previo_auditoria, HASH_ARCHIVO_CIFRADO_ESPERADO

    print("==================================================")
    print("EVALUACION EPE3")
    print("==================================================")
    print("Generara los resultados en:", RUTA_PROGRAMA)

    # Una nueva ejecucion comienza una auditoria nueva sin mezclar archivos.
    hash_previo_auditoria = "0" * 64
    if RUTA_AUDITORIA.exists():
        try:
            RUTA_AUDITORIA.unlink()
            print("Se inicio una nueva auditoria; el contenido anterior fue reemplazado.")
        except OSError as error:
            print("No fue posible reiniciar auditoria.jsonl:", error)

    # PREGUNTA 1 - Carga y validacion inicial
    print("\nPARTE I - VALIDACION, LIMPIEZA Y MODELO")
    print("PREGUNTA 1 - VALIDACION INICIAL")

    ruta_csv = RUTA_PROGRAMA / "deudas_clientes_epe3.csv"

    if not ruta_csv.exists():
        print("ERROR: No existe deudas_clientes_epe3.csv en la carpeta del programa.")
        print("El programa no puede continuar con los datos.")
        return

    def cargar_csv():
        return pd.read_csv(ruta_csv)

    try:
        df_original, tiempo_carga, memoria_carga = medir_etapa("Carga del CSV", cargar_csv)
    except Exception as error:
        print("Error al cargar el CSV:", error)
        print("Instale las librerias con: pip install pandas numpy scikit-learn cryptography")
        return

    problemas_iniciales = validar_datos(df_original)
    print("Filas iniciales:", len(df_original))
    print("Problemas iniciales:")
    if problemas_iniciales:
        for problema in problemas_iniciales:
            print("-", problema)
    else:
        print("- Sin problemas detectados.")

    # PREGUNTA 1 - Limpieza
    print("\nPREGUNTA 1 - RESULTADO DE LIMPIEZA")

    def ejecutar_limpieza():
        return limpiar(df_original)

    (df_limpio, eliminaciones), tiempo_limpieza, memoria_limpieza = medir_etapa(
        "Limpieza de datos", ejecutar_limpieza
    )

    for regla, cantidad in eliminaciones.items():
        print(f"Eliminadas/reemplazadas por {regla}: {cantidad}")
    print("Filas iniciales:", len(df_original))
    print("Filas finales:", len(df_limpio))

    registrar("Limpieza de datos", len(df_original), len(df_limpio))

    problemas_finales = validar_datos(df_limpio)
    print("Problemas finales:")
    if problemas_finales:
        for problema in problemas_finales:
            print("-", problema)
    else:
        print("- Sin problemas relevantes detectados.")

    # PREGUNTA 2 - Modelos predictivos
    print("\nPREGUNTA 2 - MODELO PREDICTIVO")

    resultados_modelos = []
    modelo_disponible = True

    try:
        if len(df_limpio) == 0:
            raise ValueError("No hay registros despues de la limpieza.")

        df_modelo = df_limpio.copy()
        columnas_numericas_modelo = [
            "edad", "monto_deuda", "ingreso_mensual", "num_cuotas",
            "tasa_interes_anual",
        ]

        for columna in columnas_numericas_modelo:
            if columna not in df_modelo.columns:
                raise ValueError("Falta la columna predictora: " + columna)
            df_modelo[columna] = pd.to_numeric(df_modelo[columna], errors="coerce")

        denominador = df_modelo["ingreso_mensual"] * 12
        df_modelo["carga_financiera"] = np.where(
            (denominador.notna()) & (denominador > 0),
            df_modelo["monto_deuda"] / denominador * 100,
            np.nan,
        )

        predictor = [
            "edad", "monto_deuda", "ingreso_mensual", "num_cuotas",
            "tasa_interes_anual", "carga_financiera",
        ]
        X = df_modelo[predictor]
        y = pd.to_numeric(df_modelo["moroso"], errors="coerce")

        valido = X.notna().all(axis=1) & y.notna()
        X = X.loc[valido]
        y = y.loc[valido]

        if X.isna().any().any():
            raise ValueError("Existen valores nulos en las variables predictoras.")

        if y.nunique() < 2:
            raise ValueError("La variable moroso tiene una sola clase.")

        conteo_clases = y.value_counts()
        if conteo_clases.min() < 2:
            raise ValueError("Cantidad insuficiente de registros para separar las clases.")

        def entrenar_modelos():
            X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
                X,
                y,
                test_size=0.25,
                stratify=y,
                random_state=42,
            )

            pipeline_logistico = Pipeline([
                ("escalado", StandardScaler()),
                ("modelo", LogisticRegression(class_weight="balanced", max_iter=1000)),
            ])
            pipeline_logistico.fit(X_entrenamiento, y_entrenamiento)
            pred_logistico = pipeline_logistico.predict(X_prueba)

            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_entrenamiento, y_entrenamiento)
            pred_dummy = dummy.predict(X_prueba)

            resultados = []
            resultados.append({
                "modelo": "LogisticRegression",
                "accuracy": accuracy_score(y_prueba, pred_logistico),
                "precision": precision_score(y_prueba, pred_logistico, zero_division=0),
                "recall": recall_score(y_prueba, pred_logistico, zero_division=0),
            })
            resultados.append({
                "modelo": "DummyClassifier",
                "accuracy": accuracy_score(y_prueba, pred_dummy),
                "precision": precision_score(y_prueba, pred_dummy, zero_division=0),
                "recall": recall_score(y_prueba, pred_dummy, zero_division=0),
            })
            return resultados

        resultados_modelos, tiempo_modelos, memoria_modelos = medir_etapa(
            "Entrenamiento de modelos", entrenar_modelos
        )
        tabla_modelos = pd.DataFrame(resultados_modelos)
        print(tabla_modelos.to_string(index=False))

        # Accuracy no basta ante posible desequilibrio de clases en morosidad.
        # Precision indica cuántos avisos eran realmente morosos.
        # Recall indica cuántos morosos reales fueron detectados.
        print("Nota: la accuracy puede ocultar errores si existe un gran desequilibrio.")
        print("La precision evalua falsos positivos; el recall evalua morosos no detectados.")

    except Exception as error:
        modelo_disponible = False
        tiempo_modelos = 0.0
        memoria_modelos = 0.0
        print("No fue posible entrenar los modelos:", error)
        print("Se continua con las etapas de privacidad y proteccion.")

    # PREGUNTA 3 - Tiempo y memoria
    print("\nPREGUNTA 3 - TIEMPO Y MEMORIA")
    mediciones = [
        ("Carga del CSV", tiempo_carga, memoria_carga),
        ("Limpieza de datos", tiempo_limpieza, memoria_limpieza),
        ("Entrenamiento de modelos", tiempo_modelos, memoria_modelos),
    ]
    for nombre, tiempo, memoria in mediciones:
        print(f"{nombre}: tiempo={tiempo:.4f} s | memoria pico={memoria:.2f} MB")

    etapa_mas_costosa = max(mediciones, key=lambda registro: registro[1])
    print("Etapa mas costosa en tiempo:", etapa_mas_costosa[0])
    # Como optimizacion, se podria cargar solo las columnas necesarias y evitar copias repetidas.
    # Para mas datos conviene muestrear solo para pruebas o usar tipos numericos mas eficientes.
    print("Optimizacion propuesta: leer columnas necesarias y reducir copias intermedias.")

    # ==================================================
    # PARTE II - PRIVACIDAD
    # ==================================================
    print("\nPARTE II - PRIVACIDAD Y PROTECCION")
    print("PREGUNTA 1 - IDENTIFICADORES Y K-ANONIMATO")

    df_privacidad = df_limpio.copy()

    rut_regex = r"(?<![A-Za-z0-9])\d{1,2}(?:\.?\d{3}){2}-[\dkK](?![A-Za-z0-9])"
    email_regex = r"[A-Za-z0-9.!$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"

    cantidad_rut = 0
    cantidad_email = 0
    muestra_rut = []
    muestra_email = []

    if "rut" in df_privacidad.columns:
        coincidencias = df_privacidad["rut"].astype("string").fillna("").str.findall(rut_regex)
        cantidad_rut = int(sum(len(grupo) for grupo in coincidencias))
        muestra_rut = [grupo[0] for grupo in coincidencias if len(grupo)]
    if "email" in df_privacidad.columns:
        coincidencias = df_privacidad["email"].astype("string").fillna("").str.findall(email_regex)
        cantidad_email = int(sum(len(grupo) for grupo in coincidencias))
        muestra_email = [grupo[0] for grupo in coincidencias if len(grupo)]

    cantidad_nombres = 0
    if "nombre" in df_privacidad.columns:
        cantidad_nombres = int(df_privacidad["nombre"].astype("string").notna().sum())

    print("Identificadores directos detectados:")
    print("- RUT:", cantidad_rut)
    print("- Email:", cantidad_email)
    print("- Nombre:", cantidad_nombres)
    print("- Ejemplos omitidos por privacidad:", len(set(muestra_rut[:3]) | set(muestra_email[:3])))

    if "edad" in df_privacidad.columns:
        df_privacidad["rango_etario"] = df_privacidad["edad"].apply(obtener_rango_etario)
    else:
        df_privacidad["rango_etario"] = "Desconocido"

    k1, tamanos1, porcentaje1 = calcular_kanonimato(
        df_privacidad, ["region", "edad", "sexo"]
    ) if all(col in df_privacidad.columns for col in ["region", "edad", "sexo"]) else (0, [], 0.0)
    k2, tamanos2, porcentaje2 = calcular_kanonimato(
        df_privacidad, ["region", "rango_etario", "sexo"]
    ) if all(col in df_privacidad.columns for col in ["region", "rango_etario", "sexo"]) else (0, [], 0.0)

    def nivel_riesgo(k):
        if k == 1:
            return "ALTO"
        if k < 5:
            return "MEDIO"
        return "BAJO"

    print("\nESCENARIO 1 - region + edad exacta + sexo")
    print("k minimo:", k1)
    print("Tamano de los grupos:", len(tamanos1), "grupos")
    print("Porcentaje en grupos unicos:", f"{porcentaje1:.2f}%")
    print("Nivel de riesgo:", nivel_riesgo(k1))

    print("\nESCENARIO 2 - region + rango_etario + sexo")
    print("k minimo:", k2)
    print("Tamano de los grupos:", len(tamanos2), "grupos")
    print("Porcentaje en grupos unicos:", f"{porcentaje2:.2f}%")
    print("Nivel de riesgo:", nivel_riesgo(k2))
    # Generalizar la edad reduce la precision y agrupa mas personas con rasgos similares.
    # Esto puede disminuir el riesgo de identificacion individual, aunque no elimina el reidentificacion.

    # PREGUNTA 2 - Retencion y proteccion
    print("\nPREGUNTA 2 - RETENCION Y ANONIMIZACION")
    fecha_referencia = datetime(2026, 11, 19)
    limite_retencion = fecha_referencia.replace(year=fecha_referencia.year - 5)

    if "fecha_origen_deuda" not in df_limpio.columns:
        print("No existe fecha_origen_deuda; no fue posible aplicar la retencion.")
        df_compartir = df_limpio.copy()
        fechas_invalidas = len(df_compartir)
        fuera_retencion = 0
    else:
        fechas = pd.to_datetime(
            df_limpio["fecha_origen_deuda"], errors="coerce", format="mixed"
        )
        fechas_invalidas = int(fechas.isna().sum())
        dentro_retencion = fechas.notna() & (fechas >= pd.Timestamp(limite_retencion)) & (fechas <= pd.Timestamp(fecha_referencia))
        fuera_retencion = int((~dentro_retencion).sum())
        df_compartir = df_limpio.loc[dentro_retencion].copy()

    print("Fecha de referencia:", fecha_referencia.strftime("%d-%m-%Y"))
    print("Limite de 5 anos:", limite_retencion.strftime("%d-%m-%Y"))
    print("Fechas invalidas:", fechas_invalidas)
    print("Registros fuera de retencion o invalidos:", fuera_retencion)
    print("Registros que deben excluirse:", fuera_retencion)

    registrar("Exclusion por retencion", len(df_limpio), len(df_compartir))

    clave_hmac_env = os.environ.get("CLAVE_HMAC")
    if clave_hmac_env:
        clave_hmac = clave_hmac_env.encode("utf-8")
        print("HMAC: usando CLAVE_HMAC del entorno.")
    else:
        clave_hmac = secrets.token_bytes(32)
        print("ADVERTENCIA: CLAVE_HMAC no existe; se uso una clave temporal.")
        print("La clave temporal no es adecuada para produccion.")

    if "rut" in df_compartir.columns:
        df_compartir["rut_hash"] = df_compartir["rut"].apply(
            lambda valor: hmac.new(
                clave_hmac,
                "" if pd.isna(valor) else str(valor).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        )

    if "ingreso_mensual" in df_compartir.columns:
        ingreso = pd.to_numeric(df_compartir["ingreso_mensual"], errors="coerce")
        piso = np.floor(ingreso / 500000) * 500000
        df_compartir["tramo_ingreso"] = piso.apply(
            lambda valor: "Sin dato" if pd.isna(valor) else f"{int(valor)}-{int(valor + 499999)}"
        )

    if "edad" in df_compartir.columns:
        df_compartir["rango_etario"] = df_compartir["edad"].apply(obtener_rango_etario)

    columnas_eliminar = [
        "id_cliente", "rut", "nombre", "email", "edad",
        "ingreso_mensual", "fecha_origen_deuda", "consentimiento",
    ]
    for columna in columnas_eliminar:
        if columna in df_compartir.columns:
            df_compartir = df_compartir.drop(columns=[columna])

    cuasi_identificadores = ["region", "rango_etario", "sexo"]
    if all(columna in df_compartir.columns for columna in cuasi_identificadores):
        tamanos_grupos = df_compartir.groupby(
            cuasi_identificadores, dropna=False
        )[cuasi_identificadores[0]].transform("size")
        conservar = tamanos_grupos >= 5
        filas_antes_suprimir = len(df_compartir)
        df_final_protegido = df_compartir.loc[conservar].copy()
        filas_suprimidas = filas_antes_suprimir - len(df_final_protegido)
    else:
        df_final_protegido = df_compartir.copy()
        filas_suprimidas = 0
        print("No fue posible suprimir por cuasi-identificadores: faltan columnas.")

    k_final, _, porcentaje_unicos_final = calcular_kanonimato(
        df_final_protegido,
        cuasi_identificadores,
    ) if all(columna in df_final_protegido.columns for columna in cuasi_identificadores) else (0, [], 0.0)

    print("Filas suprimidas por grupos menores a 5:", filas_suprimidas)
    print("Cantidad final de registros:", len(df_final_protegido))
    print("k-anonimato final:", k_final)
    print("Porcentaje de registros en grupos unicos:", f"{porcentaje_unicos_final:.2f}%")

    if k_final >= 5:
        print("Resultado: cumple k-anonimato final mayor o igual a 5.")
    else:
        print("RESULTADO: ALERTA, no se puede declarar cumplimiento k>=5.")

    try:
        ruta_protegida = RUTA_PROGRAMA / "deudas_protegidas.csv"
        df_final_protegido.to_csv(ruta_protegida, index=False)
        print("Archivo generado:", ruta_protegida.name)
    except Exception as error:
        print("Error al guardar deudas_protegidas.csv:", error)
        df_final_protegido = pd.DataFrame()

    registrar("Proteccion y anonimizacion", len(df_compartir) + filas_suprimidas, len(df_final_protegido))

    # PREGUNTA 3 - Cifrado e integridad
    print("\nPREGUNTA 3 - CIFRADO Y PRUEBAS")

    clave_fernet_env = os.environ.get("CLAVE_FERNET")
    clave_fernet = None
    if clave_fernet_env:
        try:
            clave_fernet = clave_fernet_env.encode("utf-8")
            Fernet(clave_fernet)
            print("Fernet: usando CLAVE_FERNET del entorno.")
        except Exception:
            print("ADVERTENCIA: CLAVE_FERNET existe pero no es valida.")
            clave_fernet = Fernet.generate_key()
            print("Se uso una clave temporal para pruebas.")
    else:
        clave_fernet = Fernet.generate_key()
        print("ADVERTENCIA: CLAVE_FERNET no existe.")
        print("Se uso una clave temporal; no es adecuada para produccion.")

    ruta_protegida = RUTA_PROGRAMA / "deudas_protegidas.csv"
    ruta_cifrada = RUTA_PROGRAMA / "deudas_protegidas.csv.enc"
    contenido_cifrado = None

    try:
        if not ruta_protegida.exists():
            raise FileNotFoundError("deudas_protegidas.csv no fue generado")

        contenido_original = ruta_protegida.read_bytes()
        cifrador = Fernet(clave_fernet)
        contenido_cifrado = cifrador.encrypt(contenido_original)
        ruta_cifrada.write_bytes(contenido_cifrado)

        HASH_ARCHIVO_CIFRADO_ESPERADO = hashlib.sha256(contenido_cifrado).hexdigest()
        print("Archivo cifrado generado:", ruta_cifrada.name)
        print("SHA-256 del archivo cifrado:", HASH_ARCHIVO_CIFRADO_ESPERADO)
    except Exception as error:
        print("Error de cifrado:", error)

    if contenido_cifrado is not None:
        contenido_descifrado, mensaje = recibir(contenido_cifrado, clave_fernet)
        print("1.", mensaje)
        if contenido_descifrado is not None:
            print("   Descifrado correcto:", len(contenido_descifrado), "bytes.")

        contenido_alterado = bytearray(contenido_cifrado)
        contenido_alterado[len(contenido_alterado) // 2] ^= 1
        contenido_descifrado, mensaje = recibir(bytes(contenido_alterado), clave_fernet)
        print("2.", mensaje)

        clave_incorrecta = Fernet.generate_key()
        contenido_descifrado, mensaje = recibir(contenido_cifrado, clave_incorrecta)
        print("3.", mensaje)
    else:
        print("Las pruebas de recepcion no pueden ejecutarse sin el archivo cifrado.")

    registrar(
        "Cifrado",
        len(ruta_protegida.read_bytes()) if ruta_protegida.exists() else 0,
        len(contenido_cifrado) if contenido_cifrado is not None else 0,
    )

    # ==================================================
    # PARTE III - AUDITORIA Y CUMPLIMIENTO
    # ==================================================
    print("\nPARTE III - AUDITORIA Y CUMPLIMIENTO")
    print("PREGUNTA 1 - VERIFICACION DE AUDITORIA")

    cadena_integra, cantidad_registros = verificar_cadena(RUTA_AUDITORIA)
    print("Cantidad de registros:", cantidad_registros)
    print("Resultado:", "INTEGRA" if cadena_integra else "ALTERADA")

    ruta_copia = RUTA_PROGRAMA / "auditoria_alterada_prueba.jsonl"
    try:
        if RUTA_AUDITORIA.exists():
            contenido_auditoria = RUTA_AUDITORIA.read_text(encoding="utf-8")
            lineas = [linea for linea in contenido_auditoria.splitlines() if linea.strip()]
            if lineas:
                registro = json.loads(lineas[-1])
                registro["accion"] = "ACCION ALTERADA PARA PRUEBA"
                lineas[-1] = json.dumps(registro, ensure_ascii=False, sort_keys=True)
            ruta_copia.write_text("\n".join(lineas) + ("\n" if lineas else ""), encoding="utf-8")

            copia_integra, cantidad_copia = verificar_cadena(ruta_copia)
            print("Cadena original:", "integra" if cadena_integra else "alterada")
            print("Cadena simulada alterada:", "detectada" if not copia_integra else "NO DETECTADA")

        try:
            ruta_copia.unlink()
        except OSError:
            pass
    except Exception as error:
        print("No fue posible simular la alteracion de auditoria:", error)

    print("\nPREGUNTA 2 - TABLA DE CONTROLES")

    # Los indices de la salida protegida se conservan desde df_limpio.
    # Se consultan las columnas originales para verificar los controles antes
    # de suprimirlas del archivo que se comparte.
    indices_compartidos = df_final_protegido.index
    consentimiento_verificable = "consentimiento" in df_limpio.columns
    if consentimiento_verificable:
        consentimiento_compartido = (
            df_limpio.loc[indices_compartidos, "consentimiento"]
            .astype("string")
            .str.strip()
        )
        cantidad_sin_consentimiento_compartidos = int(
            (~consentimiento_compartido.isin(["Sí"])).sum()
        )
    else:
        cantidad_sin_consentimiento_compartidos = None

    retencion_verificable = "fecha_origen_deuda" in df_limpio.columns
    if retencion_verificable:
        fechas_compartidas = pd.to_datetime(
            df_limpio.loc[indices_compartidos, "fecha_origen_deuda"],
            errors="coerce",
            format="mixed",
        )
        dentro_retencion_compartidos = (
            fechas_compartidas.notna()
            & (fechas_compartidas >= pd.Timestamp(limite_retencion))
            & (fechas_compartidas <= pd.Timestamp(fecha_referencia))
        )
        fuera_retencion_compartidos = int((~dentro_retencion_compartidos).sum())
    else:
        fuera_retencion_compartidos = None

    controles = [
        ("Clientes sin consentimiento compartidos",
         cantidad_sin_consentimiento_compartidos
         if cantidad_sin_consentimiento_compartidos is not None
         else "No verificable",
         "CUMPLE" if cantidad_sin_consentimiento_compartidos == 0 else "ALERTA"),
        ("Registros fuera de retencion compartidos",
         fuera_retencion_compartidos
         if fuera_retencion_compartidos is not None
         else "No verificable",
         "CUMPLE" if fuera_retencion_compartidos == 0 else "ALERTA"),
        ("k-anonimato final",
         k_final,
         "CUMPLE" if k_final >= 5 else "ALERTA"),
        ("Cadena de auditoria",
         "integra" if cadena_integra else "alterada",
         "CUMPLE" if cadena_integra else "ALERTA"),
    ]

    tabla_controles = pd.DataFrame(
        controles,
        columns=["Nombre", "Resultado calculado", "Estado"],
    )
    print(tabla_controles.to_string(index=False))

    print("\nRESULTADO FINAL")
    estados = [estado for _, _, estado in controles]
    if all(estado == "CUMPLE" for estado in estados):
        print("Todos los controles automaticos cumplen.")
    else:
        print("Uno o mas controles muestran ALERTA; revise los resultados anteriores.")


if __name__ == "__main__":
    main()
