# INVESTIGACIÓN — variant-confidence

Documento de investigación que sustenta el repositorio. Todo lo aquí escrito
está respaldado por evidencia verificada en el propio repo (SDD.md, CHANGELOG.md,
commits) o por llamadas reales a fuentes primarias (fechas de verificación
indicadas). No se incluyen afirmaciones sin soporte.

Autor: Pedro Sordo Martínez (amurlaniakea@gmail.com)
Fecha de la investigación base: 2026-07-18
Licencia del software: AGPL-3.0-or-later

---

## 1. El problema que motiva el proyecto

Los predictores de efecto de variante (AlphaMissense, ESM-1v, EVE) entregan un
score de patogenicidad, pero para uso clínico o de investigación lo crítico no es
solo el orden del score, sino **cuánto se puede confiar en el número**.

Evidencia del dominio (baseline real verificado, AnnotateMissense — arXiv
2605.24520, del PDF primario):

- MCC = 0.7613 en validación temporal ClinVar.
- Accuracy = 0.8798 (88%).
- F1 = 0.8750 en validación temporal ClinVar.

La brecha no es precisión (el modelo base ya la aporta), sino **calibración**:
un score crudo de 0.9 no significa necesariamente 90% de probabilidad. Actuar
sobre un score sin calibrar es un riesgo. El valor añadido de este proyecto es
**reducir el ECE (Expected Calibration Error)**, no mejorar la precisión.

Por eso el proyecto NO entrena un foundation model: reutiliza predictores ya
entrenados (AlphaMissense / ESM-1v / EVE) y les añade una capa de calibración
auditable, CPU-only (C2 de la Constitución).

---

## 2. Accesibilidad de las fuentes de datos (verificada por llamada real)

Comprobado 2026-07-18 mediante llamadas HTTP reales:

- ClinVar E-utilities: HTTP 200, sin token. ✓
- dbNSFP: HTTP 200 (descarga). ✓
- Ensembl VEP REST: HTTP 200. ✓
- AlphaMissense: repo `github.com/google-deepmind/alphamissense` HTTP 200. ✓
- gnomAD GraphQL: endpoint vivo PERO 400 en verificación → se marca como
  "verificar query válida en implementación" (AC6). NO se asume accesible por
  defecto.

---

## 3. Fugas silenciosas detectadas (el aporte real de la auditoría)

Esta es la parte sustantiva de la investigación: durante el ciclo de auditoría
independiente se cazaron **cuatro fugas que no fallaban solas** — los números
parecían plausibles mientras medían sobre el subconjunto equivocado. Catálogo
(AC11 de la SDD):

### 3.1 Cache commiteado / test dependiente de red
El test corrido en clone limpio fallaba con 403 porque dependía de una caché no
versionada.
**Fix:** fixture offline versionado en `tests/fixtures/`, `.gitignore` excluye
`data/cache/`, y `git-filter-repo` para purgar historial.

### 3.2 Generador sintético desconectado del label (ECE degenerado)
El score sintético ignoraba `y` (`_ = labels`), con AUC 0.51, y el ECE
"bajaba a 0" por colapso a la tasa base (91.5% patogénico). Era un resultado
degenerado, no una calibración real.
**Fix:** `true_p` derivado del label real + ruido; el test exige AUC > 0.7 y AUC
preservado tras calibrar. (Commit `35ac98d`.)

### 3.3 Fit/eval sin separar en calibración
isotonic ajustaba y predecía sobre el mismo array → ECE = 0.0000 engañoso.
**Fix:** `calibrate_*` reciben `fit_idx`/`eval_idx` explícitos; `_check_split`
revienta ante solapamiento; el test afirma ECE isotónico > 1e-4 sobre el holdout.
(Commit `59a7645`.)

### 3.4 Índices desalineados (el más peligroso)
`temporal_gene_isolated_split` hacía `reset_index(drop=True)`, así que
`split.test.index` era `[0,1,..]` posiciones del subconjunto, no del df original.
El pipeline indexaba scores/y/genes con esos índices → apuntaba a filas
arbitrarias. 238/1088 genes "de eval" eran en realidad de train (anulaba AC3).
**Fix:** `SplitResult` lleva `test_idx`/`train_idx` como posiciones ORIGINALES
(calculadas antes de cualquier reset). Test de guarda: genes de test disjuntos de
train vía `test_idx`, y `test_idx != rango trivial [0..n)`.
(Commit `96faa9c`.)

**Regla general extraída:** cualquier métrica de calidad (ECE, cobertura, AUC)
debe medirse sobre el conjunto que el código DICE estar usando, verificado por un
test que reconstruye el conjunto de forma independiente. Si el número "se ve
bien" pero el split es incorrecto, el bug es silencioso.

---

## 4. Notas de diseño no obvias (hallazgos arquitectónicos)

### 4.1 Mondrian-por-gen ≡ split conformal (consecuencia de AC3)
Con AC3 (gene-isolation) ningún gen de test aparece en train, y calib ⊆ train,
así que ningún gen de eval tiene cuantiles propios en fit → el fallback global se
activa para el 100% de los casos. Mondrian es entonces IDÉNTICO a split conformal
por construcción. El CLI lo declara ("mondrian fallback rate = 100%"), no lo
oculta. No es fuga, es consecuencia arquitectónica de AC3. (AC10.)

### 4.2 El patrón de missing es ESTRUCTURAL, no aleatorio
Medido contra el TSV real de AlphaMissense (2026-07-18):
- AlphaMissense cubre 19.118 proteínas uniprot. Cuando cubre una proteína, la
  cubre COMPLETAMENTE: solo 1 de 19.118 proteínas tiene <5 variantes.
- El missing NO es por variante individual suelta — es por PROTEÍNA/ISOFORMA
  ENTERA ausente.
- Implicación: al excluir variantes sin score (`on_missing='degrade'`), si las
  proteínas sin cobertura correlacionan con algo (genes raros, regiones no
  codificantes), el holdout puede sesgarse sistemáticamente. No es un bug (lo
  detecta `test_missing_pattern.py`), pero debe reportarse al interpretar la
  calibración en datos reales. (AC13b / nota de audit #2.)

---

## 5. Ambigüedad legal de AlphaMissense (hallazgo de investigación)

Fuentes primarias OFICIALES se contradicen (verificado 2026-07-18):

- README de `google-deepmind/alphamissense`: "CC BY 4.0".
- Header real del TSV + Ensembl VEP + EBI: "CC BY-NC-SA 4.0".

No es resoluble desde aquí (no soy abogado; contacto para confirmación por
escrito: `alphamissense@google.com`). Postura prudente adoptada: tratar el dato
como **RESTRINGIDO (no comercial)** hasta aclaración. Consecuencias en el repo:

- El TSV de scores (~71M filas / 613MB) NUNCA se commitea. El repo queda 100%
  AGPL-3.0 limpio.
- El usuario lo descarga localmente bajo su responsabilidad; el README documenta
  la URL y la advertencia de licencia.
- Los tests usan un fixture ESTRUCTURAL sintético (columnas idénticas al TSV
  real, valores inventados) — el repo no contiene ni un byte de AlphaMissense
  real. La validación contra el TSV real se hace en /tmp y se reporta, no se
  commite.

(AC13 de la SDD.)

---

## 6. Criterios de aceptación derivados de la investigación

La base de criterios (AC1–AC13b) se documenta íntegra en `SDD.md`. Resumen de lo
que la investigación obligó a exigir:

- AC1: método dual seleccionable (Platt/isotonic + conformal 1−α), no hardcoded.
- AC2: ECE con 10 bins + adaptive, delta antes/después, bootstrap CI (1000),
  marcar bins < 25 muestras como baja fiabilidad.
- AC3: split temporal por fecha ClinVar + aislamiento por gen, con test que
  FALLA ante overlap de gen.
- AC4: robustez ante scores ausentes — NUNCA como 0 (anti-bug #5).
- AC7: salida no engañosa — nunca score calibrado solo.
- AC9: holdout de evaluación temporal aislado mínimo ≥ 500 (parametrizable),
  para evitar "ECE ≤ 0.05" por holdout minúsculo.
- AC10: separación fit/eval obligatoria en calibración (anti-fuga de la clase de
  AC3 aplicada a la fase de calibración).
- AC12/AC13: scores ausentes nunca como 0; AlphaMissense nunca commiteado.

---

## 7. Verificación final (clone fresco, sin red)

- `ruff check .` → All checks passed!
- `pytest tests/` → **28 passed in 8.90s** (fixture offline versionado).

El release v0.1.0 y este documento se basan en ese código verificado, no en
resúmenes de memoria.

---

*Licencia del software: AGPL-3.0-or-later — Pedro Sordo Martínez
(amurlaniakea@gmail.com), 2026.*
