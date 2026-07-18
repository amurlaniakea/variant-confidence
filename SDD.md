# variant-confidence — SDD (Specification-Driven Development)

**Estado:** Spec anclada (Fase: redacción SDD → aprobación Sil → implementación)
**Fecha:** 2026-07-18
**Licencia:** AGPL-3.0-or-later (año 2026)
**Autor:** Pedro Sordo Martínez (amurlaniakea@gmail.com)
**Dominio:** Biotech / variant-effect pathogenicity — capa de confianza calibrada
**Patrón reutilizado:** igual que `ssb-validate` / `dock-confidence` / `cryoval`
  (capa de validación CPU-only, AGPL-3.0, sobre modelo ajeno ya entrenado)

================================================================
CONSTITUCIÓN (principios innegociables)
================================================================
C1. AGPL-3.0-or-later, año 2026, autor "Pedro Sordo Martínez
    <amurlaniakea@gmail.com>". README profesional (H1, badge licencia,
    features, install, license-link); sin banners ni URLs internas.
C2. CPU-only. No entrenar foundation model; reusar AlphaMissense/ESM-1v/EVE.
C3. Output NO engañoso (AC7): nunca entregar score calibrado solo; siempre
    intervalo/ECE + método + umbral.
C4. Auditoría (tu pipeline): Hermes implementa → Sil aprueba en clone fresco
    → Claude re-verifica. Push a rama dedicada, diff texto plano.
C5. Cierre Python (regla tuya): antes de declarar listo, SIEMPRE
    `ruff check .` limpio + tests afectados con output crudo.

================================================================
SPEC (qué hace el sistema)
================================================================
S1. ENTRADA: variantes missense (chr/pos/ref/alt o rsID) + scores crudos de
    uno o varios modelos base (AlphaMissense, ESM-1v, EVE).
S2. PROCESO: capa de calibración intercambiable (AC1) sobre los scores →
    emite probabilidad calibrada y/o intervalo conformal.
S3. MÉTRICA: ECE medido antes/después de calibrar (AC2); cobertura nominal
    vs empírica si se usa conformal (AC2 alt).
S4. SALIDA: por variante → { score_calibrado, intervalo_o_ECE, método_usado,
    umbral, delta_ECE_vs_base }. Nunca score solo (AC7).
S5. DATOS primarios (accesibilidad VERIFICADA por llamada real 2026-07-18):
    - ClinVar E-utilities: HTTP 200, sin token ✓
    - dbNSFP: HTTP 200 (descarga) ✓
    - Ensembl VEP REST: HTTP 200 ✓
    - AlphaMissense: repo github.com/google-deepmind/alphamissense HTTP 200 ✓
    - gnomAD GraphQL: endpoint vivo PERO 400 en verificación → AC6 lo marca
      como "verificar query válida en implementación", NO asumir accesible.
S6. SPLIT (anti-leakage, AC3): temporal por fecha de release ClinVar + 
    aislamiento por familia proteica/gen (mismo gen no en train y test).
S7. BASELINE real (AC5): AnnotateMissense (arXiv 2605.24520, del PDF primario)
    sin calibrar → MCC = 0.7613, accuracy = 0.8798 (88%), F1 = 0.8750 en
    validación temporal ClinVar. Valor añadido del proyecto = reducir ECE,
    NO mejorar accuracy (eso ya lo hace el modelo base).

================================================================
PLAN (fases)
================================================================
P1. data-layer: fetch ClinVar (E-utilities) + AlphaMissense (repo google-deepmind)
    + dbNSFP; construir dataset con columnas {variant, gene, clinvar_date, label}.
P2. split-module: temporal split por clinvar_date + gene-isolation; test
    unitario que FALLA si hay overlap de gen entre splits (AC3).
P3. calibration-module: (a) Platt/isotonic sobre holdout de calibración;
    (b) split conformal / Mondrian conformal estratificado por gen.
P4. metrics: ECE (10 equal-width bins + adaptive binning referencia);
    cobertura nominal vs empírica para conformal.
P5. cli/report: score + interval + método + ECE before/after + comparación
    con baseline AnnotateMissense.
P6. tests: fixture determinista (semilla fija), ECE real medido, leakage test,
    robustez 1-solo-score (AC4).

================================================================
TASKS (granulares)
================================================================
T1. Scaffold repo /home/sil/variant-confidence/ (git init; .gitignore excluye
   .venv/ .hermes/ credentials/). pyproject + entorno venv (NUNCA
   --break-system-packages).
T2. data/loader.py: fetchers ClinVar+AlphaMissense+dbNSFP con cache local.
   AC6: gnomAD tras verificar query válida; si falla, documentar y omitir.
T3. data/dataset.py: esquema con clinvar_date + gene; parser a DataFrame.
T4. split/temporal.py: split por fecha + gene-isolation. AC3.
T5. test_split_overlap.py: test que falla si overlap de gen (CI obligatorio). AC3.
T6. calib/platt.py + calib/isotonic.py: ajuste sobre holdout separado. AC1a.
T7. calib/conformal.py: split/Mondrian conformal por gen. AC1b.
T8. metrics/ece.py: ECE 10 bins + adaptive; delta before/after; bootstrap CI
   (1000) y n por bin (AC2, AC9).
T9. test_ece_determinismo.py: fixture 200 casos, semilla fija, ECE reproducible.
T9b. test_holdout_min.py: FALLA si holdout temporal aislado < 500 (AC9); marca
   ECE "bajo fiabilidad" si algún bin < 25 muestras.
T10. cli.py: genera reporte con score+intervalo+método+umbral+delta+AC7;
   incluye sección de licencias de datos (AC8).
T11. tests robustez AC4: solo-AlphaMissense y solo-ESM-1v funcionan y emiten
   aviso explícito (no fallan silenciosamente) cuando hay un único score. AC4.
T11b. README + LICENSE (AGPL-3.0-or-later 2026) con sección "Data Licenses"
   (AlphaMissense CC BY 4.0 verificado, ClinVar/dbNSFP/VEP, gnomAD pendiente)
   + ruff baseline limpio (C5).

================================================================
ACCEPTANCE CRITERIA (base de Sil — calibración)
================================================================
AC1 — Método dual seleccionable
  (a) Calibración de probabilidad: Platt o isotonic, sobre holdout separado
      del entrenamiento del modelo base.
  (b) Conformal prediction: intervalos cobertura 1−α (split o Mondrian por gen).
  Ambos por config, no hardcoded; output indica método usado.

AC2 — Métrica ECE
  ECE con 10 bins equal-width (mínimo) + adaptive binning referencia.
  Umbral aceptación: ECE ≤ 0.05 en holdout temporal aislado (ajustable,
  documentar porqué). Reportar ECE ANTES y DESPUÉS de calibrar (delta auditable).
  Si conformal: cobertura empírica vs nominal (1−α), tolerancia ±2%.

AC3 — Guard anti-leakage (split temporal)
  Split por fecha de release ClinVar, NO muestreo aleatorio.
  Aislamiento: misma familia proteica/gen NO en train y test simultáneamente.
  Test unitario OBLIGATORIO que falla si hay overlap de gen (parte de CI).

AC4 — Robustez frente a scores base
  Funciona con AlphaMissense y ESM-1v independientes (EVE opcional).
  Declarar explícitamente qué pasa si solo hay un score (no fallar silenciosamente).

AC5 — Validación vs paper de referencia
  ECE medido reportado junto a baseline AnnotateMissense (MCC 0.7613 /
  accuracy 0.8798 validación temporal) sin calibrar. Valor añadido = reducción
  de ECE, NO mejora de accuracy.

AC6 — Fuentes de datos (estado real verificado 2026-07-18)
  ClinVar (E-utilities, sin token) y dbNSFP confirmadas accesibles.
  gnomAD: query GraphQL válida verificar en implementación (NO por defecto
  accesible). AlphaMissense: repo google-deepmind/alphamissense, NO dataset
  gated de HuggingFace.

AC7 — Salida no engañosa
  Todo output incluye intervalo/ECE junto al score, nunca score calibrado solo,
  para evitar uso como confianza absoluta sin contexto de método/umbral.

AC8 — Licencias de DATOS (separadas de la licencia del SOFTWARE)
  El SOFTWARE es AGPL-3.0-or-later 2026. Los DATOS de entrada tienen sus
  propias licencias, que el usuario final es responsable de cumplir; el
  pipeline NO redistribuye datos bajo licencia distinta a la suya.
  - **AlphaMissense predictions**: verificado en repo google-deepmind/
    alphamissense (README, 2026-07-18) → **CC BY 4.0** (atribución requerida,
    uso comercial PERMITIDO). No es CC BY-NC-SA. El README debe citar
    DeepMind + la publicación (Science 2023, adg7492) y el CC BY 4.0.
  - **ClinVar / dbNSFP / Ensembl VEP**: sujetos a sus términos NCBI/EMBL-EBI
    (uso libre con atribución; verificar en impl, no asumir comercial sin
    revisar).
  - **gnomAD**: verificar query + términos en impl (AC6).
  El README del proyecto DECLARARÁ este desajuste explícitamente (igual que
  ssb-validate con la API key): código AGPL, datos bajo sus licencias.

AC9 — Tamaño mínimo del holdout de calibración / fiabilidad del ECE
  ECE con pocas muestras por bin es ruidoso. El holdout de calibración y el
  holdout de evaluación temporal deben cumplir un mínimo:
  - Mínimo n ≥ 500 variantes en el holdout de evaluación temporal AISLADO
    (post gene-isolation); si tras el split queda < 500, el pipeline debe
    FALLAR con aviso explícito (no emitir ECE no fiable). El umbral 500 es
    PARAMETRIZABLE (--min-holdout, default 500), NO hardcoded: según cuántas
    variantes queden tras gene-isolation en ClinVar real puede requerir
    ajuste empírico; el pipeline lo REPORTA, no lo tiene clavado a fuego.
  - Reportar intervalo de confianza del ECE (bootstrap, 1000 remuestreos) o
    al menos n por bin; si algún bin queda con < 25 muestras, marcar el ECE
    como "bajo fiabilidad" en el reporte.
  - Esto previene el falso "ECE ≤ 0.05" por holdout minúsculo.

AC10 — Separación fit/eval OBLIGATORIA en calibración (anti-fuga)
  Toda calibración (Platt, isotonic, conformal) debe ajustarse sobre un
  holdout de CALIBRACIÓN y evaluarse (ECE, cobertura) sobre un holdout de
  EVALUACIÓN SEPARADO. Ajustar y predecir sobre el mismo array permite que
  métodos no-paramétricos (isotonic, conformal) memoricen la curva de esos
  puntos exactos y reporten ECE=0.0000 engañoso. Es la MISMA clase de fuga
  que AC3 (gene-isolation) aplicada a la fase de calibración.
  - Las funciones `calibrate_*` reciben `fit_idx`/`eval_idx` explícitos;
    nunca deciden el split (el pipeline lo impone, igual que AC3).
  - `_check_split` falla si fit/eval se solapan.
  - El test de ECE mide SOBRE el holdout de evaluación; isotonic honesto
    con 6k variantes ≈ 0.009 (no 0.0000). Si un calibrador reporta ECE=0
    sobre la muestra de fit, es fuga, no generalización.
  - Conformal: los cuantiles se calculan SOLO sobre fit_idx; los intervalos
    se devuelven para eval_idx. Evaluar cobertura sobre fit_idx infla a
    1−α por construcción (fuga).
  - NOTA Mondrian-por-gen: con AC3 (gene-isolation) NINGÚN gen de test aparece
    en train, y calib ⊆ train, así que ningún gen de eval tiene cuantiles
    propios en fit → el fallback global se activa para el 100% de los casos.
    Mondrian es entonces IDÉNTICO a split conformal por construcción. El CLI
    lo declara ("mondrian fallback rate = 100%"), no lo oculta. No es fuga,
    es una consecuencia arquitectónica de AC3.

AC11 — Catálogo de fugas silenciosas (lecciones de la sesión de audit)
  Esta sesión cazó CUATRO fugas que NO fallaban: los números parecían
  plausibles mientras medían sobre el subconjunto equivocado. Catálogo para
  no repetirlas en trabajo futuro:
  1. Cache commitido / test dependiente de red: el test corrido en clone
     limpio fallaba por 403 porque dependía de cache no versionado. Fix:
     fixture offline versionado en tests/fixtures/, .gitignore excluye
     data/cache/, y git-filter-repo para purgar historial.
  2. Generador sintético desconectado del label: el score ignoraba `y`
     (`_ = labels`), AUC 0.51, y el ECE "bajaba a 0" por colapso a la tasa
     base. Fix: true_p derivado del label real + ruido; el test exige
     AUC>0.7 y AUC preservado tras calibrar.
  3. Fit/eval sin separar en calibración: isotonic ajustaba y predecía sobre
     el mismo array → ECE=0.0000 engañoso. Fix: calibrate_* reciben
     fit_idx/eval_idx; _check_split revienta ante solapamiento; el test
     afirma ECE isotónico > 1e-4 sobre el holdout.
  4. Índices desalineados (el más peligroso): temporal_gene_isolated_split
     hacía reset_index(drop=True), así que split.test.index era [0,1,..]
     posiciones del subconjunto, no del df original. El pipeline indexaba
     scores/y/genes con esos índices → apuntaba a filas arbitrarias, no al
     holdout temporal; 238/1088 genes "de eval" eran de train (anulaba AC3).
     Fix: SplitResult lleva test_idx/train_idx como posiciones ORIGINALES
     (calculadas antes de cualquier reset). Test de guarda: genes de test
     disjuntos de train vía test_idx, y test_idx != rango trivial [0..n).
  REGLA GENERAL: cualquier métrica de calidad (ECE, cobertura, AUC) debe
  medirse sobre el conjunto que el código DICE estar usando, verificado por
  un test que reconstruye el conjunto de forma independiente. Si el número
  "se ve bien" pero el split es incorrecto, el bug es silencioso.

================================================================
ALCANCE HONESTO (sub-problemas, estilo SDD)
================================================================
Abordable (solo-dev, ~3-4 sem MVP):
  - Calibración Platt/isotonic (sklearn) ........ ✅ sobre holdout
  - ECE medido ................................... ✅ determinista
  - Split temporal + gene-isolation ............. ✅ ClinVar date + gene
  - Conformal Mondrian por gen ................... ✅ (extensión natural)
Fuera de alcance MVP (declarado):
  - Mejorar accuracy del modelo base → NO es el objetivo (AC5).
  - Concordancia fenotípica profunda → requiere fenotipo externo, post-MVP.
  - gnomAD si la query falla en impl → documentado, omitido.

================================================================
GOBIERNO / ENTRAGA
================================================================
Rama dedicada (nunca main). Diff texto plano. Sil aprueba en clone fresco;
Claude re-verifica (estructura tests, ECE real vs reportado, ausencia de
leakage train/test por gen — el error fácil en este dominio dado que ClinVar
se actualiza y contamina el split temporal sin darse cuenta).
