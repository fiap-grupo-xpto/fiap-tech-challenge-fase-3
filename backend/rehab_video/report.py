from __future__ import annotations

import numpy as np

from backend.rehab_video.features import KEYPOINT_COUNT


VISIBILITY_THRESHOLD = 0.2
MIN_VISIBLE_RATIO = 0.6
ASYMMETRY_LIMIT = 0.35
TRUNK_LEAN_LIMIT_DEG = 40.0

# COCO-17: 5/6 ombros, 7/8 cotovelos, 11/12 quadris, 13/14 joelhos, 15/16 tornozelos.
# (ponto proximal, vertice, ponto distal) por lado (esquerdo, direito).
EXERCISE_RULES = {
    1: {
        "joint": "ombro",
        "triplets": ((7, 5, 11), (8, 6, 12)),
        "min_rom": 60.0,
        "bilateral": False,
        "check_trunk": True,
    },
    2: {
        "joint": "ombro",
        "triplets": ((7, 5, 11), (8, 6, 12)),
        "min_rom": 40.0,
        "bilateral": True,
        "check_trunk": True,
    },
    3: {
        "joint": "cotovelo",
        "triplets": ((5, 7, 9), (6, 8, 10)),
        "min_rom": 40.0,
        "bilateral": True,
        "check_trunk": False,
    },
    4: {
        "joint": "quadril",
        "triplets": ((5, 11, 13), (6, 12, 14)),
        "min_rom": 20.0,
        "bilateral": False,
        "check_trunk": True,
    },
    5: {
        "joint": "joelho",
        "triplets": ((11, 13, 15), (12, 14, 16)),
        "min_rom": 40.0,
        "bilateral": False,
        "check_trunk": True,
    },
    6: {
        "joint": "joelho",
        "triplets": ((11, 13, 15), (12, 14, 16)),
        "min_rom": 40.0,
        "bilateral": True,
        "check_trunk": True,
    },
}
SIDES = ("esquerdo", "direito")


def _angle_series(poses: np.ndarray, triplet: tuple[int, int, int]) -> np.ndarray:
    a, b, c = triplet
    valid = (poses[:, [a, b, c], 2] >= VISIBILITY_THRESHOLD).all(axis=1)
    first = poses[:, a, :2] - poses[:, b, :2]
    second = poses[:, c, :2] - poses[:, b, :2]
    norm = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
    valid &= norm > 1e-6
    angles = np.full(len(poses), np.nan, dtype=np.float64)
    cosine = np.einsum("ij,ij->i", first, second)[valid] / norm[valid]
    angles[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angles


def _range_of_motion(angles: np.ndarray) -> float | None:
    finite = angles[np.isfinite(angles)]
    if len(finite) < 3:
        return None
    return float(finite.max() - finite.min())


def _trunk_lean(poses: np.ndarray) -> float | None:
    torso = poses[:, [5, 6, 11, 12], :]
    valid = (torso[:, :, 2] >= VISIBILITY_THRESHOLD).all(axis=1)
    if not valid.any():
        return None
    shoulders = poses[valid][:, [5, 6], :2].mean(axis=1)
    hips = poses[valid][:, [11, 12], :2].mean(axis=1)
    vector = shoulders - hips
    # Eixo y da imagem cresce para baixo; tronco ereto tem vetor ~ (0, -1).
    lean = np.degrees(np.arctan2(np.abs(vector[:, 0]), np.maximum(-vector[:, 1], 1e-6)))
    return float(np.mean(lean))


def build_deviation_report(poses: np.ndarray, exercise_id: int) -> dict:
    """Relatorio heuristico de desvios a partir da sequencia de poses normalizadas."""
    sequence = np.asarray(poses, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (KEYPOINT_COUNT, 3):
        raise ValueError(
            f"Expected pose sequence with shape (frames, 17, 3), received {sequence.shape}."
        )
    rules = EXERCISE_RULES.get(exercise_id)
    if rules is None:
        raise ValueError(f"Unsupported exercise_id: {exercise_id}")

    findings: list[dict] = []
    joint = rules["joint"]
    visible_ratio = float((sequence[:, :, 2] >= VISIBILITY_THRESHOLD).mean())
    roms = [
        _range_of_motion(_angle_series(sequence, triplet))
        for triplet in rules["triplets"]
    ]
    measured = [rom for rom in roms if rom is not None]
    metrics = {
        "visible_keypoint_ratio": round(visible_ratio, 3),
        f"amplitude_{joint}_esquerdo_graus": None if roms[0] is None else round(roms[0], 1),
        f"amplitude_{joint}_direito_graus": None if roms[1] is None else round(roms[1], 1),
        "inclinacao_tronco_graus": None,
    }

    if visible_ratio < MIN_VISIBLE_RATIO or not measured:
        findings.append(
            {
                "code": "dados_insuficientes",
                "severity": "info",
                "message": (
                    "Poucos pontos do corpo foram detectados com confianca; "
                    "a analise de desvios nao e confiavel neste trecho."
                ),
                "measured": round(visible_ratio, 3),
                "expected": MIN_VISIBLE_RATIO,
            }
        )
    else:
        best = max(measured)
        if best < rules["min_rom"]:
            findings.append(
                {
                    "code": "amplitude_reduzida",
                    "severity": "warning",
                    "message": (
                        f"Amplitude de movimento do {joint} abaixo do esperado "
                        f"({best:.0f} graus; referencia minima {rules['min_rom']:.0f})."
                    ),
                    "measured": round(best, 1),
                    "expected": rules["min_rom"],
                }
            )
        if rules["bilateral"] and len(measured) == 2 and max(measured) > 0:
            gap = abs(measured[0] - measured[1]) / max(measured)
            metrics["assimetria_relativa"] = round(float(gap), 3)
            if gap > ASYMMETRY_LIMIT:
                weaker = SIDES[0] if measured[0] < measured[1] else SIDES[1]
                findings.append(
                    {
                        "code": "assimetria",
                        "severity": "warning",
                        "message": (
                            f"Assimetria entre os lados ({gap:.0%} de diferenca na "
                            f"amplitude do {joint}); menor amplitude no lado {weaker}."
                        ),
                        "measured": round(float(gap), 3),
                        "expected": ASYMMETRY_LIMIT,
                    }
                )
        if rules["check_trunk"]:
            lean = _trunk_lean(sequence)
            if lean is not None:
                metrics["inclinacao_tronco_graus"] = round(lean, 1)
                if lean > TRUNK_LEAN_LIMIT_DEG:
                    findings.append(
                        {
                            "code": "inclinacao_tronco",
                            "severity": "warning",
                            "message": (
                                f"Inclinacao lateral excessiva do tronco "
                                f"({lean:.0f} graus; limite {TRUNK_LEAN_LIMIT_DEG:.0f})."
                            ),
                            "measured": round(lean, 1),
                            "expected": TRUNK_LEAN_LIMIT_DEG,
                        }
                    )

    deviations = [item for item in findings if item["severity"] == "warning"]
    if deviations:
        summary = f"{len(deviations)} desvio(s) detectado(s): " + "; ".join(
            item["message"] for item in deviations
        )
    elif findings:
        summary = findings[0]["message"]
    else:
        summary = "Nenhum desvio detectado pelas regras de amplitude, simetria e tronco."
    return {
        "summary": summary,
        "deviation_count": len(deviations),
        "findings": findings,
        "metrics": metrics,
        "method": (
            "Regras heuristicas sobre angulos articulares da pose (YOLOv8n-pose); "
            "limiares de referencia nao validados clinicamente."
        ),
    }
