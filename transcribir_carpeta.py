from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import warnings
from collections.abc import Generator
from typing import NoReturn

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*symlinks.*")

from faster_whisper import WhisperModel  # type: ignore[import-untyped]

# --- INYECTAR FFMPEG EN EL PATH DEL SISTEMA ---
try:
    import imageio_ffmpeg  # type: ignore[import-untyped]

    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
except Exception:
    pass

EXTENSIONS: tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".m4a")

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("transcripcion")

# --- FLAG DE INTERRUPCIÓN ---
detener: bool = False


def manejar_signal(sig: int, frame: object) -> NoReturn | None:
    global detener
    if detener:
        log.warning(" Interrupción forzada. Saliendo...")
        sys.exit(1)
    detener = True
    log.warning(" Ctrl+C detectado. Finalizando transcripción actual... (presioná Ctrl+C de nuevo para forzar)")


signal.signal(signal.SIGINT, manejar_signal)


def validar_srt(ruta: str) -> bool:
    if not os.path.exists(ruta):
        return False
    with open(ruta, "r", encoding="utf-8") as f:
        contenido: str = f.read().strip()
    if not contenido:
        return False
    lineas: list[str] = contenido.split("\n")
    if not lineas[0].isdigit():
        return False
    if "-->" not in (lineas[1] if len(lineas) > 1 else ""):
        return False
    return True


def generar_srt(segmentos: Generator) -> tuple[list[str], int]:
    srt_lines: list[str] = []
    idx: int = 1
    total_segs: int = 0

    for seg in segmentos:
        if detener:
            raise KeyboardInterrupt()

        inicio_seg: float = seg.start
        fin_seg: float = seg.end
        texto: str = seg.text.strip()

        if not texto:
            continue

        inicio_srt: str = (
            f"{int(inicio_seg // 3600):02d}:{int((inicio_seg % 3600) // 60):02d}:{inicio_seg % 60:06.3f}".replace(
                ".", ","
            )
        )
        fin_srt: str = (
            f"{int(fin_seg // 3600):02d}:{int((fin_seg % 3600) // 60):02d}:{fin_seg % 60:06.3f}".replace(
                ".", ","
            )
        )

        srt_lines.append(f"{idx}")
        srt_lines.append(f"{inicio_srt} --> {fin_srt}")
        srt_lines.append(texto)
        srt_lines.append("")
        idx += 1
        total_segs += 1

    return srt_lines, total_segs


def procesar_video(
    archivo: str,
    carpeta_entrada: str,
    carpeta_salida: str,
    model: WhisperModel,
    language: str,
) -> None:
    global detener

    if detener:
        log.warning("Procesamiento interrumpido por el usuario.")
        return

    ruta_origen_video: str = os.path.join(carpeta_entrada, archivo)
    nombre_base: str = os.path.splitext(archivo)[0]

    ruta_srt_final: str = os.path.join(carpeta_salida, f"{nombre_base}.srt")
    ruta_destino_video: str = os.path.join(carpeta_salida, archivo)

    if os.path.exists(ruta_srt_final):
        log.info("Ya existe transcripción para '%s', saltando...", archivo)
        if os.path.exists(ruta_origen_video) and not os.path.exists(ruta_destino_video):
            os.replace(ruta_origen_video, ruta_destino_video)
        return

    log.info("Transcribiendo: %s ...", archivo)
    inicio: float = time.time()

    try:
        segmentos, info = model.transcribe(
            ruta_origen_video,
            language=language,
            beam_size=1,
            temperature=0,
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=False,
            no_speech_threshold=0.6,
            log_progress=True,
        )

        srt_lines, total_segs = generar_srt(segmentos)

        if not srt_lines:
            log.warning("No se generaron segmentos para '%s'.", archivo)
            return

        with open(ruta_srt_final, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_lines))

        if not validar_srt(ruta_srt_final):
            log.error("El SRT generado para '%s' parece inválido, eliminando...", archivo)
            os.remove(ruta_srt_final)
            return

        elapsed: float = time.time() - inicio
        log.info("Transcripción guardada: %s.srt (%d segmentos en %.0f s)", nombre_base, total_segs, elapsed)

        os.replace(ruta_origen_video, ruta_destino_video)
        log.info("Video movido a '%s'", carpeta_salida)

    except KeyboardInterrupt:
        log.warning("Transcripción de '%s' interrumpida.", archivo)
        raise
    except Exception as e:
        log.error("Error al procesar '%s': %s", archivo, e)


def crear_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcripción automatizada de videos con faster-whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        default="por_transcribir",
        help="Carpeta de entrada con videos (default: por_transcribir)",
    )
    parser.add_argument(
        "-o", "--output",
        default="transcripciones",
        help="Carpeta de salida para transcripciones (default: transcripciones)",
    )
    parser.add_argument(
        "-m", "--model",
        default="tiny",
        help="Modelo de Whisper (default: tiny)",
    )
    parser.add_argument(
        "-l", "--language",
        default="es",
        help="Idioma de transcripción (default: es)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo de inferencia (default: cpu)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Tipo de cómputo (default: int8)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Hilos de CPU (default: 4)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Workers del modelo (default: 1)",
    )
    return parser


def main() -> None:
    parser = crear_argparser()
    args: argparse.Namespace = parser.parse_args()

    os.makedirs(args.input, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    print("========================================")
    print(" Transcripción Automatizada BTL")
    print(f" Modelo: faster-whisper {args.model} {args.compute_type}")
    print("========================================")
    print()

    log.info("Iniciando modelo...")
    model: WhisperModel = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.threads,
        num_workers=args.workers,
    )

    try:
        archivos: list[str] = os.listdir(args.input)
    except Exception as e:
        log.error("No se pudo leer la carpeta '%s': %s", args.input, e)
        sys.exit(1)

    videos_a_procesar: list[str] = [f for f in archivos if f.lower().endswith(EXTENSIONS)]

    if not videos_a_procesar:
        log.info("No hay videos nuevos en '%s'.", args.input)
        sys.exit(0)

    log.info("Se encontraron %d video(s) para procesar.", len(videos_a_procesar))
    print()

    for archivo in videos_a_procesar:
        if detener:
            log.warning("Procesamiento interrumpido por el usuario.")
            break
        procesar_video(archivo, args.input, args.output, model, args.language)
        print("-" * 50)

    print()
    log.info("Procesamiento de lote finalizado.")


if __name__ == "__main__":
    main()
