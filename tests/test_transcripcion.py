from __future__ import annotations

import os
import tempfile

import pytest

from transcribir_carpeta import validar_srt


def _crear_srt(lines: list[str]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines))
        return f.name


class TestValidarSrt:
    def test_srt_valido(self) -> None:
        ruta = _crear_srt([
            "1",
            "00:00:01,000 --> 00:00:04,000",
            "Hola mundo",
            "",
            "2",
            "00:00:05,000 --> 00:00:08,000",
            "Segundo segmento",
            "",
        ])
        assert validar_srt(ruta)
        os.unlink(ruta)

    def test_srt_vacio(self) -> None:
        ruta = _crear_srt([])
        assert not validar_srt(ruta)
        os.unlink(ruta)

    def test_srt_sin_numeros(self) -> None:
        ruta = _crear_srt([
            "Hola",
            "00:00:01,000 --> 00:00:04,000",
            "texto",
        ])
        assert not validar_srt(ruta)
        os.unlink(ruta)

    def test_srt_sin_tiempo(self) -> None:
        ruta = _crear_srt([
            "1",
            "sin formato de tiempo",
            "texto",
        ])
        assert not validar_srt(ruta)
        os.unlink(ruta)

    def test_archivo_inexistente(self) -> None:
        assert not validar_srt("/ruta/inexistente/archivo.srt")


if __name__ == "__main__":
    pytest.main()
