# Podcast Analytics — Resenha Tática

## Contexto

Podcast de análise tática, técnica e biomecânica de futebol.
Time: Bruno (técnico/produtor) + 3 educadores físicos (conteúdo esportivo).
Diferencial: análise biomecânica de lances — por que aquele gol foi perdido, como aquele cruzamento foi executado.

## Stack

- Python 3.9.6 (local) / 3.11 (cloud)
- StatsBomb Open Data → posicionamento, passes, pressão
- yt-dlp (download YouTube), ffmpeg (frames), PIL (imagens)
- YOLOv8 ultralytics (detecção de jogadores)
- MediaPipe mp.solutions.pose (biomecânica — 6 articulações)
- Streamlit (dashboard)

## Comandos

```bash
podcast-dash          # sobe o dashboard local em localhost:8501
cc-podcast-analytics  # abre nova sessão Claude neste projeto
```

## GitHub

https://github.com/bbarandas/podcast-analytics (público)
Hugging Face Spaces: https://huggingface.co/spaces/bbarandas/podcast-analytics

---

## Status do Deploy

**Plataforma:** Hugging Face Spaces (Docker)
**URL:** https://huggingface.co/spaces/bbarandas/podcast-analytics
**Status:** ✅ Funcionando

**Decisões tomadas:**
- Migrado do Streamlit Cloud (libGL.so.1 irresolúvel no Debian trixie)
- ultralytics removido do requirements.txt — YOLO é secundário ao MediaPipe
- mediapipe fixado em 0.10.14 (mp.solutions removido em versões mais novas)
- Dockerfile usa libgl1 + libglib2.0-0 + ffmpeg no Debian trixie
