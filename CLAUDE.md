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
Streamlit Cloud: https://podcast-analytics-4memhp7rcmupqdwasfrgph.streamlit.app

---

## ⚠️ PENDENTE — Retomar aqui

### Deploy Streamlit Cloud com erro libGL

**Problema:** `ImportError: libGL.so.1` ao carregar páginas de Vídeo e Biomecânica.

**Causa:** ultralytics instala opencv-python (full) que precisa de libGL. No Debian trixie do Streamlit Cloud, nenhum pacote apt resolve isso sem conflito de dependências.

**Último fix aplicado (commit e775d37):** removeu todos os `import cv2` do nosso código — usa PIL + ffmpeg + numpy. Ultralytics pode ainda importar cv2 internamente. **Ainda não confirmado se funcionou** — Bruno foi dormir antes de ver o resultado.

**Próximos passos:**
1. Abrir o Streamlit Cloud e verificar se o último deploy passou
2. Se ainda falhar → avaliar remover ultralytics do cloud e usar detecção alternativa
3. O diferencial real é a biomecânica (MediaPipe) — o YOLO é secundário

**O que já foi tentado e não funcionou:**
- packages.txt com libgl1-mesa-glx, libglib2.0-0, libgl1-mesa-dri, libgl1
- opencv-python-headless (ultralytics sobrescreve com full)
- runtime.txt para Python 3.11 (funcionou para mediapipe, não para libGL)
