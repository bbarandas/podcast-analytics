# Podcast Analytics — Design Spec

**Data:** 2026-04-15
**Projeto:** Podcast de futebol com análise tática, técnica e biomecânica
**Autor:** Bruno Barandas

---

## Contexto

Podcast criado por Bruno e 3 educadores físicos obcecados por futebol. Diferencial: análise biomecânica dos lances — por que o jogador perdeu aquele gol? como aquele cruzamento foi executado? Os 3 co-hosts cuidam do conteúdo esportivo; Bruno é produtor, roteirista e responsável por toda a infraestrutura técnica (usando Claude como ferramenta).

O sistema de analytics serve para **preparar episódios** — gerar visualizações internas para embasar as discussões e exportar imagens para redes sociais do podcast.

---

## Decisões de Design

- **Dados:** exclusivamente gratuitos/públicos
- **Cobertura:** qualquer liga/competição dependendo do episódio
- **Stack:** Python + Streamlit (toolkit modular + dashboard interativo)
- **Usuário único:** Bruno, rodando localmente

---

## Arquitetura

```
podcast-analytics/
├── data/                  # dados baixados/cacheados localmente
├── modules/
│   ├── data_sources/      # conectores: StatsBomb, Understat, FBref
│   ├── positioning/       # mapas de calor, passes, pressão defensiva
│   ├── video/             # extração de frames, detecção de jogadores/bola
│   └── biomechanics/      # estimativa de pose, cálculo de ângulos articulares
├── visualizations/        # utilitários compartilhados, exportação PNG
├── exports/               # imagens geradas para redes sociais
├── dashboard/             # app Streamlit (app.py + pages/)
└── notebooks/             # análises exploratórias e rascunhos
```

---

## Fontes de Dados

| Fonte | Biblioteca | Dados disponíveis | Cobertura |
|---|---|---|---|
| StatsBomb Open Data | `statsbombpy` | Eventos (passes, chutes, dribles) + dados 360 de posicionamento | Competições europeias selecionadas (UCL, La Liga, etc.) |
| Understat | `understat` (async) | xG por chute, mapa de chutes | Top 5 ligas europeias |
| FBref | `soccerdata` | Stats de jogadores e partidas | Ampla cobertura global |
| Vídeo | `yt-dlp` + `opencv-python` | Qualquer partida disponível no YouTube | Universal |

**Limitação conhecida:** dados estruturados de futebol brasileiro são escassos nas fontes gratuitas. Para episódios sobre o Brasileirão, a análise se apoia mais em vídeo + biomecânica do que em dados de evento.

---

## Módulos de Análise

### 1. Posicionamento (`modules/positioning/`)

Consome dados StatsBomb 360 e gera visualizações sobre campo renderizado com `mplsoccer`:

- **Mapa de calor** — densidade de presença de um jogador ao longo de uma partida ou período
- **Mapa de passes** — linhas origem → destino com espessura proporcional à frequência
- **Mapa de pressão defensiva** — regiões de maior intensidade de pressão do time
- **Posicionamento médio** — posição média de cada jogador em campo

Filtros disponíveis: jogador, período (1º tempo / 2º tempo / minuto específico), tipo de evento.

### 2. Vídeo (`modules/video/`)

Recebe MP4 (local ou baixado via `yt-dlp`) e processa um lance de interesse:

- **Extração de frames** — seleciona intervalo por minuto:segundo
- **Detecção e rastreamento** — YOLO para identificar jogadores e bola quadro a quadro
- **Output:** vídeo anotado com bounding boxes + sequência de frames PNG para uso na análise biomecânica

### 3. Biomecânica (`modules/biomechanics/`)

Aplica MediaPipe Pose nos frames extraídos para um jogador específico:

- **Skeleton overlay** — sobreposição do esqueleto articular nos frames
- **Ângulos articulares** — joelho, quadril, tornozelo calculados quadro a quadro
- **Gráfico temporal** — evolução dos ângulos durante o movimento (chute, cruzamento, sprint)
- **Anotações manuais** — marcação de momentos-chave: "momento do contato", "ponto de máxima extensão", etc.

Este módulo é o diferencial do podcast: explica *por que* um movimento foi bem ou mal executado com embasamento biomecânico.

### 4. Visualizações Compartilhadas (`visualizations/`)

- Wrapper sobre `mplsoccer` e `plotly` com identidade visual do podcast (cores, logo)
- Exportação em dois formatos:
  - **Interativo:** renderizado no Streamlit com `plotly`
  - **Estático PNG:** salvo em `/exports/` com layout para Instagram/LinkedIn (1080×1080 ou 1080×1350)

---

## Dashboard Streamlit

**Sidebar:**
- Seletor: competição → partida → jogador
- Botão "Carregar dados" (baixa e cacheia em `/data/`)

**Aba Posicionamento:**
- Seleção de visualização (calor / passes / pressão / posição média)
- Filtros de período
- Botão "Exportar PNG"

**Aba Vídeo:**
- Input: upload de MP4 ou URL do YouTube
- Seletor de intervalo de frames
- Preview do vídeo anotado

**Aba Biomecânica:**
- Seleção do jogador detectado
- Frames com skeleton overlay
- Gráfico de ângulos articulares
- Campo de anotações por momento-chave
- Botão "Exportar PNG"

---

## Fluxo Típico de Uso (preparar um episódio)

1. Escolhem o lance do episódio
2. **Aba Vídeo** → fazem upload ou colam URL → isolam os frames do lance → YOLO detecta jogadores
3. **Aba Biomecânica** → selecionam o jogador → analisam ângulos → anotam momentos-chave → exportam PNG
4. **Aba Posicionamento** → selecionam partida no StatsBomb → contextualizam taticamente → exportam PNG
5. Usam as imagens no roteiro do episódio e nas redes sociais

---

## Dependências Principais

```
statsbombpy
soccerdata
understat
yt-dlp
opencv-python
mediapipe
ultralytics        # YOLO
mplsoccer
matplotlib
plotly
streamlit
pandas
numpy
Pillow
```

---

## Limitações e Riscos

| Risco | Mitigação |
|---|---|
| Dados brasileiros escassos | Foco em vídeo + biomecânica para Brasileirão; dados de evento para ligas europeias |
| Detecção YOLO imprecisa em vídeos de baixa resolução | Permitir seleção manual de jogador por bounding box |
| MediaPipe perde pose em oclusões | Filtrar frames ruins e interpolar ângulos |
| YouTube remove vídeos ou bloqueia download | Suporte a upload de arquivo local como fallback |

---

## Fora de Escopo (v1)

- Backend/API — tudo roda localmente
- Autenticação ou multi-usuário
- Dados em tempo real (ao vivo)
- Análise de áudio do podcast
- Integração com plataformas de podcast (Spotify, Apple)
