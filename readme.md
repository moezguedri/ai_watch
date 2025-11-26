# AI Watch Pro — Radar IA Investisseur Long Terme

AI Watch Pro est un outil personnel d’analyse macro des géants de l’intelligence artificielle.
Il te donne une lecture **claire, synthétique et orientée investissement long terme** du cycle IA mondial.

L’objectif n’est pas de prédire les cours, mais de répondre à ces questions :

> Est-ce que l’IA est en phase d’expansion ?
>  
> Sommes-nous en surchauffe ?
>  
> Faut-il juste tenir… ou au contraire étreindre le cycle ?

---

## ✅ Ce que fait l’outil

### Analyse financière
- Télécharge les performances boursières de :
  - MSFT, GOOGL, AMZN
  - META
  - NVDA, AMD
  - ASML, AVGO
- Compare leurs performances à 1 an au S&P 500
- Calcule :
  - croissance IA moyenne
  - surperformance vs benchmark
  - score par groupe
  - score global IA

---

### Analyse de sentiment (news)
Si tu fournis une clé NewsAPI :
- Analyse des news IA mondiales
- Analyse par entreprise
- Transformation en **NewsScore** (0 → 100)
- Intégration dans le score final

---

### Classification automatique
Chaque exécution produit :

| Signal | Signification |
|--------|----------------|
| 🟢 | Zone favorable |
| 🟡 | Neutre / plateau |
| 🔴 | Stress / prudence |
| ▲ | Score en amélioration |
| ▼ | Score en baisse |
| ▶ | Stable |

---

### Statut macro automatique

Exemple :

- "Cycle IA fort / haussier"
- "IA en normalisation"
- "IA en stress"
- "Cycle positif mais lent"

---

### Recommandation long terme

Tu reçois une phrase de synthèse, par exemple :

> Cycle IA modérément haussier : ne rien faire de spécial, laisser tourner ton plan automatique.

ou

> Zone de stress IA : n’ajouter que progressivement, éviter toute décision émotionnelle.

---

### Historique et tendance
À chaque lancement :
- Enregistre les scores dans `ai_watch_history.csv`
- Compare automatiquement au dernier snapshot
- Génère des alertes si :
  - baisse brutale (> 15 points)
  - euphorie (> 85)
  - zone de danger (< 55)

---

## 🔧 Installation

### 1. Dépendances

```bash
pip install yfinance pandas requests
