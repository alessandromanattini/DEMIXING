# DEMIXING - Test-Filippo Branch

## Stato delle cose sul branch (04/04)
- ⁠la funzione per calcolare l'SDR con pyTorch funziona correttamente fino ad Analyze/SDR escluso
- provando a richiamarla in "very efficient evaluation of SDR using MPS" si ottengo risultati diversi rispetto a quello di mir_eval


## Prossimo step
- risolvere il problema in efficient evaluation of SDR using MPS

**Controlli effettuati:**
- durante la chiamata si passa lo stesso vettore alla funzione ✅
- la permutazione di mir_eval non influenza il risultato: modificando il parametro nella funzione della cella precedente per disattivarla si ottiene comunque lo stesso risultato ✅
- per sicurezza si è provato ad implementare un piccolo algoritmo in grado di fare la permutazione ma i risultati sono gli stessi ✅