# DEMIXING

## Group:

- ####  Giorgio Magalini &nbsp;([@Giorgio-Magalini](https://github.com/Giorgio-Magalini))<br> codice persona Giò &nbsp;&nbsp; giorgio.magalini@mail.polimi.it

- ####  Alessandro Manattini &nbsp;([@alessandromanattini](https://github.com/alessandromanattini))<br> codice Ale &nbsp;&nbsp; alessandro.manattini@mail.polimi.it

- ####  Filippo Marri &nbsp;([@filippomarri](https://github.com/filippomarri))<br> 10110508 &nbsp;&nbsp; filippo.marri@mail.polimi.it

## Checklist

### Macrotasks
1.	Preparare (un tot di secondi per ogni traccia) e caricare il database ✅
2.	Mixaggio a 0,25 per tutte le stems ed eventuale confronto con mixture ✅
3.	Analisi del predittore Oracle e prove of concept: dimostrare che alzando il volume (coefficiente) di una stem rispetto alle altre, si ottiene un risultato migliore in fase di estrazione della stem stessa. 🔄
4.	Capire come modificare la schedule nel modello
5.	Scegliere la schedule che ci piace di più e dare una motivazione.



### Microtasks Completate
- Implementazione ⁠SDR, SIR, SAR
- Riscrittura del codice come Jupyter Notebook
- ⁠Taratura della soglia che definisce l’energia necessaria affinché un chunk sia considerato non silenzioso
- ⁠Valutazione dell’overlap: inserirlo o evitarlo?

### Microtasks Da fare

- ⁠Risolvere problemi con mps o cuda per scheda grafica
- ⁠Ottimizzare bss_eval_sources sostituendola con eval_mus. Problemi: la prima richiede troppo tempo per essere eseguita, la seconda prende in ingresso anche window_length e hop_size che vanno tarati
- ⁠l’SDR non è abbastanza alto (se la differenza non è 0 i risultati hanno senso,  se invece lo è perché sono così bassi?)

- ⁠Provare altri gain per vedere quali ottimizzano i risultati dell'algoritmo
- ⁠Verificare che la funzione find_non_silent_segment funzioni effettivamente facendo la print() della waveform e controllando che nei chunk non ci saino effettivamente parti silenziose
- ⁠Rendere iterativa la procedura di aggiornamento dei gain da mandare in ingresso alla rete di demixing

