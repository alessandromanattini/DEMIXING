# DEMIXING - Test-Filippo Branch

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
5.	Scegliere la schedule che ci piace di più e dare una motivazione fra queste [schedulesssss](https://arxiv.org/pdf/2206.00364).



### Microtasks Completate
- Implementazione ⁠SDR, SIR, SAR
- Riscrittura del codice come Jupyter Notebook
- ⁠Taratura della soglia che definisce l’energia necessaria affinché un chunk sia considerato non silenzioso
- ⁠Valutazione dell’overlap: inserirlo o evitarlo?
- ⁠Rendere iterativa la procedura di aggiornamento dei gain da mandare in ingresso alla rete di demixing
- Tentativo di sostituzione di bss_eval_sources con eval_mus. Lasciata bss_eval_sources nonostante l'elevato onere computazionale perché l'altra dà problemi di compatibilità

### Microtasks Da fare
- Controllare la procedura iterativa. Problema: controllare il comportamento delle ultime due canzoni, una ha valori negativi che scendono al posto di salire e e l’altra ha valori che salgono e scendono
- ⁠Risolvere problemi con mps o cuda per device di esecuzione ausiliario
- controllare come mai ⁠l’SDR non è abbastanza alto per quanto riguarda la macrotask 2 (se la differenza non è 0 i risultati hanno senso,  se invece lo è perché sono così bassi?)
- ⁠Verificare che la funzione find_non_silent_segment funzioni effettivamente facendo la print() della waveform e controllando che nei chunk non ci saino effettivamente parti silenziose
- Controllo Git


