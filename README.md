# DEMIXING

## Group:

- ####  Giorgio Magalini &nbsp;([@Giorgio-Magalini](https://github.com/Giorgio-Magalini))<br> codice persona Giò &nbsp;&nbsp; giorgio.magalini@mail.polimi.it

- ####  Alessandro Manattini &nbsp;([@alessandromanattini](https://github.com/alessandromanattini))<br> codice Ale &nbsp;&nbsp; alessandro.manattini@mail.polimi.it

- ####  Filippo Marri &nbsp;([@filippomarri](https://github.com/filippomarri))<br> 10110508 &nbsp;&nbsp; filippo.marri@mail.polimi.it

## Checklist

### Fatte
- Implementazione ⁠SDR, SIR, SAR
- Riscrittura del codice come Jupyter Notebook
- ⁠Taratura della soglia che definisce l’energia necessaria affinché un chunk sia considerato non silenzioso
- ⁠Valutazione dell’overlap: inserirlo o evitarlo?

### Da fare

- ⁠Risolvere problemi con mps o cuda per scheda grafica
- ⁠Ottimizzare bss_eval_sources sostituendola con eval_mus. Problemi: la prima richiede troppo tempo per essere eseguita, la seconda prende in ingresso anche window_length e hop_size che vanno tarati
- ⁠l’SDR non è abbastanza alto (se la differenza non è 0 i risultati hanno senso,  se invece lo è perché sono così bassi?)

- ⁠Provare altri gain per vedere quali ottimizzano i risultati dell'algoritmo
- ⁠Verificare che la funzione find_non_silent_segment funzioni effettivamente facendo la print() della waveform e controllando che nei chunk non ci saino effettivamente parti silenziose
- ⁠Rendere iterativa la procedura di aggiornamento dei gain da mandare in ingresso alla rete di demixing

