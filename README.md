# DEMIXING

## Group:

- ####  Giorgio Magalini &nbsp;([@Giorgio-Magalini](https://github.com/Giorgio-Magalini))<br> codice persona Giò &nbsp;&nbsp; giorgio.magalini@mail.polimi.it

- ####  Alessandro Manattini &nbsp;([@alessandromanattini](https://github.com/alessandromanattini))<br> codice Ale &nbsp;&nbsp; alessandro.manattini@mail.polimi.it

- ####  Filippo Marri &nbsp;([@filippomarri](https://github.com/filippomarri))<br> 10110508 &nbsp;&nbsp; filippo.marri@mail.polimi.it

## Checklist

### Macrotasks
1.	Preparare (un tot di secondi per ogni traccia) e caricare il database ✅
2.	Mixaggio a 0,25 per tutte le stems ed eventuale confronto con mixture ✅
3.	Analisi del predittore Oracle e prove of concept: dimostrare che alzando il volume (coefficiente) di una stem rispetto alle altre, si ottiene un risultato migliore in fase di estrazione della stem stessa. ✅ *DA RIVEDERE*
4.	Capire come modificare la schedule nel modello 🔄
5.	Scegliere la schedule che ci piace di più e dare una motivazione fra queste [schedulesssss](https://arxiv.org/pdf/2206.00364).
6.	Si fanno i test con diverse schedule confrontandoli con DEMUCS chiamato una volta sola
7.	Si scrive il report

--------------------most valuable product--------------------

8.	In base a quanto tempo abbiamo e a che punto siamo si possono aggiungere, volendo, altre schedules, altre stems o, addirittura, condurre la stessa analisi su un altro modello.

### Microtasks Completate
- Implementazione ⁠SDR, SIR, SAR
- Riscrittura del codice come Jupyter Notebook
- ⁠Taratura della soglia che definisce l’energia necessaria affinché un chunk sia considerato non silenzioso
- ⁠Valutazione dell’overlap: inserirlo o evitarlo?
- ⁠Rendere iterativa la procedura di aggiornamento dei gain da mandare in ingresso alla rete di demixing
- Tentativo di sostituzione di bss_eval_sources con eval_mus. Lasciata bss_eval_sources nonostante l'elevato onere computazionale perché l'altra dà problemi di compatibilità
- prendere tutte le canzoni del dataset, applicare la funzione trim di librosa e salvare le tracce processate in una cartella locale in modo tale da prendere sempre i soliti primi trenta secondi sicuri che non ci sia silenzio né prima né dopo. Consiglia quindi di salvare tutto in una cartella creandoci un dataset personale.
- sostituire un mix create con gain omeogeni dagli stems alla mixture da usare come reference track.
- Controllare che i file audio abbiano tutti la stessa bitdepth. (NON IMPLEMENTATO MA CONTROLLATO DAL FINDER)

### Microtasks Da fare
- Controllare sui datasheet di DEMUCS la normalizzazione usata per l’allenamento del modello e utilizzare la stessa nel codice.
- Vanno normalizzate le stems per la new_mixture?
- Sositutire la funzione di mir_eval con quella che si trova a questi [link](https://lightning.ai/docs/torchmetrics/stable/audio/scale_invariant_signal_distortion_ratio.html)
- Utilizzare già una sorta di schedule e anziché fare un bar plot si raffigura un line plot con sdr dello stem target sull’asse delle y e numero di iterazioni sull’asse delle ascisse.
- Confronto fra gli SDR delle tracce estratte dichiarati da DEMUX ed i nostri (con gain uniformi a 0.25) in fase di valutazione del corretto funzionamento del modello.
- Implementare diverse schedules
- Per un’intuitiva rappresentazione dei risultati, ci consiglia di raffigurare l’SDR dell’oracle predictor come limite superiore e quello del DEMUCS con una sola passata come limite inferiore. I nostri risultati saranno (o ci aspettiamo che siano) collocati all’interno di questo intervallo.

## Branches
- main: codice funzionante ma che gira su CPU
- test_Filippo: codice che prova a girare su MPS


