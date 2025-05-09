#  Progressive Inference for Music Demixing

Using denoising diffusion approaches to train music demixing (MDX) models is promising but requires retraining large and carefully tuned neural networks (Plaja-Roglans,2022). Instead, we will explore a related yet different approach: can we improve separation quality solely by scheduling the inference process using a diffusion-inspired strategy even without retraining? By experimenting with existing MDX models (Spleeter by Deezer, Meta’s Demucs, ByteDance’s BS-Roformer, etc.), this project focuses on an exciting opportunity to explore and possibly enhance the performance of state-of-the-art AI techniques.

## Group:

- ####  Giorgio Magalini &nbsp;([@Giorgio-Magalini](https://github.com/Giorgio-Magalini))<br> 10990259 &nbsp;&nbsp; giorgio.magalini@mail.polimi.it

- ####  Alessandro Manattini &nbsp;([@alessandromanattini](https://github.com/alessandromanattini))<br> 11006826 &nbsp;&nbsp; alessandro.manattini@mail.polimi.it

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
- sostituire un mix create con gain omogeneo dagli stems alla mixture da usare come reference track.
- Controllare che i file audio abbiano tutti la stessa bitdepth. (NON IMPLEMENTATO MA CONTROLLATO DAL FINDE: siccome passiamo da 16 bit a 32 bit non ci sono problemi. Come sappiamo infatti, il quantisation noise viene introdotto quando si passa da valori più alti a valori più bassi di bit-depth)
- Vanno normalizzate le stems per la new_mixture? No perché la variazione di SDR in funzione della differenza di volume dei segnali è trascurabile nella modalità attuale di valutazione.
- Sostituire la funzione di mir_eval con quella che si trova a questi [link](https://lightning.ai/docs/torchmetrics/stable/audio/scale_invariant_signal_distortion_ratio.html)
- Controllare sui datasheet di DEMUCS la normalizzazione usata per l’allenamento del modello e utilizzare la stessa nel codice. Durante la costruzione dei metadati nel metodo build_metadata, se normalize è impostato su True, viene calcolato il valore RMS dell'intera traccia per ciascuna sorgente. Questi valori RMS vengono poi utilizzati per normalizzare i segmenti audio durante il training. La normalizzazione in DEMUCS viene quindi effettuata a livello di dataset, utilizzando i valori RMS precomputati delle tracce complete. <mark> Questo approccio garantisce che le variazioni di ampiezza tra le diverse tracce non influenzino negativamente il processo di apprendimento del modello. (con normalizzazione peggiora di 1‰).
Per quanto riguarda il dataset creato si è preferito normalizzare le tracce a 1 perché con l'RMS distorce. Senza normalizzazione il volume sarebbe troppo basso. </mark>
- Confronto fra gli SDR delle tracce estratte dichiarati da DEMUCS ed i nostri (con gain uniformi a 0.25) in fase di valutazione del corretto funzionamento del modello. 
    - SDR ≈ 5-10 dB per separazioni meno selettive (traccia others) (ad esempio, separazione di voce da accompagnamento musicale).
    - SDR ≈ 10-20 dB per separazioni più accurate (come la separazione di basso, batteria, e voce da un mix complesso).
- Utilizzare già una sorta di schedule e anziché fare un bar plot si raffigura un line plot con sdr dello stem target sull’asse delle y e numero di iterazioni sull’asse delle ascisse.
- Implementare diverse schedules
- Rimettere a posto nel file word la schedule lineare (ci va messa quella vera e non quella costante).
- Implementare funzione che esegue un wiener filtering stereo

### Microtasks Da fare
- Andare avanti sul modello proposto dal professore (voce da aggiungere al mix). Il nostro può anche essere cancellato.
- Ridefinire la schedule in modo più elegante come fatto sui paper. Si definisce un fattore β e, conseguentemente, il gain dell’accompagnamento come (1-β) e quello della voce come β. Prendere come esempio lo pseudocodice scritto dal professore nel PDF del 6 Maggio. Va rimessa a posto nella funzione che avevamo creato.
- Controllare dal codice il max della waveform post mix e, con l'orecchio, se i file wav salvati sono distorti una volta ricreato il mix (non con Ipython).
- Implementazione del WF secondo due diverse modalità (dettagli nel PDF del 6 Maggio):
    1.	Alla fine delle iterazioni (come nel paper)
    2.	Per ogni iterazione: ogni step di inferenza può introdurre noise che viene contrastato con la tecnica del WF del paper.
- Per un’intuitiva rappresentazione dei risultati, ci consiglia di raffigurare l’SDR dell’oracle predictor come limite superiore e quello del DEMUCS con una sola passata come limite inferiore. I nostri risultati saranno (o ci aspettiamo che siano) collocati all’interno di questo intervallo.
- Controllare l'overlapping nella funzione separate_sources.


## Branches
- main: codice su cui lavoriamo
- test_Filippo: codice non funzionante che può essere cancellato

