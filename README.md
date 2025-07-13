#  Progressive Inference for Music Demixing

Isolating individual sources from a musical mixture remains a key challenge in audio engineering, with wide-ranging applications from production to restoration. This work investigates whether a progressive inference strategy, inspired by diffusion models, can enhance the performance of an existing demixing network such as HDemucs. The proposed method iteratively refines the target stem by remixing the previous estimate into the original mixture using a scheduled weighting, and reapplying the separation at each step. As a preliminary validation, we demonstrate that increasing the relative gain of a stem in the mixture improves its separation quality—a lemma referred to as the “oracle predictor implication”—thereby justifying the progressive approach. The system is evaluated on the MUSDB18-HQ dataset using scale-invariant signal-to-distortion ratio (SI-SDR) as the metric, and tested under different update strategies and configurations. While early iterations show slight improvements for the target source, the performance plateaus and eventually declines, suggesting that fixed schedules and lack of model adaptation limit the effectiveness of the approach. The findings indicate that simply reusing a pre-trained separator in an iterative loop is insufficient, and that meaningful gains could be achieved by incorporating noise modeling, adaptive blending, and end-to-end retraining within a diffusion-based framework.

## Group:

- ####  Giorgio Magalini &nbsp;([@Giorgio-Magalini](https://github.com/Giorgio-Magalini))<br> 10990259 &nbsp;&nbsp; giorgio.magalini@mail.polimi.it

- ####  Alessandro Manattini &nbsp;([@alessandromanattini](https://github.com/alessandromanattini))<br> 11006826 &nbsp;&nbsp; alessandro.manattini@mail.polimi.it

- ####  Filippo Marri &nbsp;([@filippomarri](https://github.com/filippomarri))<br> 10110508 &nbsp;&nbsp; filippo.marri@mail.polimi.it

## How to run the code

Two codes are provided in the folder:
>[L12.ipynb](L12.ipynb): this is the Jupyter Notebook to be run to replicate the results.

>[standard_SDR_ealuation.py](standard_SDR_ealuation.py): this script is created separate in order not to overload the kernel. It computes the average standard SDR of all the songs contained in the test set without any pre-processing.