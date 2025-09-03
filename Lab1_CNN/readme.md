# Lab1_CNN

Questo progetto implementa una rete convoluzionale profonda (CNN) per compiti di classificazione di immagini, basata sull'architettura ResNet-18 e sulla tecnica di Class Activation Maps (CAM) per l'interpretabilità del modello. L'obiettivo è sfruttare un modello pre-addestrato ResNet-18 per estrarre caratteristiche efficaci e utilizzare CAM per visualizzare le aree di interesse del modello durante le predizioni.

## Architettura

- **ResNet-18**: Il modello utilizza l'architettura ResNet-18, introdotta da He et al. (2016), che ha rivoluzionato il training di reti molto profonde tramite blocchi residuali. ResNet-18 è composta da 17 layer convoluzionali, un max pooling layer e uno strato fully connected. La caratteristica chiave è l'uso di connessioni skip (residual connections) che mitigano il problema del gradiente che svanisce, consentendo un training più stabile e profondo [He et al., 2016].[^1]
- **Class Activation Maps (CAM)**: La tecnica CAM consente di localizzare le regioni dell'immagine che contribuiscono maggiormente alla decisione del modello, migliorando l'interpretabilità. CAM calcola una mappa di attivazione pesata delle feature maps estratte dall'ultimo layer convoluzionale tramite i pesi del layer fully connected. Introdotto da Zhou et al. (2015), CAM è fondamentale per la visualizzazione delle regioni discriminanti del modello [Zhou et al., 2015; Lee et al., 2021].[^2][^3]


## Dataset

Il modello può essere addestrato e testato su dataset di immagini comunemente usati per classificazione, come:

- **ImageNet**: Ampio dataset di immagini con 1000 classi, tipicamente usato per il pretraining dei modelli ResNet.
- Altri dataset di dominio specifico (medicale, industriale, etc.) possono essere adottati a seconda dell'applicazione.


## Installazione

1. Clonare il repository:

```
git clone https://github.com/lucrezio001/Deep-Leaning-Application-Luca-Capece.git
cd Deep-Leaning-Application-Luca-Capece/Lab1_CNN
```

2. Installare le dipendenze Python:

```
pip install -r requirements.txt
```


## Uso

- Il training si basa su ResNet-18 pre-addestrato su ImageNet, con possibilità di fine-tuning sul dataset specifico.
- Per eseguire il training o la valutazione, utilizzare gli script Python presenti nella cartella.
- CAM è integrato per generare mappe di attivazione visualizzabili per ogni immagine classificata.


## Riferimenti

- K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. [DOI PDF](https://arxiv.org/abs/1512.03385)[^1]
- B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba, "Learning Deep Features for Discriminative Localization," arXiv preprint arXiv:1512.04150, 2015. [arXiv PDF](https://arxiv.org/abs/1512.04150)[^3]
- H. Lee et al., "Relevance-CAM: Your Model Already Knows Where To Look," in *CVPR 2021*, 2021. [PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Relevance-CAM_Your_Model_Already_Knows_Where_To_Look_CVPR_2021_paper.pdf)[^2]
