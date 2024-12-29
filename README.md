# Assignment 1: Strumento per la navigazione di immagini a 360°


## Obiettivo
L'obiettivo di questo assegnamento è creare uno strumento per navigare immagini catturate da camere a 360° (immagini equirettangolari).  

## Funzionalità principali
- **Input:**  
  - Un'immagine (o video) catturata da una camera a 360°.  
  - Una vista iniziale espressa in latitudine e longitudine (in gradi).  
  - Un campo visivo (FOV) espresso in gradi (default: 60°).  
- **Output:**  
  - Visualizzazione dell'immagine rettificata proiettata su un piano tangente alla sfera nella vista iniziale.  
- **Navigazione:**  
  - L'utente può modificare la vista iniziale utilizzando, ad esempio, i tasti freccia (sinistra/destra/su/giù) per cambiare la vista di un numero discreto di gradi predefiniti.  
- **Zoom digitale:**  
  - Opzionale: implementare uno zoom digitale per mettere a fuoco oggetti vicini o lontani.  

## Restrizioni
- È consentito utilizzare solo codice sviluppato autonomamente (non si possono usare librerie che eseguono automaticamente la proiezione).  
- Lo strumento deve funzionare indipendentemente dalla dimensione dell'immagine/frame.  


---

# Assignment 2: Rilevamento Pedoni

## Obiettivo
L'obiettivo di questo assegnamento è addestrare un rilevatore di pedoni basato su HOG (Histogram of Oriented Gradients) e SVM (Support Vector Machine) utilizzando:  
- OpenCV per l'estrazione delle caratteristiche HOG.  
- Scikit-learn per l'addestramento di un SVM lineare.  

## Dataset
Il rilevatore deve essere addestrato, validato e testato utilizzando il dataset **WiderPerson** (http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/).  

## Requisiti principali
1. **Preparazione dei dati:**  
   - Positivi: Usa i bounding box della classe "pedestrian" (classe 1) annotati nei file `.txt` forniti.  
   - Negativi: Usa finestre casuali estratte da un file scaricabile fornito (vedi link nell'assegnamento).  
   - Ridimensiona le immagini a `64x128` pixel per estrarre i descrittori HOG con `cv2.HOGDescriptor()`.  
2. **Pipeline attesa:**  
   - Preparazione dei dati (crop e ridimensionamento).  
   - Estrazione dei descrittori HOG.  
   - Addestramento di un SVM lineare con Scikit-learn (`svm.LinearSVC`).  
   - Implementazione di una procedura multi-scala con sliding window e stride per rilevare pedoni a diverse scale.  
   - Implementazione di una procedura di soppressione delle non-massime (NMS) basata su una soglia IoU (`T`).  
3. **Metriche di valutazione:**  
   - Usa `TP`, `FP`, `FN` per valutare le prestazioni (IoU > 0.5 è considerato un rilevamento corretto).  
   - Calcola precisione e richiamo.  
4. **Confronto con il rilevatore predefinito di OpenCV:**  
   - Usa il rilevatore predefinito `cv2.HOGDescriptor_getDefaultPeopleDetector()`.  
   - Confronta le prestazioni con il tuo modello.  

---

# Assignment 3: Tracciamento Pedoni

## Obiettivo
L'obiettivo è sviluppare un Multi-Object Tracker online basato sul paradigma **tracking-by-detection**, con focus sul tracciamento di pedoni.  

## Componenti principali
1. **Rilevatore di pedoni:**  
   Utilizza il modello DETR (*End-to-End Object Detection with Transformers*).  
   - Installa `torch` e `torchvision`.  
   - Crea il modello con i pesi pre-addestrati:  
     ```python
     model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
     ```  
2. **Gestione dei tracciamenti:**  
   - Implementa una strategia per aggiungere nuovi tracciamenti quando nuovi pedoni entrano nella scena.  
   - Rimuovi tracciamenti quando i pedoni lasciano la scena.  
3. **Associazione dei tracciamenti:**  
   - Usa il metodo di *maximum bipartite matching* per associare i bounding box rilevati alle identità conosciute.  
   - Implementa il metodo usando `scipy.optimize.linear_sum_assignment`.  
   - Definisci una metrica di distanza personalizzata (puoi utilizzare le uscite del trasformatore di DETR o una rete pre-addestrata).  

## Dataset
Usa il dataset **MOT17** (https://motchallenge.net/data/MOT17/) per validare e testare il metodo.  

## Metriche di valutazione
- Calcola le metriche **HOTA** e **CLEAR MOT** usando il framework **TrackEval** (https://github.com/JonathonLuiten/TrackEval).  
- Valuta il modello su un set di validazione per selezionare i migliori iperparametri (come soglie). Usa questi stessi iperparametri per il test set.  
- Riporta i risultati medi sulle metriche per diversi valori di iperparametri sul set di validazione.  
