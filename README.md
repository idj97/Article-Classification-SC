# Article-Classification-SC

**Tim:**
Djordje Ivkovic, SW54-2016

**Definicija problema:**
Odredjivanje kojoj klasi/sekciji clanak iz online vesti pripada. U ovom projektu cu klasifikovati vesti u jednu od 5 kategorija (business, entertainment, politics, sport, tech). Koristicu vise pristupa i na kraju ih uporediti kako bih izabrao najbolji.

**Skup podataka:**
Koristicu BBC news dataset koji se moze naci na: http://mlg.ucd.ie/datasets/bbc.html
Dataset je podeljen u 5 kategorija (business, entertainment, politics, sport, tech) i ima ukupno
2225 dokumenata.
Skup podataka ce biti podeljen na skup za trening, validaciju i testiranje.

**Metodologija:**
1. Preprocesiranje podataka tj dokumenata (ignorisanje raznih znakova i stop reci, stemming)
2. Kreiranje vokabulara koji se koristi za numericko predstavljanje dokumenata (counting, frequency, tl-idf, word hashing tj razne **bag-of-words** tehnike)
3. Treniranje vise ML modela ciji ce se rezultati uporedjivati. U ovom trenutku nisam siguran koje bih modele definitivno probao, verovatno ce to biti (SVM, KNN, Random Forests, Neural Network)

**Evaulacija:**
Za biranje najboljeg kandidata (tehnika za kreiranje vokabulara + ML model) birace se accuracy nad testnim skupom.
