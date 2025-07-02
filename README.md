# Sea Animal Classification with CNN & ResNet

## Περιγραφή
Το project υλοποιεί ταξινόμηση εικόνων θαλάσσιων ζώων χρησιμοποιώντας:
- Δύο custom CNN μοντέλα (CNN01 & CNN02)
- Transfer Learning με ResNet18

## Δομή του Repo

- `src/` : Όλος ο κώδικας της εργασίας (μοντέλα, train, dataset, evaluation)
- `slides/` : Η παρουσίαση της εργασίας σε μορφή .pptx
- `outputs/` : Checkpoints και αποτελέσματα από την αξιολόγηση
- `notebooks/` : Exploratory Data Analysis (EDA)
- `experiments.csv` : Τα αποτελέσματα όλων των runs
- `README.md` : Αυτή η περιγραφή

## Google Colab Notebook
Το script της εργασίας είναι διαθέσιμο σε μορφή Colab:

https://colab.research.google.com/drive/13-W0WvcVGKc-I15UlMUwi2A4AMgdNxpw?usp=sharing

## Αποτελέσματα
Δοκιμάστηκαν 3 μοντέλα με διαφορετικούς συνδυασμούς hyperparameters.  
Δείτε τα τελικά αποτελέσματα στο `experiments.csv` και στην παρουσίαση.
Η τελική αξιολόγηση έγινε στο test set χρησιμοποιώντας το καλύτερο μοντέλο (ResNet18).

## Παρουσίαση
Η παρουσίαση του project βρίσκεται στο φάκελο `slides`:

[`Final_Presentation_CNN_ResNet_SeaAnimals.pptx`](slides/Final_Presentation_CNN_ResNet_SeaAnimals.pptx

## ✍️ Ομάδα
- Γιουλάτου Άννα Βασιλική
- Κατσούλη Καλλιόπη
