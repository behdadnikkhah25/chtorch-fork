# Lag en ny klasse EnsembleEstimator
# med samme interface som Estimator. Inni train må du trene og evalere forskjellige configurereinger av estimator og lagre scores
#I predict må du predicte med hver av configured estimators. Og returnere en weighted average av prediksjoner
# Lag en test som kjoerer nye klassen slik at du kan bruke debuggern
# EnsembleEstimator boer ligger i ensemble_estimator.py på samme nivaa som estimator.py