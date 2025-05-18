# Copyright Labeller
This is a small tool to automatically assign labels according to the visible copyright to locations for a given geoguessr map `.json`.

This is not 100% accurate, as it's using computer vision to extract the copyright watermark from google street view images.
It will work pretty well on locations with a clear sky, because the grey copyright watermark text has enough contrast relative
to the sky to be recognizable to the OCR algorithm.

In my experiments, it found a copyright for approximately 2/3 of locations, and out of the assigned labels, about 95% were correct.
