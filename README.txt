This file contains

	- Task-C.py
		- it contains the code implementation for SubTask-C to find hypernyms and synonyms pair given a text.

	- TaskA_un_super.py
		- it contains the code implementation for SubTask-A (Unsupervised) to extract keyphrases from a text.
	- LSTM-model.py
		- it contains the code implementation for SubTask-A & SubTask-B
		 to extract keyphrases and classify them respectively.
		- Download the data and checkpoint/LSTM from the link Below.
		- For running SubTask-B: comment dataloader from subtask-A and uncomment for subtask-B and vice-verca

	- BERT.py
		- it contains the code implementation for SubTask-A & SubTask-B
		 to extract keyphrases and classify them respectively.
		 - Download the data and checkpoint/BERT and pretrained bert model from the link Below.
		- For running SubTask-B: just change the data_dir under paramets to './data/task2/'.
		For running experiments on sci_bert: just uncomment bert_model_dir = './model_sci/' and comment above command under paramets 
		


	- ALl the code are originally implemented in notebook.

	- 1. Theory Questions.pdf
		- contains answers for the theory questions

	- 2. Report.pdf
		- contains full discription of task,Experiments and Results with analysis.

	- Data, model weights file can be accessed using this link
		- https://drive.google.com/drive/folders/1kNcQVjWE97VN9gyVx1Diym-nf5ggMJW8?usp=sharing