# Social License to Operate Triple-Bottom-Line Topic Classification Project

## Project Vision:

The purpose of this research project is to assist in the evaluation of the Social License to Operate for organizations and other entities.  Eventually, we hope to be able to provide machine learning trained models that can predict with relative accuracy how accepted a project is in the eyes of the public and stakeholders via textual data obtained from social media outlets like Twitter.

&nbsp;

## Codebase Execution:

Run slo_topic_classification.py

### Description:

The entire project can be executed by running this file as of the current implementation.

Note: The import of the required datasets assumes a relative path from current working directory of :

- /tbl-datasets
- /borg-SLO-classifiers

Note: The datasets are NOT included in the GitHub repository and must be obtained separately.

- dataset_20100101-20180510_tok.csv (obtained from Borg supercomputer)
- tbl_training_set.csv (obtained from Professor VanderLinden)
- tbl_kvlinden.csv (obtained from Professor VanderLinden)

&nbsp;

The Python code will automatically import the datasets, pre-process, post-process, and tokenize the Tweet data.  Then, it will run various Classifiers and output metric summarizations for each one to the console output.

TODO - implement data visualizations.

TODO - tune hyper parameters for better training results and enhanced predictive ability.

TODO - clean the code base for style and readability purposes.

TODO - update this README.md as necessary to reflect updates to project.

&nbsp;

## GitHub Repositories:

SLO Research Repository:</p>
https://github.com/Calvin-CS/slo-classifier

&nbsp;

Joseph Jinn - CS 344 Repository:</p>
https://github.com/J-Jinn/cs344