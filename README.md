# Data Science Lab 2023: Group 5 Targaryen 🐉
This repository contains our project of the phase 2 of the **Practical Course: Data Science for Scientific Data** at Karlsruhe Institute of Technology (KIT).

## Group Members 👤 
| Forename | Surname  | Matr.#  |
|----------|----------|---------|
| Nina     | Mertins  | 2107539 |
| Kevin    | Hartmann | 1996265 |
| Alessio  | Negrini  | 2106547 |

## Folder Structure 🗂️
```
📦phase-2
 ┣ 📂config                    <-- Configuration files for the pipeline
 ┣ 📂data                      <-- Data used as input during development with Jupyter notebooks. 
 ┃ ┣ 📂predictions             <-- Contains the predicted data build during development.
 ┃ ┣ 📂raw                     <-- Contains the raw data provided by the supervisors.
 ┃ ┗ 📂processed               <-- Contains the processed data build during development.
 ┣ 📂models                    <-- Saved models during Development.
 ┣ 📂notebooks                 <-- Jupyter Notebooks used in development.
 ┃ ┗ 📂weekXX                  <-- Contains the notebooks for weekly subtasks and experimenting.
 ┣ 📂src                       <-- Source code.
 ┃ ┣ 📜TBD                     <-- TBD
 ┣ 📜.gitignore                <-- Specifies intentionally untracked files to ignore when using Git.
 ┣ 📜README.md                 <-- The top-level README for developers using this project. 
 ┣ 📜main.py                   <-- Main function to execute pipeline   
 ┗ 📜requirements.txt          <-- The requirenments file for reproducing the environment, e.g. generated with 
                                    'pip freeze > requirements.txt'.
```

## Setting up the environment and run the code ▶️

1. Clone the repository by running the following command in your terminal:

   ```
   https://git.scc.kit.edu/data-science-lab-2023/group-5-targaryen/phase-2.git
   ```


2. Navigate to the project root directory by running the following command in your terminal:

   ```
   cd phase-2
   ```

3. [Optional] Create a virtual environment and activate it. For example, using the built-in `venv` module in Python:
   ```
   python3 -m venv venv-phase-2
   source venv-phase-2/bin/activate
   ```

4. Install the required packages by running the following command in your terminal:

   ```
   pip install -r requirements.txt
   ```

5. Place the data in the phase-2/data/raw folder. Ensure that the data is in the appropriate format and structure 
required by the pipeline.

6. Run the pipeline with the following command:

   ```
   python3 main.py --config "configs/config.yml"
   ```

By following these steps, you should be able to successfully run the  pipeline on the data and obtain the desired 
results. You can also monitor the pipeline's progress through the logs printed in the terminal. If any errors or issues 
occur, the logs will provide valuable information for troubleshooting. 
