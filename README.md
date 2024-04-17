# xtream AI Challenge

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* Your understanding of the data
* The clarity and completeness of your findings
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation
---

## How to run

Para correr los scripts de este repositorio se debe tener instalado Python en su versión 3.10.10 y Pyenv para manejar environments. [Aquí](https://realpython.com/intro-to-pyenv/) también puede encontrarse una guía sobre la instalación.

Una vez instalado, coloquese en la carpeta de ´xtream-ai-assignment-engineer´.

Si usted está trabajando en Linux, corra el script ["build_env_linux.sh"](build_env_linux.sh) con el siguiente comanto en la terminal: `.\build_env_linux.sh`, el cual se encargará de crear el environment con los requirements necesarios para correr los scripts. De trabajar en Windows, corra el script análogo ["build_env_windows.sh"](build_env_windows.ps1) con el siguiente comanto en la terminal: `.\build_env_windows.ps1`.

### Diamonds

**Problem type**: Regression

**Dataset description**: [Diamonds Readme](./datasets/diamonds/README.md)

Primer paso: EDA. Por qué? Qué se hizo? Path a la notebook del EDA

#### Challenge 1

Path a la notebook del entrenamiento.
Qué variaciones se realizaron? Por qué esos modelos? Qué resultados se obtuvieron path a ese archivo. Cuáles son las conclusiones?

#### Challenge 2

**Assignment**: Develop an automated pipeline that trains your model with fresh data,
keeping it as sharp as the diamonds it assesses.

#### Challenge 3

**Assignment**: Build a REST API to integrate your model into a web app,
making it a cinch for his team to use.



#### Challenge 4

**Assignment**: Using your favorite cloud provider, either AWS, GCP, or Azure, design cloud-based training and serving pipelines.
You should not implement the solution, but you should provide a **detailed explanation** of the architecture and the services you would use, motivating your choices.

![alt text](./images/ServicesAndArchitecture.png)

The pipeline will start with data ingestion, provided by a front end that will load the files by communicating through a service hosted on Cloud Run. This service serves not only for updating the inventory but also for predicting the final price in real time containing the whole code with Docker containers and linking it's developing with multiple services like Cloud Build, Artifact Registry and Github. Cloud Run will be responsible for processing, cleaning, transforming the files into the required input format for storage and prediction. Alongside with Cloud SQL it will update de database and mantain it.

Storage will be done through BigQuery since the dataset Diamonds presents tabular and structured data. Through this, queries can be made. To analyze the data, prepare it for training, and save the prediction results, as well as a label provided in the iterative model tuning phase by a reviewer.

Within the inference part, Cloud Run will make a request to Vertex endpoints where the initial model will be deployed to make predictions according to the entered data. The endpoint will make the prediction and return the value to the same container in Cloud Run, which will communicate with a web application, the front end, to display the predicted price and obtain feedback from the reviewer during the iteration phase. Both the feedback and the prediction will be stored within the table architecture in BigQuery.

An automation will be scheduled with Cloud Scheduler every certain period (to be determined) for regular model training with the data loaded through the system and the feedback obtained from the reviewer. This will impact the model training and management service, which is composed first of an initial container in the Artifact Registry that will be used to build the Vertex AI Training service for the XGBoost model. The artifacts generated in training, both the model and the metadata, will be stored in Model Artifacts. From the best model, a new version of the model will be built in Vertex Model Registry and then redeployed.

Finally, we have two components. The first is a database visualization to provide users with information on model results and diamond characteristics using Looker dashboards. On the other hand, we have the monitoring and logging module. Through the Quality evaluator, the quality of predictions from the deployed model can be evaluated. Cloud Logging, Monitoring, and Error Reporting services will be essential for maintaining training pipelines, data loading, and prediction, diagnosing errors, and performance issues in the model.