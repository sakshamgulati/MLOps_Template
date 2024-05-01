This project provides a comprehensive guide for implementing an end-to-end ML Lifecycle using Continuous Integration and Continuous Deployment (CICD) in the development of ML products. It encompasses several crucial aspects:

1. **Development and Experimentation:** The journey starts with rapid prototyping in Jupyter notebooks, followed by algorithm experimentation using Weights and Biases for iteration tracking. Internal modules facilitate data loading, preprocessing, model training, and deployment. While tailored to the project, a simplified example is provided in the repository to illustrate the approach, currently based on the Diabetes example, with more complex examples in progress.

2. **Tracking and Environment:** Pipenv is utilized to create virtual environments for trackable experiments.

3. **Production:** Metaflow is employed to create Directed Acyclic Graphs (DAGs) for deployment on cloud providers like AWS and GCP. Although the default deployment method is local, adjustments can be easily made in the `.metaflow` configuration files. Models are stored on the Model Registry (Weights and Biases) for inference.

4. **Continuous Integration/Continuous Deployment:** GitHub Actions and Metaflow scripts form our CICD pipeline, enabling model training on cost-free GitHub Ubuntu machines. Trained models can be seamlessly published in the model repository managed by Weights and Biases.

To train a different or improved model, the same scripts can be utilized for training and publishing within the same repository. However, transitioning to deploying the model in a production environment necessitates passing multiple unit tests and adhering to linting standards, facilitated by Pytest.

For further details, visit [here](https://www.notion.so/Machine-Learning-Ops-3983660707ff41fdaa37717bc8153140?pvs=4).

**Next Steps:**
1. Replace Diabetes Classification with a Time-Series Model, ensuring model retraining capability.
2. Incorporate Model Retraining and replacement with CICD.
3. Capture Model/Feature Drift using Evidently AI.
4. Shift Model to AWS.
5. Dockerization and shift model to K8s.

**To Do:**
- [ ] add setup instructions in the README.md
- [ ] add Metaflow setup instructions in the README.md 
- [ ] Delete unnecessary files and folders in the repository
- [ ] Add linter, unit tests, formatting (ex, black) in the CI/CD pipeline

**Inspiration:**  
Idea inspired by [Simon](https://github.com/simonprudhomme)
![MLOps's Image](https://github.com/sakshamgulati/MLOps_Template/assets/16202917/c175e03e-c753-474f-b6a5-17b792b297e2)
